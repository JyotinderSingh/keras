import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.layers.core.einsum_dense import EinsumAxes
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_test_utils import Pointwise1D
from keras.src.quantizers.quantization_test_utils import ThirdPartyProjection
from keras.src.quantizers.quantization_test_utils import assert_float


def _expert_slices(x):
    # `btei,eid->bted`: expert `e` sees `x[:, :, e, :]`.
    return np.stack(
        [x[:, :, e, :].reshape(-1, x.shape[-1]) for e in range(x.shape[2])]
    )


def _head_slices(x):
    # `btnh,nh->btn`: head `n` sees `x[:, :, n, :]`.
    return x.transpose(2, 0, 1, 3).reshape(x.shape[2], -1, x.shape[-1])


class ContractionViewTest(testing.TestCase):
    """The contraction view follows the equation, not the kernel shape."""

    @parameterized.named_parameters(
        (
            "qkv",
            "btd,dnh->btnh",
            (None, 2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (0, 1, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "attention_output",
            "btnh,nhd->btd",
            (None, 4),
            (None, 5, 2, 3),
            (1, 6, 4),
            (0, 1, 2),
            lambda x: x.reshape(-1, 6),
        ),
        (
            "gemma_q",
            "btd,ndh->btnh",
            (None, 2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (1, 0, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "expert_gate",
            "btd,edi->btei",
            (None, 3, 6),
            (None, 5, 4),
            (1, 4, 18),
            (1, 0, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "expert_down",
            "btei,eid->bted",
            (None, 3, 4),
            (None, 5, 3, 6),
            (3, 6, 4),
            (0, 1, 2),
            _expert_slices,
        ),
        (
            # The inputs carry the contracted axes in the other order: the
            # inputs view follows the kernel's `[n, h]`.
            "input_order_differs",
            "bhn,nhd->bd",
            (4,),
            (None, 2, 3),
            (1, 6, 4),
            (0, 1, 2),
            lambda x: x.transpose(0, 2, 1).reshape(-1, 6),
        ),
        (
            "ellipsis",
            "...d,dnh->...nh",
            (2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (0, 1, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "four_d",
            "abc,cdef->abdef",
            (3, 2, 3, 2),
            (None, 3, 4),
            (1, 4, 12),
            (0, 1, 2, 3),
            lambda x: x.reshape(-1, 4),
        ),
        (
            # A value head: the kernel has no free axis, so one column.
            "no_free_axis",
            "btd,d->bt",
            (None,),
            (None, 5, 4),
            (1, 4, 1),
            (0,),
            lambda x: x.reshape(-1, 4),
        ),
        (
            # A per-head gate: `n` is a batch axis and `h` contracted.
            "batch_axis_without_free_axis",
            "btnh,nh->btn",
            (None, 3),
            (None, 5, 3, 4),
            (3, 4, 1),
            (0, 1),
            _head_slices,
        ),
    )
    def test_einsum_view(
        self,
        equation,
        output_shape,
        input_shape,
        expected_dims,
        expected_permutation,
        expected_inputs_view,
    ):
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        view = layer._quantization_geometry().contraction_view()
        batch, rows, columns = expected_dims
        self.assertEqual((view.batch, view.rows, view.columns), expected_dims)
        self.assertEqual(view.kernel_permutation, expected_permutation)

        # The kernel view flattens the permuted axes and restores exactly.
        kernel_shape = tuple(layer.kernel.shape)
        kernel = np.arange(np.prod(kernel_shape), dtype="float32").reshape(
            kernel_shape
        )
        kernel_view = view.kernel_to_view(kernel)
        self.assertEqual(tuple(kernel_view.shape), (batch, rows, columns))
        self.assertAllClose(
            kernel_view,
            np.transpose(kernel, expected_permutation).reshape(
                batch, rows, columns
            ),
        )

        # The inputs view lines its rows up with the kernel view's rows.
        x = (
            np.random.default_rng(0)
            .standard_normal((2,) + tuple(input_shape[1:]))
            .astype("float32")
        )
        self.assertEqual(view.input_features(x), rows)
        self.assertAllClose(view.inputs_to_view(x), expected_inputs_view(x))

    def test_dense_view(self):
        layer = layers.Dense(6)
        layer.build((None, 4))
        view = layer._quantization_geometry().contraction_view()
        self.assertEqual((view.batch, view.rows, view.columns), (1, 4, 6))
        self.assertEqual(view.kernel_permutation, (0, 1))
        x = (
            np.random.default_rng(0)
            .standard_normal((2, 5, 4))
            .astype("float32")
        )
        self.assertEqual(view.input_features(x), 4)
        self.assertAllClose(view.inputs_to_view(x), x.reshape(-1, 4))
        kernel = np.arange(24, dtype="float32").reshape(4, 6)
        self.assertAllClose(
            ops.reshape(view.kernel_to_view(kernel), (4, 6)), kernel
        )

    @parameterized.named_parameters(
        # No axis of the kernel is contracted with the inputs.
        ("outer_product", "bt,nh->btnh", (None, 2, 3), (None, 5)),
        # The inputs are summed over `b` on their own.
        ("summed_input_axis", "abc,cd->ad", (4,), (None, 3, 8)),
    )
    def test_equation_without_a_view_raises(
        self, equation, output_shape, input_shape
    ):
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot derive a contraction view"
        ):
            layer._quantization_geometry().contraction_view()

    @parameterized.named_parameters(
        ("missing", (0,), (1,)),
        ("duplicated", (0, 1), (1, 2)),
        ("negative", (0,), (1, -1)),
        ("out_of_range", (0,), (1, 5)),
    )
    def test_axes_must_partition_the_kernel(self, contracted, free):
        kernel_axes = KernelAxes(contracted=contracted, free=free)
        with self.assertRaisesRegex(
            ValueError, r"partition the kernel axes.*rank-3.*kernel_axes="
        ):
            ContractionView(
                (2, 3, 4),
                kernel_axes,
                input_batch_axes=(),
                input_contracted_axes=(-1,),
            )


class KernelMatrixTest(testing.TestCase):
    @parameterized.named_parameters(
        ("matmul", "ab,bc->ac", (None, 8), (6,)),
        ("gemma_q", "btd,ndh->btnh", (None, 5, 8), (None, 2, 4)),
        ("expert_gate", "btd,edi->btei", (None, 5, 8), (None, 3, 4)),
        ("two_contracted", "abcd,cde->abe", (None, 4, 3, 5), (None, 6)),
        ("one_expert", "btei,eid->bted", (None, 5, 1, 7), (None, 1, 8)),
    )
    def test_matches_the_contraction_view_without_batch_problems(
        self, equation, input_shape, output_shape
    ):
        # int4 folds batch axes into the columns and GPTQ/AWQ stack them
        # along the rows; without batch problems the two layouts agree.
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        geometry = layer._quantization_geometry()
        kernel = (
            np.random.default_rng(0)
            .standard_normal(geometry.weight_shape)
            .astype("float32")
        )
        permutation, rows, columns = geometry.kernel_axes.matrix(
            kernel.shape, batch_in="columns"
        )
        view = geometry.contraction_view()
        self.assertEqual(view.batch, 1)
        self.assertAllEqual(
            np.reshape(np.transpose(kernel, permutation), (rows, columns)),
            np.reshape(
                ops.convert_to_numpy(view.kernel_to_view(kernel)),
                (view.rows, view.columns),
            ),
        )


class KernelAxesTest(testing.TestCase):
    def test_the_two_matrices_of_a_kernel_with_a_batch_axis(self):
        # `btei,eid->bted`: `e` is a batch axis, `i` contracted, `d` free.
        axes = KernelAxes(contracted=(1,), free=(2,), batch=(0,))
        self.assertEqual(axes.sizes((3, 4, 5)), (3, 4, 5))
        # GPTQ and AWQ stack the problems along the rows.
        self.assertEqual(
            axes.matrix((3, 4, 5), batch_in="rows"), ((0, 1, 2), 12, 5)
        )
        # int4 keeps the batch axis among the columns.
        self.assertEqual(
            axes.matrix((3, 4, 5), batch_in="columns"), ((1, 0, 2), 4, 15)
        )
        with self.assertRaisesRegex(ValueError, "batch_in"):
            axes.matrix((3, 4, 5), batch_in="free")

    def test_axes_are_stored_as_tuples(self):
        self.assertEqual(
            KernelAxes(contracted=[0, 1], free=[2]),
            KernelAxes(contracted=(0, 1), free=(2,)),
        )


class EinsumAxesTest(testing.TestCase):
    @parameterized.named_parameters(
        ("plain", "btd,ndh->btnh", 3, "btd", "btnh", (1,), (0, 2), ()),
        ("left_ellipsis", "...d,dh->...h", 4, "ABCd", "ABCh", (0,), (1,), ()),
        (
            "right_ellipsis",
            "ab...,bc->ac...",
            4,
            "abAB",
            "acAB",
            (0,),
            (1,),
            (),
        ),
        ("batch_axis", "btei,eid->bted", 4, "btei", "bted", (1,), (2,), (0,)),
    )
    def test_labels_and_roles(
        self, equation, rank, inputs, output, contracted, free, batch
    ):
        axes = EinsumAxes.from_equation(equation, rank)
        self.assertEqual((axes.inputs, axes.output), (inputs, output))
        self.assertEqual(axes.kernel_axes, KernelAxes(contracted, free, batch))
        # The input axes that carry the kernel's contracted labels are the
        # ones the equation reduces.
        self.assertEqual(axes.input_axes(contracted), axes.input_reduced_axes)
        self.assertEqual(
            axes.gradient_equation, f"{output},{axes.kernel}->{inputs}"
        )

    @parameterized.named_parameters(
        ("matmul", "ab,bc->ac", (None, 8), (6,), (None, 1), (1, 6)),
        (
            "qkv",
            "btd,dnh->btnh",
            (None, 5, 4),
            (None, 2, 3),
            (None, None, 1, 2),
            (1, 1, 2, 3),
        ),
        (
            "gemma_q",
            "btd,ndh->btnh",
            (None, 5, 4),
            (None, 2, 3),
            (None, None, 0, 2),
            (1, 1, 2, 3),
        ),
        (
            "batch_axis",
            "btei,eid->bted",
            (None, 5, 3, 7),
            (None, 3, 6),
            (None, None, 0, 2),
            (1, 1, 3, 6),
        ),
        # The free axis `g` has size one and keeps its own place.
        (
            "size_one_free_axis",
            "acd,gecd->aeg",
            (None, 4, 5),
            (6, 1),
            (None, 1, 0),
            (1, 6, 1),
        ),
    )
    def test_int8_scale_is_stored_in_the_outputs_layout(
        self, equation, input_shape, output_shape, scale_axes, scale_shape
    ):
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        geometry = layer._quantization_geometry()
        self.assertEqual(geometry.kernel_scale_axes, scale_axes)
        x = (
            np.random.default_rng(0)
            .standard_normal((2,) + tuple(input_shape[1:]))
            .astype("float32")
        )
        y = ops.convert_to_numpy(layer(x))
        # Weight-only, so the outputs depend on the stored scale alone.
        layer.quantize(
            "int8", config=Int8QuantizationConfig(activation_quantizer=None)
        )
        self.assertEqual(tuple(layer.kernel_scale.shape), scale_shape)
        self.assertAllClose(layer(x), y, atol=0.1 * np.abs(y).max())


class MinimalGeometry(ProjectionGeometry):
    """The `(1, in, out)` kernel of `Pointwise1D`, with three overrides."""

    @property
    def kernel_axes(self):
        return KernelAxes(contracted=(0, 1), free=(2,))

    def contract(self, inputs, kernel):
        return ops.einsum("btc,kcd->btd", inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        return ops.einsum("btd,kcd->btc", upstream, float_kernel)


class MinimalPointwise(Pointwise1D):
    def _quantization_geometry(self):
        return MinimalGeometry(self)


class ContractOnlyGeometry(ProjectionGeometry):
    """`MinimalGeometry` that keeps the 2-D default of `kernel_axes`."""

    contract = MinimalGeometry.contract
    contract_grad = MinimalGeometry.contract_grad


class NegativeAxesGeometry(MinimalGeometry):
    kernel_axes = KernelAxes(contracted=(-3, -2), free=(-1,))


class ExpertsGeometry(ProjectionGeometry):
    """A mixture-of-experts kernel `(experts, in, out)`.

    The inputs `(experts, tokens, in)` and the outputs
    `(experts, tokens, out)` lead with the expert axis.
    """

    @property
    def kernel_axes(self):
        return KernelAxes(contracted=(1,), free=(2,), batch=(0,))

    def contract(self, inputs, kernel):
        return ops.einsum("etc,ecd->etd", inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        return ops.einsum("etd,ecd->etc", upstream, float_kernel)


class ScaledExpertsGeometry(ExpertsGeometry):
    # The stored scale `(experts, 1, out)` lines up with the outputs.
    kernel_scale_axes = (0, None, 2)


class Experts(ThirdPartyProjection):
    """One `(in, out)` kernel per index of the first axis of the inputs."""

    def float_kernel_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], self.units)

    def _quantization_geometry(self):
        return ExpertsGeometry(self)


def _with_geometry(layer_cls, geometry_cls):
    """A subclass of `layer_cls` whose geometry is `geometry_cls`."""

    class Declared(layer_cls):
        def _quantization_geometry(self):
            return geometry_cls(self)

    return Declared


class CustomProjectionTest(testing.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", None, (6,)),
        ("int4_grouped", "int4", 4, (2, 6)),
        ("int4_per_channel", "int4", -1, (6,)),
    )
    def test_nd_kernel_describes_its_axes_once(
        self, mode, block_size, scale_shape
    ):
        # A 3-D kernel overrides `kernel_axes`, `contract` and
        # `contract_grad`. The stored layout of each mode, and the
        # calibration view, derive from `kernel_axes`.
        layer = MinimalPointwise(6, use_bias=False)
        layer.build((None, 5, 8))
        kernel = ops.convert_to_numpy(layer._kernel)
        view = layer._quantization_geometry().contraction_view()
        self.assertEqual((view.batch, view.rows, view.columns), (1, 8, 6))
        x = (
            np.random.default_rng(0)
            .standard_normal((2, 5, 8))
            .astype("float32")
        )
        if mode == "int8":
            config = Int8QuantizationConfig(activation_quantizer=None)
        else:
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        self.assertEqual(tuple(layer.kernel_scale.shape), scale_shape)

        # The view reads the stored scale in its layout, and the forward
        # pass uses the same weight.
        weight = layer._quantized_weight().dequantize("float32")
        self.assertAllClose(weight, kernel, atol=0.1 * np.abs(kernel).max())
        self.assertAllClose(
            layer(x),
            ops.einsum("btc,kcd->btd", x, weight),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.named_parameters(
        ("default_axes_int8", ContractOnlyGeometry, "int8"),
        ("default_axes_int4", ContractOnlyGeometry, "int4"),
        ("negative_axes_int8", NegativeAxesGeometry, "int8"),
        ("negative_axes_int4", NegativeAxesGeometry, "int4"),
    )
    def test_kernel_axes_must_fit_the_kernel(self, geometry_cls, mode):
        # `kernel_axes` left at the 2-D default for a 3-D kernel, or with
        # negative axes, is refused before the layer changes.
        layer = _with_geometry(Pointwise1D, geometry_cls)(6, use_bias=False)
        layer.build((None, 5, 8))
        kernel = ops.convert_to_numpy(layer._kernel)
        with self.assertRaisesRegex(
            ValueError, r"rank-3 kernel.*kernel_axes=KernelAxes"
        ):
            layer.quantize(mode)
        assert_float(self, layer)
        self.assertAllEqual(layer._kernel, kernel)

    def test_batch_axes_need_a_scale_axes_override(self):
        # As many tokens as experts, so a scale laid out for the wrong
        # axis would still broadcast. The default `kernel_scale_axes`
        # cannot place the expert axis in the outputs, so int8 refuses the
        # layer and leaves it as it was.
        x = (
            np.random.default_rng(0)
            .standard_normal((4, 4, 8))
            .astype("float32")
        )
        layer = Experts(6, use_bias=False)
        layer.build(x.shape)
        kernel = ops.convert_to_numpy(layer._kernel)
        with self.assertRaisesRegex(
            NotImplementedError, "must override `kernel_scale_axes`"
        ):
            layer.quantize("int8")
        assert_float(self, layer)
        self.assertAllEqual(layer._kernel, kernel)

        # With the outputs' layout given, int8 stays within the error
        # bound of the int8 rows of the conformance test.
        layer = _with_geometry(Experts, ScaledExpertsGeometry)(
            6, use_bias=False
        )
        layer.build(x.shape)
        y = ops.convert_to_numpy(layer(x))
        layer.quantize("int8")
        self.assertEqual(tuple(layer.kernel_scale.shape), (4, 1, 6))
        error = ops.convert_to_numpy(layer(x)) - y
        self.assertLess(np.linalg.norm(error) / np.linalg.norm(y), 0.05)

    def test_default_scale_axes_follow_the_kernel_axes(self):
        class Geometry(ProjectionGeometry):
            kernel_axes = KernelAxes(contracted=(1,), free=(2, 0))

        self.assertEqual(Geometry(None).kernel_scale_axes, (0, 2))
        self.assertEqual(ProjectionGeometry(None).kernel_scale_axes, (1,))

    def test_replaced_hook_is_refused(self):
        with self.assertRaisesRegex(
            TypeError, "`kernel_reduced_axes` with `kernel_axes`"
        ):

            class Stale(ProjectionGeometry):
                @property
                def kernel_reduced_axes(self):
                    return (0, 1)
