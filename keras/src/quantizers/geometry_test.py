import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.layers.core.einsum_dense import EinsumAxes
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes


def _expert_slices(x):
    # `btei,eid->bted`: expert `e` sees `x[:, :, e, :]`.
    return np.stack(
        [x[:, :, e, :].reshape(-1, x.shape[-1]) for e in range(x.shape[2])]
    )


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
            ValueError, "Cannot derive a contraction view"
        ):
            layer._quantization_geometry().contraction_view()

    def test_axes_must_partition_the_kernel(self):
        with self.assertRaisesRegex(ValueError, "partition the kernel axes"):
            ContractionView(
                (2, 3, 4),
                KernelAxes(contracted=(0,), free=(1,)),
                input_batch_axes=(),
                input_contracted_axes=(-1,),
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
