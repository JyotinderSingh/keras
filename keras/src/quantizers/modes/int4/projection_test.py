"""int4 stores an einsum kernel along its contraction, like `Dense`, and
takes its input gradient through the dequantized kernel."""

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing.test_utils import named_product

# Each equation with the kernel permutation that puts its contracted axes
# first and every other axis after them in the kernel's order, and the
# `(rows, columns)` of the resulting matrix.
EQUATIONS = [
    # Gemma's query projection `[heads, d_model, head_dim]`.
    dict(
        testcase_name="gemma_q",
        equation="btd,ndh->btnh",
        input_shape=(None, 5, 8),
        output_shape=(None, 2, 4),
        permutation=(1, 0, 2),
        dims=(8, 8),
    ),
    # A mixture-of-experts gate `[experts, d_model, inner]`.
    dict(
        testcase_name="expert_gate",
        equation="btd,edi->btei",
        input_shape=(None, 5, 8),
        output_shape=(None, 3, 4),
        permutation=(1, 0, 2),
        dims=(8, 12),
    ),
    # A mixture-of-experts down projection `[experts, inner, d_model]`:
    # the expert axis is shared by the inputs, so it joins the columns and
    # every column belongs to one expert. With groups of 3, the last group
    # is shorter.
    dict(
        testcase_name="expert_down",
        equation="btei,eid->bted",
        input_shape=(None, 5, 3, 7),
        output_shape=(None, 3, 8),
        permutation=(1, 0, 2),
        dims=(7, 24),
    ),
    # The same down projection with the contracted axis leading.
    dict(
        testcase_name="expert_down_contracted_first",
        equation="btie,ied->bted",
        input_shape=(None, 5, 7, 3),
        output_shape=(None, 3, 8),
        permutation=(0, 1, 2),
        dims=(7, 24),
    ),
    # A batch axis and no free axis: one column per head.
    dict(
        testcase_name="per_head_dot",
        equation="abc,cb->ab",
        input_shape=(None, 3, 8),
        output_shape=(3,),
        permutation=(0, 1),
        dims=(8, 3),
    ),
    # The contracted axis already leads.
    dict(
        testcase_name="contracted_first",
        equation="abc,cde->abde",
        input_shape=(None, 5, 8),
        output_shape=(5, 2, 4),
        permutation=(0, 1, 2),
        dims=(8, 8),
    ),
    dict(
        testcase_name="matmul",
        equation="ab,bc->ac",
        input_shape=(None, 8),
        output_shape=(6,),
        permutation=(0, 1),
        dims=(8, 6),
    ),
    # No contracted axis, and inputs summed over an axis of their own.
    dict(
        testcase_name="outer_product",
        equation="bt,nh->btnh",
        input_shape=(None, 5),
        output_shape=(None, 2, 3),
        permutation=(0, 1),
        dims=(1, 6),
    ),
    dict(
        testcase_name="summed_input_axis",
        equation="abc,cd->ad",
        input_shape=(None, 3, 8),
        output_shape=(4,),
        permutation=(0, 1),
        dims=(8, 4),
    ),
    dict(
        testcase_name="two_contracted_trailing",
        equation="abcd,edc->abe",
        input_shape=(None, 4, 3, 5),
        output_shape=(None, 6),
        permutation=(1, 2, 0),
        dims=(15, 6),
    ),
    dict(
        testcase_name="two_contracted_apart",
        equation="abcd,ced->abe",
        input_shape=(None, 4, 3, 5),
        output_shape=(None, 6),
        permutation=(0, 2, 1),
        dims=(15, 6),
    ),
]


def _kernel(equation, output_shape, input_shape, seed):
    layer = layers.EinsumDense(equation, output_shape=output_shape)
    layer.build(input_shape)
    shape = tuple(layer.kernel.shape)
    rng = np.random.default_rng(seed)
    kernel = rng.standard_normal(shape).astype("float32")
    # Kernel slices of different magnitudes, as trained heads and experts
    # have.
    return kernel * np.exp(rng.standard_normal(shape[:1])).astype(
        "float32"
    ).reshape((-1,) + (1,) * (len(shape) - 1))


def _quantize(layer, kernel, block_size):
    layer.kernel.assign(kernel)
    layer.quantize("int4", config=Int4QuantizationConfig(block_size=block_size))
    return layer


def _einsum(equation, output_shape, input_shape, kernel, block_size, **kw):
    layer = layers.EinsumDense(equation, output_shape=output_shape, **kw)
    layer.build(input_shape)
    return _quantize(layer, kernel, block_size)


def _dense(kernel, block_size):
    layer = layers.Dense(kernel.shape[1], use_bias=False)
    layer.build((None, kernel.shape[0]))
    return _quantize(layer, kernel, block_size)


def _dequantized(layer):
    return ops.convert_to_numpy(layer._quantized_weight().dequantize("float32"))


def _inputs(input_shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2,) + tuple(input_shape[1:])).astype("float32")


class Int4EinsumLayoutTest(testing.TestCase):
    """int4 splits an einsum kernel by its equation, not its shape."""

    @parameterized.named_parameters(
        named_product(EQUATIONS, block_size=[-1, 3])
    )
    def test_stores_the_contraction_like_dense(
        self,
        equation,
        input_shape,
        output_shape,
        permutation,
        dims,
        block_size,
    ):
        rows, columns = dims
        kernel = _kernel(equation, output_shape, input_shape, seed=1)
        layer = _einsum(equation, output_shape, input_shape, kernel, block_size)
        matrix = np.transpose(kernel, permutation).reshape(dims)

        # The layer stores exactly what a `Dense` stores for the matrix
        # whose rows are the contracted axes.
        dense = _dense(matrix, block_size)
        for name in ("_kernel", "kernel_scale", "kernel_zero", "g_idx"):
            stored, expected = getattr(layer, name), getattr(dense, name)
            if expected is None:
                self.assertIsNone(stored)
            else:
                self.assertAllEqual(stored, expected)
        self.assertEqual(tuple(layer._kernel.shape), (rows, -(-columns // 2)))
        self.assertAllClose(
            np.transpose(_dequantized(layer), permutation).reshape(dims),
            _dequantized(dense),
        )

        # The forward pass contracts the dequantized kernel in its own
        # layout.
        x = _inputs(input_shape, seed=2)
        self.assertAllClose(
            layer(x),
            np.einsum(equation, x, _dequantized(layer)),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    @parameterized.named_parameters(
        named_product(
            [
                # Gemma's query projection against the same weights laid
                # out `[d_model, heads, head_dim]`.
                dict(
                    testcase_name="gemma_q",
                    equation="btd,ndh->btnh",
                    reference_equation="btd,dnh->btnh",
                    input_shape=(None, 5, 32),
                    output_shape=(None, 4, 6),
                ),
                # A mixture-of-experts gate against `[d_model, experts,
                # inner]`.
                dict(
                    testcase_name="expert_gate",
                    equation="btd,edi->btei",
                    reference_equation="btd,dei->btei",
                    input_shape=(None, 5, 32),
                    output_shape=(None, 4, 6),
                ),
                # A mixture-of-experts down projection against
                # `[inner, experts, d_model]`.
                dict(
                    testcase_name="expert_down",
                    equation="btei,eid->bted",
                    reference_equation="btei,ied->bted",
                    input_shape=(None, 5, 4, 32),
                    output_shape=(None, 4, 6),
                ),
            ],
            block_size=[-1, 8],
        )
    )
    def test_error_matches_the_contracted_first_layout(
        self,
        equation,
        reference_equation,
        input_shape,
        output_shape,
        block_size,
    ):
        kernel = _kernel(equation, output_shape, input_shape, seed=3)
        layer = _einsum(equation, output_shape, input_shape, kernel, block_size)
        reference = _einsum(
            reference_equation,
            output_shape,
            input_shape,
            np.transpose(kernel, (1, 0, 2)),
            block_size,
        )

        # The same weights up to the transpose, so the same error.
        self.assertAllClose(
            _dequantized(layer),
            np.transpose(_dequantized(reference), (1, 0, 2)),
        )
        x = _inputs(input_shape, seed=4)
        y_float = np.einsum(equation, x, kernel)
        error = np.linalg.norm(ops.convert_to_numpy(layer(x)) - y_float)
        reference_error = np.linalg.norm(
            ops.convert_to_numpy(reference(x)) - y_float
        )
        self.assertAllClose(error, reference_error, rtol=1e-5)

    @parameterized.named_parameters(
        named_product(EQUATIONS, block_size=[-1, 3])
    )
    def test_round_trips_through_variables(
        self,
        equation,
        input_shape,
        output_shape,
        permutation,
        dims,
        block_size,
    ):
        del permutation, dims
        kernel = _kernel(equation, output_shape, input_shape, seed=5)
        layer = _einsum(equation, output_shape, input_shape, kernel, block_size)

        store = {}
        layer.save_own_variables(store)
        restored = layers.EinsumDense(
            equation,
            output_shape=output_shape,
            dtype=layer.dtype_policy.name,
        )
        restored.build(input_shape)
        restored.load_own_variables(store)
        self.assertAllClose(restored.kernel, layer.kernel)
        x = _inputs(input_shape, seed=6)
        self.assertAllClose(restored(x), layer(x))

    def test_many_groups_under_mixed_bfloat16(self):
        # 300 groups: indices past 256 are not exact in bfloat16, so the
        # group index must not be autocast in the forward pass.
        equation, input_shape, output_shape = (
            "btei,eid->bted",
            (None, 2, 2, 600),
            (None, 2, 4),
        )
        kernel = _kernel(equation, output_shape, input_shape, seed=7)
        x = _inputs(input_shape, seed=8)
        outputs = []
        for policy in ("float32", "mixed_bfloat16"):
            layer = _einsum(
                equation,
                output_shape,
                input_shape,
                kernel,
                block_size=2,
                dtype=dtype_policies.get(policy),
            )
            self.assertEqual(int(ops.max(layer.g_idx)), 299)
            outputs.append(ops.convert_to_numpy(ops.cast(layer(x), "float32")))
        # A misrouted group gives NaN, an index error or a wrong scale; the
        # bfloat16 compute itself stays within a few percent.
        self.assertTrue(np.all(np.isfinite(outputs[1])))
        error = np.linalg.norm(outputs[1] - outputs[0])
        self.assertLess(error / np.linalg.norm(outputs[0]), 0.05)


def _float_twin(layer, equation, output_shape, input_shape):
    """A float `EinsumDense` that holds `layer`'s dequantized kernel."""
    twin = layers.EinsumDense(equation, output_shape=output_shape)
    twin.build(input_shape)
    twin.kernel.assign(_dequantized(layer))
    return twin


class Int4InputGradientTest(testing.TestCase):
    """The input gradient is the float layer's on the dequantized kernel."""

    @parameterized.named_parameters(
        named_product(EQUATIONS, block_size=[-1, 3])
    )
    @pytest.mark.requires_trainable_backend
    def test_weight_only_gradient_is_the_float_layers(
        self,
        equation,
        input_shape,
        output_shape,
        permutation,
        dims,
        block_size,
    ):
        del permutation, dims
        kernel = _kernel(equation, output_shape, input_shape, seed=9)
        layer = _einsum(equation, output_shape, input_shape, kernel, block_size)
        twin = _float_twin(layer, equation, output_shape, input_shape)
        x = _inputs(input_shape, seed=10)
        self.assertAllClose(
            input_gradient(layer, x),
            input_gradient(twin, x),
            atol=1e-6,
            rtol=1e-6,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    @parameterized.named_parameters(
        [
            case
            for case in EQUATIONS
            if case["testcase_name"]
            in ("matmul", "gemma_q", "expert_down", "two_contracted_apart")
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_activation_quantizer_gradient_is_straight_through(
        self, equation, input_shape, output_shape, permutation, dims
    ):
        # The rounding of the inputs passes the gradient on unchanged.
        del permutation, dims
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        layer.kernel.assign(
            _kernel(equation, output_shape, input_shape, seed=11)
        )
        layer.quantize(
            "int4",
            config=Int4QuantizationConfig(
                block_size=-1, activation_quantizer=AbsMaxQuantizer()
            ),
        )
        twin = _float_twin(layer, equation, output_shape, input_shape)
        x = _inputs(input_shape, seed=12)
        self.assertAllClose(
            input_gradient(layer, x),
            input_gradient(twin, x),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
