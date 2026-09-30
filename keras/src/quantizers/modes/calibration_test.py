"""The calibration modes' shared chassis, tested once per mode."""

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq import GPTQCalibrator
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.testing.test_utils import named_product

MODES = ["gptq", "awq"]


def _config(mode, **kwargs):
    if mode == "gptq":
        return GPTQConfig(dataset=None, tokenizer=None, **kwargs)
    kwargs.setdefault("num_grid_points", 5)
    kwargs.setdefault("apply_clip", False)
    return AWQConfig(dataset=None, tokenizer=None, **kwargs)


def _calibrate(layer, mode, x, **kwargs):
    """Quantizes `layer` with `mode`, calibrates it on `x`, and returns the
    calibrator."""
    config = _config(mode, **kwargs)
    layer.quantize(mode, config=config)
    calibrator = strategy_registry.get_strategy(mode).calibrator_cls(
        layer, config
    )
    calibrator.observe(x)
    calibrator.quantize()
    return calibrator


def _statistic(calibrator):
    """The statistic a calibrator accumulates from the layer's inputs."""
    if isinstance(calibrator, GPTQCalibrator):
        return calibrator.hessian
    return calibrator.activation_magnitudes


def _einsum(equation, output_shape, input_shape, kernel=None):
    layer = layers.EinsumDense(equation, output_shape=output_shape)
    layer.build(input_shape)
    if kernel is not None:
        layer.kernel.assign(kernel)
    return layer


def _dense(kernel):
    layer = layers.Dense(kernel.shape[1])
    layer.build((None, kernel.shape[0]))
    layer.kernel.assign(kernel)
    return layer


def _dequantized(layer):
    return ops.convert_to_numpy(layer._quantized_weight().dequantize("float32"))


class CalibrationEinsumLayoutTest(testing.TestCase):
    """GPTQ and AWQ split an einsum kernel by its equation."""

    @parameterized.named_parameters(
        named_product(
            [
                # Gemma's query projection `[heads, d_model, head_dim]`
                # against the same weights laid out `[d_model, heads,
                # head_dim]`.
                dict(
                    testcase_name="gemma_q",
                    equation="btd,ndh->btnh",
                    reference_equation="btd,dnh->btnh",
                    output_shape=(None, 2, 4),
                ),
                # A mixture-of-experts gate `[experts, d_model, inner]`
                # against `[d_model, experts, inner]`.
                dict(
                    testcase_name="expert_gate",
                    equation="btd,edi->btei",
                    reference_equation="btd,dei->btei",
                    output_shape=(None, 3, 6),
                ),
            ],
            mode=MODES,
        )
    )
    def test_kernel_only_axes_share_one_statistic(
        self, mode, equation, reference_equation, output_shape
    ):
        input_shape = (None, 5, 8)
        rng = np.random.default_rng(seed=3)
        layer = _einsum(equation, output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel)
        reference = _einsum(
            reference_equation,
            output_shape,
            input_shape,
            kernel=np.transpose(kernel, (1, 0, 2)),
        )
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        reference_calibrator = _calibrate(reference, mode, x, group_size=4)

        # One statistic over the model width, shared by every head or
        # expert, and the same weights up to the transpose.
        statistic = _statistic(calibrator)
        self.assertEqual(ops.shape(statistic)[0], 8)
        self.assertAllClose(statistic, _statistic(reference_calibrator))
        self.assertAllClose(
            _dequantized(layer),
            np.transpose(_dequantized(reference), (1, 0, 2)),
        )
        self.assertAllClose(layer(x), reference(x))
        # Stored as `(d_model, packed columns)`, like the reference.
        columns = int(np.prod(output_shape[1:]))
        self.assertEqual(tuple(layer.quantized_kernel.shape), (8, columns // 2))
        self.assertEqual(tuple(layer.g_idx.shape), (8,))

    @parameterized.named_parameters(
        named_product(
            [
                # A QKV projection `[d_model, heads, head_dim]` is the
                # `Dense` kernel `(d_model, heads * head_dim)`.
                dict(
                    testcase_name="qkv",
                    equation="btd,dnh->btnh",
                    output_shape=(None, 2, 4),
                    input_shape=(None, 5, 8),
                ),
                # An attention output projection `[heads, head_dim,
                # d_model]` is the `Dense` kernel `(heads * head_dim,
                # d_model)`.
                dict(
                    testcase_name="attention_output",
                    equation="btnh,nhd->btd",
                    output_shape=(None, 8),
                    input_shape=(None, 5, 2, 4),
                ),
            ],
            mode=MODES,
        )
    )
    def test_contracted_axes_match_dense(
        self, mode, equation, output_shape, input_shape
    ):
        rng = np.random.default_rng(seed=5)
        layer = _einsum(equation, output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel)
        dense = _dense(kernel.reshape(8, 8))
        x = rng.standard_normal((2,) + input_shape[1:]).astype("float32")
        x_dense = x.reshape(2, 5, 8)
        calibrator = _calibrate(layer, mode, x, group_size=4)
        dense_calibrator = _calibrate(dense, mode, x_dense, group_size=4)

        self.assertAllClose(
            _statistic(calibrator), _statistic(dense_calibrator)
        )
        self.assertAllClose(
            _dequantized(layer).reshape(8, 8), _dequantized(dense)
        )
        self.assertAllClose(ops.reshape(layer(x), (2, 5, 8)), dense(x_dense))

    @parameterized.named_parameters(
        named_product(
            [
                dict(testcase_name="one_group", inner=6, group_size=-1),
                dict(testcase_name="grouped", inner=6, group_size=3),
                # The last group of every expert is shorter.
                dict(testcase_name="ragged", inner=7, group_size=3),
            ],
            mode=MODES,
        )
    )
    def test_batch_axis_calibrates_each_expert_alone(
        self, mode, inner, group_size
    ):
        # A mixture-of-experts down projection `[experts, inner, d_model]`:
        # the expert axis is shared by the inputs, so every expert is its
        # own problem, calibrated from its own slice of the inputs.
        experts, width = 3, 8
        kwargs = dict(group_size=group_size)
        if group_size != -1:
            # GPTQ's permuted groups and AWQ's clipping sample, per expert.
            if mode == "gptq":
                kwargs["activation_order"] = True
            else:
                kwargs["apply_clip"] = True
        rng = np.random.default_rng(seed=11)
        layer = _einsum(
            "btei,eid->bted", (None, experts, width), (None, 5, experts, inner)
        )
        kernel = ops.convert_to_numpy(layer.kernel)
        x = rng.standard_normal((2, 5, experts, inner)).astype("float32")
        calibrator = _calibrate(layer, mode, x, **kwargs)
        statistic = ops.convert_to_numpy(_statistic(calibrator))
        self.assertEqual(statistic.shape[:2], (experts, inner))

        dequantized = _dequantized(layer)
        outputs = ops.convert_to_numpy(layer(x))
        n_groups = 1 if group_size == -1 else -(-inner // group_size)
        g_idx = ops.convert_to_numpy(layer.g_idx).reshape(experts, inner)
        for expert in range(experts):
            dense = _dense(kernel[expert])
            x_expert = x[:, :, expert, :]
            dense_calibrator = _calibrate(
                dense, mode, x_expert.reshape(-1, inner), **kwargs
            )
            self.assertAllClose(statistic[expert], _statistic(dense_calibrator))
            self.assertAllClose(dequantized[expert], _dequantized(dense))
            self.assertAllClose(outputs[:, :, expert, :], dense(x_expert))
            # The experts stack along the rows, each with its own groups.
            self.assertAllClose(
                g_idx[expert],
                ops.convert_to_numpy(dense.g_idx) + expert * n_groups,
            )
        self.assertEqual(
            tuple(layer.quantized_kernel.shape),
            (experts * inner, width // 2),
        )
        self.assertEqual(
            tuple(layer.kernel_scale.shape), (experts * n_groups, width)
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_batch_axis_groups_past_256_under_mixed_bfloat16(self, mode):
        # 4 experts of 80 groups each, numbered across the experts up to
        # 319. bfloat16 holds integers exactly only up to 256, so an
        # autocast `g_idx` would send later rows to another group.
        equation, output_shape = "btei,eid->bted", (None, 4, 4)
        input_shape = (None, 3, 4, 160)
        rng = np.random.default_rng(seed=17)
        layer = _einsum(equation, output_shape, input_shape)
        x = rng.standard_normal((2, 3, 4, 160)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=2)
        self.assertEqual(int(ops.max(layer.g_idx)), 319)
        expected = ops.convert_to_numpy(layer(x))

        store = {}
        layer.save_own_variables(store)
        policy = calibrator.config.dtype_policy_string()
        restored = layers.EinsumDense(
            equation,
            output_shape=output_shape,
            dtype=f"{policy}_from_mixed_bfloat16",
        )
        restored.build(input_shape)
        restored.load_own_variables(store)
        y = ops.convert_to_numpy(ops.cast(restored(x), "float32"))
        atol = 0.02 * np.abs(expected).max()
        self.assertAllClose(y, expected, rtol=0.02, atol=atol)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_permuted_layout_round_trips_through_variables(self, mode):
        rng = np.random.default_rng(seed=13)
        layer = _einsum("btd,ndh->btnh", (None, 2, 4), (None, 5, 8))
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=-1)
        y = ops.convert_to_numpy(layer(x))

        store = {}
        layer.save_own_variables(store)
        restored = layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, 2, 4),
            dtype=f"{calibrator.config.dtype_policy_string()}_from_float32",
        )
        restored.build((None, 5, 8))
        restored.load_own_variables(store)
        self.assertEqual(tuple(restored.quantized_kernel.shape), (8, 4))
        self.assertAllClose(restored(x), y)
