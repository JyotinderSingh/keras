"""The calibration modes' shared chassis, tested once per mode."""

import os
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
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
    """Quantizes `layer` with `mode` and calibrates it on `x`.

    Returns the calibrator.
    """
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


# Dense, a 2-D einsum kernel, the keras-hub Gemma query projection (whose
# contraction view permutes the kernel), a mixture-of-experts down
# projection (whose expert axis is a batch of calibration problems), and a
# kernel whose last axis is such a batch axis.
KINDS = ["dense", "einsum", "permuted", "batched", "batch_last"]


def _make_layer(kind, rng, **kwargs):
    if kind == "dense":
        layer = layers.Dense(16, **kwargs)
        input_shape = (None, 12)
        x = rng.standard_normal((32, 12))
    elif kind == "einsum":
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 16), bias_axes="d", **kwargs
        )
        input_shape = (None, 3, 12)
        x = rng.standard_normal((8, 3, 12))
    elif kind == "permuted":
        layer = layers.EinsumDense(
            "btd,ndh->btnh", output_shape=(None, 4, 6), bias_axes="nh", **kwargs
        )
        input_shape = (None, 3, 12)
        x = rng.standard_normal((8, 3, 12))
    elif kind == "batched":
        layer = layers.EinsumDense(
            "bec,ecd->bed", output_shape=(4, 6), bias_axes="d", **kwargs
        )
        input_shape = (None, 4, 12)
        x = rng.standard_normal((8, 4, 12))
    else:
        layer = layers.EinsumDense(
            "bce,dce->bde", output_shape=(6, 4), bias_axes="d", **kwargs
        )
        input_shape = (None, 12, 4)
        x = rng.standard_normal((8, 12, 4))
    layer.build(input_shape)
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    return layer, x.astype("float32")


def _set_adapter(layer, rng, scale):
    """Random LoRA factors whose product is about `scale` of the kernel."""
    a = rng.standard_normal(layer.lora_kernel_a.shape).astype("float32")
    b = rng.standard_normal(layer.lora_kernel_b.shape).astype("float32")
    layer.lora_kernel_a.assign(a)
    layer.lora_kernel_b.assign(b * scale)


def _delta(layer):
    a = ops.convert_to_numpy(layer.lora_kernel_a)
    b = ops.convert_to_numpy(layer.lora_kernel_b)
    return (layer.lora_alpha / layer.lora_rank) * (a @ b)


def _save(layer):
    store = {}
    layer.save_own_variables(store)
    return store


def _stored(layer, store):
    """The saved store keyed by serialization-spec name."""
    spec = layer.variable_serialization_spec[layer.quantization_mode]
    return {name: store[str(i)] for i, name in enumerate(spec)}


def _policy_built(layer, config):
    """A fresh layer of `layer`'s kind, built under `config`'s policy."""
    kwargs = layer.get_config()
    for key in (
        "name",
        "dtype",
        "quantization_config",
        "lora_rank",
        "lora_alpha",
    ):
        kwargs.pop(key, None)
    kwargs["dtype"] = f"{config.dtype_policy_string()}_from_float32"
    new = type(layer).from_config(kwargs)
    new.build(layer._build_shapes_dict["input_shape"])
    return new


class CalibrationLoRATest(testing.TestCase):
    """LoRA on the calibration modes: forward, calibration, merged save."""

    @parameterized.named_parameters(named_product(mode=MODES, kind=KINDS))
    def test_lora_delta_matches_float(self, mode, kind):
        # The LoRA update rides on the dequantized contraction, so its
        # contribution equals the float layer's for every kernel layout.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        float_layer, _ = _make_layer(kind, rng)
        float_layer.kernel.assign(layer.kernel)
        _calibrate(layer, mode, x, group_size=4)

        def lora_delta(layer):
            before = ops.convert_to_numpy(layer(x))
            layer.enable_lora(2)
            _set_adapter(layer, np.random.default_rng(1), scale=0.05)
            return ops.convert_to_numpy(layer(x)) - before

        self.assertAllClose(
            lora_delta(layer),
            lora_delta(float_layer),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
        # `kernel` keeps the kernel's shape, as for int8 and int4.
        self.assertEqual(
            tuple(layer.kernel.shape), tuple(float_layer.kernel.shape)
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    @pytest.mark.requires_trainable_backend
    def test_lora_fit_updates_factors(self, mode):
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        a_before = ops.convert_to_numpy(layer.lora_kernel_a).copy()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, rng.standard_normal((32, 16)), epochs=2, verbose=0)
        self.assertGreater(
            np.abs(ops.convert_to_numpy(layer.lora_kernel_a) - a_before).max(),
            0.0,
        )
        self.assertGreater(
            np.abs(ops.convert_to_numpy(layer.lora_kernel_b)).max(), 0.0
        )

    @parameterized.named_parameters(
        ("gptq_2bit", "gptq", dict(weight_bits=2, group_size=8), "float32"),
        ("gptq_3bit", "gptq", dict(weight_bits=3, group_size=4), "float32"),
        (
            "gptq_4bit_act_order",
            "gptq",
            dict(weight_bits=4, group_size=4, activation_order=True),
            "float32",
        ),
        (
            "gptq_8bit_symmetric",
            "gptq",
            dict(weight_bits=8, group_size=-1, symmetric=True),
            "float32",
        ),
        (
            "gptq_8bit_mixed_bfloat16",
            "gptq",
            dict(weight_bits=8, group_size=8),
            "mixed_bfloat16",
        ),
        ("awq", "awq", dict(group_size=4), "float32"),
        ("awq_mixed_bfloat16", "awq", dict(group_size=4), "mixed_bfloat16"),
    )
    def test_lora_merge_with_zero_delta_is_identity(self, mode, kwargs, dtype):
        # A merged save rounds the merged weight onto the calibrated grid,
        # so with nothing to merge every stored variable comes back as it
        # is, whatever the compute dtype.
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng, dtype=dtype)
        _calibrate(layer, mode, x, **kwargs)
        layer.enable_lora(2)
        for name, value in _stored(layer, _save(layer)).items():
            self.assertAllEqual(value, getattr(layer, name), msg=name)

    @parameterized.named_parameters(
        named_product(mode=MODES, dtype=["float32", "mixed_bfloat16"])
    )
    def test_lora_merge_keeps_calibrated_parameters(self, mode, dtype):
        # The scale, zero point, group index (here an activation-order
        # permutation) and AWQ scales are the calibration's result; a
        # merged save changes only the codes, whatever the compute dtype.
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng, dtype=dtype)
        kwargs = dict(group_size=8)
        if mode == "gptq":
            kwargs["activation_order"] = True
        _calibrate(layer, mode, x, **kwargs)
        g_idx = ops.convert_to_numpy(layer.g_idx)
        if mode == "gptq":
            self.assertFalse(np.all(np.diff(g_idx) >= 0))
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        store = _stored(layer, _save(layer))
        for name in ("kernel_scale", "kernel_zero", "g_idx", "awq_scales"):
            if name in store:
                self.assertAllEqual(store[name], getattr(layer, name), msg=name)
        self.assertFalse(
            np.array_equal(
                ops.convert_to_numpy(store["quantized_kernel"]),
                ops.convert_to_numpy(layer.quantized_kernel),
            )
        )

    @parameterized.named_parameters(
        named_product(
            mode=MODES, kind=["dense", "permuted", "batched", "batch_last"]
        )
    )
    def test_lora_merged_weight_is_within_half_a_step(self, mode, kind):
        # Round-to-nearest under the calibrated parameters: every merged
        # weight the grid covers lands within half a code of the sum of
        # the dequantized weight and the delta, and a layer built under
        # the mode's policy reads the merged store back to that weight.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        config = _calibrate(layer, mode, x, group_size=4).config
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        quantized_weight = layer._quantized_weight()
        merged = ops.convert_to_numpy(
            quantized_weight.dequantize("float32")
        ) + _delta(layer)

        new = _policy_built(layer, config)
        new.load_own_variables(_save(layer))
        reloaded = ops.convert_to_numpy(
            new._quantized_weight().dequantize("float32")
        )

        image = ops.convert_to_numpy(quantized_weight.code_image(merged))
        low, high = quantized_weight.scheme.code_range
        covered = (image >= low - 0.5) & (image <= high + 0.5)
        # One real-valued unit per position, in the stored layout.
        step = ops.convert_to_numpy(
            quantized_weight.code_image(
                np.ones(quantized_weight.shape, "float32")
            )
        ) - ops.convert_to_numpy(
            quantized_weight.code_image(
                np.zeros(quantized_weight.shape, "float32")
            )
        )
        error = np.abs(
            ops.convert_to_numpy(quantized_weight._as_stored(reloaded - merged))
        )
        bound = 0.5 / np.abs(step) + 1e-6
        self.assertGreater(covered.mean(), 0.9)
        self.assertTrue(np.all(error[covered] <= bound[covered]))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_lora_merge_clips_outside_the_calibrated_range(self, mode):
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        layer.lora_kernel_a.assign(np.ones(layer.lora_kernel_a.shape))
        layer.lora_kernel_b.assign(np.full(layer.lora_kernel_b.shape, 5.0))
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            store = _stored(layer, _save(layer))
        messages = [
            str(w.message)
            for w in warned
            if "Merging the LoRA" in str(w.message)
        ]
        self.assertLen(messages, 1)
        self.assertIn(f"layer '{layer.name}'", messages[0])
        self.assertIn(mode.upper(), messages[0])
        quantized_weight = layer._quantized_weight()
        codes = quantized_weight.layout.unpack(store["quantized_kernel"])
        high = quantized_weight.scheme.code_range[1]
        self.assertAllEqual(codes, np.full(layer.kernel_shape, high, "uint8"))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_lora_merge_within_the_range_does_not_warn(self, mode):
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        # An update far below half a code moves no weight past its range.
        _set_adapter(layer, rng, scale=0.0005)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            _save(layer)
        self.assertLen(
            [w for w in warned if "Merging the LoRA" in str(w.message)], 0
        )

    @parameterized.named_parameters(named_product(mode=MODES, kind=KINDS))
    def test_enable_lora_before_calibration(self, mode, kind):
        # The pending forward already carries the update; the calibrators
        # quantize the base kernel, so the update stays a separate term.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        twin, _ = _make_layer(kind, rng)
        twin.kernel.assign(layer.kernel)
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.05)
        float_output = ops.convert_to_numpy(layer(x))

        config = _config(mode, group_size=4)
        layer.quantize(mode, config=config)
        self.assertTrue(layer.calibration_pending)
        self.assertFalse(layer._kernel.trainable)
        self.assertAllClose(
            layer(x), float_output, atol=1e-6, tpu_atol=1e-2, tpu_rtol=1e-2
        )
        with self.assertRaisesRegex(ValueError, "never been calibrated"):
            layer.save_own_variables({})

        twin.quantize(mode, config=config)
        for target in (layer, twin):
            calibrator = strategy_registry.get_strategy(mode).calibrator_cls(
                target, config
            )
            calibrator.observe(x)
            calibrator.quantize()
        self.assertFalse(hasattr(layer, "_kernel"))
        self.assertAllEqual(layer.quantized_kernel, twin.quantized_kernel)
        twin.enable_lora(2)
        twin.lora_kernel_a.assign(layer.lora_kernel_a)
        twin.lora_kernel_b.assign(layer.lora_kernel_b)
        self.assertAllClose(
            layer(x), twin(x), atol=1e-6, tpu_atol=1e-2, tpu_rtol=1e-2
        )

    @parameterized.named_parameters(
        named_product(mode=MODES, kind=["dense", "permuted"])
    )
    def test_lora_model_round_trip(self, mode, kind):
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        config = _calibrate(layer, mode, x, group_size=8).config
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        model = models.Sequential([layer])
        model(x)
        # What every reload must reproduce: the merged codes.
        merged = _policy_built(layer, config)
        merged.load_own_variables(_save(layer))
        expected = ops.convert_to_numpy(merged(x))
        self.assertNotAllClose(expected, model(x))

        # A `.keras` file re-creates LoRA from the config with zero factors.
        path = os.path.join(self.get_temp_dir(), "lora.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        self.assertTrue(reloaded.layers[0].lora_enabled)
        self.assertAllEqual(
            reloaded.layers[0].lora_kernel_b,
            np.zeros(layer.lora_kernel_b.shape, "float32"),
        )
        self.assertAllClose(reloaded(x), expected, atol=1e-6)

        # Weights only, into a model built under the mode's policy.
        path = os.path.join(self.get_temp_dir(), "lora.weights.h5")
        model.save_weights(path)
        fresh = models.Sequential([_policy_built(layer, config)])
        fresh.build((None,) + x.shape[1:])
        fresh.load_weights(path)
        self.assertFalse(fresh.layers[0].lora_enabled)
        self.assertAllClose(fresh(x), expected, atol=1e-6)

        # A plain checkpoint back into the LoRA model zeroes its factors.
        fresh.save_weights(path)
        model.load_weights(path)
        self.assertAllClose(model(x), expected, atol=1e-6)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_policy_built_layer_with_lora_rank(self, mode):
        # A layer built from a saved config carries `lora_rank`; it has
        # codes and no float kernel, so `enable_lora` has nothing to freeze.
        layer = layers.Dense(16, dtype=f"{mode}/4/8_from_float32", lora_rank=2)
        layer.build((None, 12))
        self.assertTrue(layer.lora_enabled)
        self.assertFalse(hasattr(layer, "_kernel"))
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 4 if mode == "gptq" else 5)
