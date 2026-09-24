"""LoRA on the calibration modes: forward, calibration and merged save."""

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
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.testing.test_utils import named_product

MODES = ["gptq", "awq"]
# Dense, a 2-D einsum kernel, the keras-hub Gemma query projection (whose
# calibration view permutes the kernel) and a mixture-of-experts down
# projection (whose expert axis is a batch of calibration problems).
KINDS = ["dense", "einsum", "permuted", "batched"]


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
    else:
        layer = layers.EinsumDense(
            "bec,ecd->bed", output_shape=(4, 6), bias_axes="d", **kwargs
        )
        input_shape = (None, 4, 12)
        x = rng.standard_normal((8, 4, 12))
    layer.build(input_shape)
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    return layer, x.astype("float32")


def _config(mode, **kwargs):
    if mode == "gptq":
        return GPTQConfig(dataset=None, tokenizer=None, **kwargs)
    kwargs.setdefault("num_grid_points", 5)
    return AWQConfig(dataset=None, tokenizer=None, **kwargs)


def _calibrate(layer, mode, x, **kwargs):
    """Quantizes `layer` with `mode` and calibrates it on `x`."""
    config = _config(mode, **kwargs)
    layer.quantize(mode, config=config)
    calibrator = strategy_registry.get_strategy(mode).calibrator_cls(
        layer, config
    )
    calibrator.observe(x)
    calibrator.quantize()
    calibrator.release()
    return config


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
        # `kernel` stays the codes-plus-delta convention of int8 and int4.
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
        store = {}
        layer.save_own_variables(store)
        for name, value in _stored(layer, store).items():
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
        store = _stored(layer, self._save(layer))
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
        named_product(mode=MODES, kind=["dense", "permuted", "batched"])
    )
    def test_lora_merged_weight_is_within_half_a_step(self, mode, kind):
        # Round-to-nearest under the calibrated parameters: every merged
        # weight the grid covers lands within half a code of the sum of
        # the dequantized weight and the delta, and a layer built under
        # the mode's policy reads the merged store back to that weight.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        config = _calibrate(layer, mode, x, group_size=4)
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        qvariable = layer._qvariable()
        merged = ops.convert_to_numpy(
            qvariable.dequantize(dtype="float32")
        ) + _delta(layer)

        new = _policy_built(layer, config)
        new.load_own_variables(self._save(layer))
        reloaded = ops.convert_to_numpy(
            new._qvariable().dequantize(dtype="float32")
        )

        image = ops.convert_to_numpy(qvariable.code_image(merged))
        low, high = qvariable.scheme.code_range
        covered = (image >= low - 0.5) & (image <= high + 0.5)
        # One real-valued unit per position, in the stored layout.
        step = ops.convert_to_numpy(
            qvariable.code_image(np.ones(qvariable.shape, "float32"))
        ) - ops.convert_to_numpy(
            qvariable.code_image(np.zeros(qvariable.shape, "float32"))
        )
        error = np.abs(
            ops.convert_to_numpy(qvariable._as_stored(reloaded - merged))
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
            store = _stored(layer, self._save(layer))
        messages = [
            str(w.message)
            for w in warned
            if "Merging the LoRA" in str(w.message)
        ]
        self.assertLen(messages, 1)
        self.assertIn(f"layer '{layer.name}'", messages[0])
        self.assertIn(mode.upper(), messages[0])
        codes = layer._qvariable().layout.unpack(store["quantized_kernel"])
        high = layer._qvariable().scheme.code_range[1]
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
            self._save(layer)
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

        for target in (layer, twin):
            if target is twin:
                twin.quantize(mode, config=config)
            calibrator = strategy_registry.get_strategy(mode).calibrator_cls(
                target, config
            )
            calibrator.observe(x)
            calibrator.quantize()
            calibrator.release()
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
        config = _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        model = models.Sequential([layer])
        model(x)
        # What every reload must reproduce: the merged codes.
        merged = _policy_built(layer, config)
        merged.load_own_variables(self._save(layer))
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

    def _save(self, layer):
        store = {}
        layer.save_own_variables(store)
        return store
