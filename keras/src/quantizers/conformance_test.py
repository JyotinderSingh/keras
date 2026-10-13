"""Conformance of every quantizable layer to every mode it supports.

`SUPPORTED` is the written (mode, layer) table: a change to the support
matrix is a change to this table. Every supported pair runs the same
stages: the life cycle, LoRA, the dtype policy setter, the input
gradient, the TF SavedModel export and `from_config` with `set_weights`.
The reversible lookups also run the reverse call with a soft cap. The
calibration modes refuse `quantize` on one layer, which stays float, and
also run through `Model.quantize`. Every other pair is refused before
the layer changes.

The table holds the third-party layers of `quantization_test_utils`: a
layer outside Keras that follows the quantization protocol runs every
stage with every mode it lists whose geometry the mode handles, and the
calibration modes also run on it through `Model.quantize`. The ternary
mode handles only a 2-D kernel, so it refuses `Pointwise1D` before the
layer changes. `ReleasedProtocolLayer` covers a layer on the per-layer
protocol of Keras 3.12-3.15.
"""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_test_utils import CALIBRATION_MODES
from keras.src.quantizers.quantization_test_utils import CUSTOM_OBJECTS
from keras.src.quantizers.quantization_test_utils import LAYERS
from keras.src.quantizers.quantization_test_utils import ReleasedProtocolLayer
from keras.src.quantizers.quantization_test_utils import build_layer
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import layer_inputs
from keras.src.quantizers.quantization_test_utils import policy_built
from keras.src.quantizers.quantization_test_utils import tiny_calibration_model
from keras.src.quantizers.quantization_test_utils import token_dataset


class ModeCase:
    """One row of the mode table.

    Args:
        mode: The registered mode.
        make_config: Returns the config to quantize with.
        max_error: The largest relative output error against the float
            layer, or `None` for no bound.
        weight_only: Whether the forward pass is the float forward pass on
            the dequantized weight.
        policy_carries_config: Whether the policy name that quantization
            gives holds every parameter of the config.
    """

    def __init__(
        self,
        mode,
        make_config,
        max_error,
        weight_only,
        policy_carries_config=True,
    ):
        self.mode = mode
        self.make_config = make_config
        self.max_error = max_error
        self.weight_only = weight_only
        self.policy_carries_config = policy_carries_config


MODES = {
    "int8": ModeCase("int8", lambda: None, 0.05, False),
    "int8_weight_only": ModeCase(
        "int8",
        lambda: Int8QuantizationConfig(activation_quantizer=None),
        0.05,
        True,
        policy_carries_config=False,
    ),
    "int4_per_channel": ModeCase(
        "int4", lambda: Int4QuantizationConfig(block_size=-1), 0.3, True
    ),
    "int4_grouped": ModeCase(
        "int4", lambda: Int4QuantizationConfig(block_size=4), 0.3, True
    ),
    "float8": ModeCase("float8", lambda: None, 0.2, False),
    "ternary": ModeCase("ternary", lambda: None, None, True),
    "gptq": ModeCase(
        "gptq", lambda: calibration_config("gptq", group_size=4), 0.3, True
    ),
    "awq": ModeCase(
        "awq",
        lambda: calibration_config("awq", group_size=4, num_grid_points=5),
        0.3,
        True,
    ),
}

PROJECTIONS = ["dense", "einsum", "einsum_permuted", "einsum_batched"]
LOOKUPS = ["embedding", "reversible_tied", "reversible_untied"]
THIRD_PARTY_PROJECTIONS = ["permuted_dense", "pointwise"]
THIRD_PARTY_LOOKUPS = ["token_table"]
ALL_PROJECTIONS = PROJECTIONS + THIRD_PARTY_PROJECTIONS
ALL_LOOKUPS = LOOKUPS + THIRD_PARTY_LOOKUPS

# The support matrix, written out.
SUPPORTED = {
    "int8": ALL_PROJECTIONS + ALL_LOOKUPS,
    "int8_weight_only": ALL_PROJECTIONS + ALL_LOOKUPS,
    "int4_per_channel": ALL_PROJECTIONS + ALL_LOOKUPS,
    "int4_grouped": ALL_PROJECTIONS + ALL_LOOKUPS,
    "float8": ALL_PROJECTIONS,
    # `pointwise` lists ternary, but its kernel is 3-D.
    "ternary": ["dense", "ternary_dense", "permuted_dense"],
    "gptq": ALL_PROJECTIONS,
    "awq": ALL_PROJECTIONS,
}


def _pairs(keep):
    return [
        dict(
            testcase_name=f"{mode_name}_{kind}", mode_name=mode_name, kind=kind
        )
        for mode_name in MODES
        for kind in LAYERS
        if keep(mode_name, kind)
    ]


PAIRS = _pairs(lambda m, k: k in SUPPORTED[m])
REFUSED = _pairs(lambda m, k: k not in SUPPORTED[m])
CALIBRATED_PAIRS = _pairs(
    lambda m, k: k in SUPPORTED[m] and MODES[m].mode in CALIBRATION_MODES
)
UNCALIBRATED_PAIRS = _pairs(
    lambda m, k: k in SUPPORTED[m] and MODES[m].mode not in CALIBRATION_MODES
)
PROJECTION_PAIRS = _pairs(
    lambda m, k: k in SUPPORTED[m] and not LAYERS[k].is_lookup
)
REVERSIBLE_PAIRS = _pairs(
    lambda m, k: k in SUPPORTED[m] and k.startswith("reversible")
)


def _skip_known_gaps(test, mode_name, kind):
    """Skips a pair with a defect that a later change fixes."""
    if (
        mode_name == "int8"
        and kind == "einsum_batched"
        and backend.backend() == "jax"
        and not (testing.jax_uses_gpu() or testing.jax_uses_tpu())
    ):
        test.skipTest(
            "XLA:CPU computes the int8 dot of a batch-axis einsum wrongly."
        )


def _quantize(layer, case, x):
    """Quantizes `layer` with `case`, calibrating it on `x` if needed."""
    config = case.make_config()
    if case.mode in CALIBRATION_MODES:
        calibrate_layer(layer, config, x)
    else:
        layer.quantize(case.mode, config=config)


def _twin(kind, layer):
    """A float layer of row `kind` with `layer`'s weights."""
    twin = build_layer(kind)
    twin.set_weights(layer.get_weights())
    return twin


def _numpy(x):
    return ops.convert_to_numpy(x)


def _relative_error(y, reference):
    y, reference = _numpy(y), _numpy(reference)
    return np.linalg.norm(y - reference) / np.linalg.norm(reference)


def _snapshot(layer):
    """The layer's variables and their values."""
    return [(v, _numpy(v).copy()) for v in layer.weights]


def _store(layer):
    store = {}
    layer.save_own_variables(store)
    return {key: _numpy(value) for key, value in store.items()}


def _spec_variable(layer, entry, weight_name):
    """The variable a spec entry names (`kernel` is held at `_kernel`)."""
    if entry == weight_name:
        return getattr(layer, f"_{entry}", None)
    return getattr(layer, entry, None)


def _variable_table(layer):
    return sorted(
        (v.name, tuple(v.shape), backend.standardize_dtype(v.dtype))
        for v in layer.weights
    )


def _set_lora_factors(layer, weight_name, scale):
    rng = np.random.default_rng(1)
    for factor in ("a", "b"):
        variable = getattr(layer, f"lora_{weight_name}_{factor}")
        variable.assign(scale * rng.standard_normal(variable.shape))


def _lora_update(layer, weight_name):
    """The scaled LoRA update of the weight, in the weight's shape."""
    a = _numpy(getattr(layer, f"lora_{weight_name}_a"))
    b = _numpy(getattr(layer, f"lora_{weight_name}_b"))
    return (layer.lora_alpha / layer.lora_rank) * (a @ b)


def _dequantized(strategy, layer):
    """The real-valued weight that the layer's codes store."""
    return _numpy(strategy.quantized_weights(layer)[0].dequantize("float32"))


class QuantizationConformanceTest(testing.TestCase):
    def test_table_covers_every_registered_mode(self):
        self.assertEqual(
            sorted({case.mode for case in MODES.values()}),
            sorted(strategy_registry.registered_modes()),
        )

    @parameterized.named_parameters(PAIRS)
    def test_life_cycle(self, mode_name, kind):
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        weight_name = LAYERS[kind].weight_name
        strategy = strategy_registry.get_strategy(case.mode)
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        twin = _twin(kind, layer)
        x = layer_inputs(kind, rng)
        y_float = _numpy(layer(x))

        _quantize(layer, case, x)
        self.assertEqual(layer.quantization_mode, case.mode)
        # The policy name parses back to the same policy.
        name = layer.dtype_policy.name
        self.assertTrue(name.startswith(case.mode), msg=name)
        self.assertTrue(name.endswith("_from_float32"), msg=name)
        self.assertEqual(dtype_policies.get(name).name, name)
        if case.mode in CALIBRATION_MODES:
            # The policy name gives the bit width and the group size, so a
            # calibrated layer keeps no config, in memory or serialized.
            self.assertIsNone(layer.quantization_config)
            self.assertIsNone(layer.get_config()["quantization_config"])

        y = _numpy(layer(x))
        if case.max_error is not None:
            self.assertLess(_relative_error(y, y_float), case.max_error)
        if kind == "ternary_dense":
            # Quantizing freezes the forward value of the ternary kernel.
            self.assertAllClose(y, y_float, atol=1e-5, rtol=1e-5)

        views = strategy.quantized_weights(layer)
        if strategy.owns_weight_storage:
            self.assertLen(views, 2 if kind == "reversible_untied" else 1)
            for view in views:
                self.assertEqual(
                    tuple(view.dequantize("float32").shape), view.shape
                )
            float_weight = getattr(twin, f"_{weight_name}")
            self.assertEqual(views[0].shape, tuple(float_weight.shape))
            # A lookup never quantizes its integer inputs.
            weight_only = case.weight_only or LAYERS[kind].is_lookup
            if weight_only and kind != "ternary_dense":
                float_weight.assign(views[0].dequantize("float32"))
                self.assertAllClose(
                    y, twin(x), atol=1e-5, rtol=1e-5, tpu_atol=1e-2
                )
        else:
            self.assertEqual(views, ())

        # The store is positional, in the spec's order.
        store = _store(layer)
        spec = layer.variable_serialization_spec[case.mode]
        stored = [
            entry
            for entry in spec
            if _spec_variable(layer, entry, weight_name) is not None
        ]
        self.assertEqual(list(store), [str(i) for i in range(len(stored))])
        for index, entry in enumerate(stored):
            self.assertAllEqual(
                store[str(index)],
                _spec_variable(layer, entry, weight_name),
                msg=entry,
            )

        # A layer built under the policy holds the same variables and
        # reads the store back.
        restored = policy_built(layer)
        self.assertEqual(restored.dtype_policy.name, name)
        self.assertEqual(_variable_table(restored), _variable_table(layer))
        restored.load_own_variables(store)
        self.assertAllClose(restored(x), y)

        # `.keras` and `.weights.h5` round trips.
        model = models.Sequential([layer])
        model(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded = saving.load_model(path, custom_objects=CUSTOM_OBJECTS)
        self.assertEqual(loaded.layers[0].dtype_policy.name, name)
        self.assertAllClose(loaded(x), y)
        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)
        fresh = models.Sequential([policy_built(layer)])
        fresh(x)
        fresh.load_weights(path)
        self.assertAllClose(fresh(x), y)

    @parameterized.named_parameters(PAIRS)
    def test_lora(self, mode_name, kind):
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        weight_name = LAYERS[kind].weight_name
        strategy = strategy_registry.get_strategy(case.mode)
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        x = layer_inputs(kind, rng)
        if not hasattr(layer, "enable_lora"):
            self.skipTest("The layer has no LoRA.")
        if not strategy.supports_lora:
            # Refused in either order, before the layer changes.
            message = (
                f"lora is not currently supported with {case.mode.upper()}"
            )
            lora_first = build_layer(kind)
            lora_first.enable_lora(2)
            before = _snapshot(lora_first)
            with self.assertRaisesRegex(NotImplementedError, message):
                _quantize(lora_first, case, x)
            self.assertIsNone(lora_first.quantization_mode)
            self.assertEqual(len(lora_first.weights), len(before))
            for variable, (expected, value) in zip(lora_first.weights, before):
                self.assertIs(variable, expected)
                self.assertAllEqual(variable, value)
            _quantize(layer, case, x)
            with self.assertRaisesRegex(NotImplementedError, message):
                layer.enable_lora(2)
            self.assertFalse(layer.lora_enabled)
            return

        twin = _twin(kind, layer)
        _quantize(layer, case, x)

        def lora_delta(target):
            before = _numpy(target(x))
            target.enable_lora(2)
            _set_lora_factors(target, weight_name, scale=0.3)
            return _numpy(target(x)) - before

        # The update rides on the quantized forward in low-rank form.
        delta = lora_delta(layer)
        self.assertGreater(np.abs(delta).max(), 0)
        self.assertAllClose(
            delta, lora_delta(twin), atol=1e-5, rtol=1e-5, tpu_atol=1e-2
        )

        # A merged save loads into a layer without LoRA, and its weight
        # carries the update: it is much closer to the quantized weight
        # plus the update than the quantized weight is. A calibration mode
        # rounds onto its calibrated grid, which clips the update at the
        # range of each group, so its bound is looser.
        merged = policy_built(layer)
        merged.load_own_variables(_store(layer))
        self.assertFalse(merged.lora_enabled)
        weight = _dequantized(strategy, layer)
        update = _lora_update(layer, weight_name)
        error = np.linalg.norm(_dequantized(strategy, merged) - weight - update)
        bound = 0.9 if case.mode in CALIBRATION_MODES else 0.5
        self.assertLess(error, bound * np.linalg.norm(update))

    @parameterized.named_parameters(REFUSED)
    def test_unsupported_pair_is_refused_before_any_change(
        self, mode_name, kind
    ):
        case = MODES[mode_name]
        layer = build_layer(kind)
        before = _snapshot(layer)
        with self.assertRaises(NotImplementedError):
            layer.quantize(case.mode, config=case.make_config())
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(len(layer.weights), len(before))
        for variable, (expected, value) in zip(layer.weights, before):
            self.assertIs(variable, expected)
            self.assertAllEqual(variable, value)

    @parameterized.named_parameters(REVERSIBLE_PAIRS)
    def test_reverse_call_and_soft_cap(self, mode_name, kind):
        case = MODES[mode_name]
        strategy = strategy_registry.get_strategy(case.mode)
        rng = np.random.default_rng(0)
        capped = build_layer(kind, logit_soft_cap=2.0)
        plain = build_layer(kind)
        plain.set_weights(capped.get_weights())
        hidden = (5.0 * rng.standard_normal((4, 8))).astype("float32")
        y_float = _numpy(plain(hidden, reverse=True))
        _quantize(capped, case, None)
        _quantize(plain, case, None)

        logits = _numpy(plain(hidden, reverse=True))
        self.assertAllClose(
            capped(hidden, reverse=True),
            2.0 * np.tanh(logits / 2.0),
            atol=1e-5,
            rtol=1e-5,
        )
        views = strategy.quantized_weights(plain)
        if kind == "reversible_untied":
            table = _numpy(views[1].dequantize("float32"))
        else:
            table = _numpy(views[0].dequantize("float32")).T
        if case.weight_only:
            self.assertAllClose(logits, hidden @ table, atol=1e-4, rtol=1e-5)
        else:
            self.assertLess(_relative_error(logits, y_float), case.max_error)

    @parameterized.named_parameters(UNCALIBRATED_PAIRS)
    def test_policy_setter(self, mode_name, kind):
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        twin = _twin(kind, layer)
        x = layer_inputs(kind, rng)
        _quantize(layer, case, x)
        name = layer.dtype_policy.name

        # Setting the policy on a built float layer quantizes it with the
        # policy's parameters.
        twin.dtype_policy = name
        self.assertEqual(twin.quantization_mode, case.mode)
        self.assertEqual(twin.dtype_policy.name, name)
        store, twin_store = _store(layer), _store(twin)
        self.assertEqual(list(twin_store), list(store))
        for key, value in store.items():
            self.assertAllEqual(twin_store[key], value, msg=key)
        if case.policy_carries_config:
            self.assertAllClose(twin(x), layer(x))

    @parameterized.named_parameters(CALIBRATED_PAIRS)
    def test_policy_setter_refuses_calibration_modes(self, mode_name, kind):
        case = MODES[mode_name]
        layer = build_layer(kind)
        before = _snapshot(layer)
        name = f"{case.make_config().dtype_policy_string()}_from_float32"
        with self.assertRaisesRegex(ValueError, "Implicitly enabling"):
            layer.dtype_policy = name
        self.assertEqual(len(layer.weights), len(before))
        for variable, (expected, value) in zip(layer.weights, before):
            self.assertIs(variable, expected)
            self.assertAllEqual(variable, value)

    @parameterized.named_parameters(CALIBRATED_PAIRS)
    def test_refused_policy_setter_keeps_the_float_policy(
        self, mode_name, kind
    ):
        self.skipTest("The dtype policy setter keeps a policy it refused.")
        case = MODES[mode_name]
        layer = build_layer(kind)
        name = f"{case.make_config().dtype_policy_string()}_from_float32"
        with self.assertRaisesRegex(ValueError, "Implicitly enabling"):
            layer.dtype_policy = name
        self.assertEqual(layer.dtype_policy.name, "float32")
        self.assertIsNone(layer.quantization_mode)

    @parameterized.named_parameters(CALIBRATED_PAIRS)
    def test_layer_quantize_refuses_calibration_modes(self, mode_name, kind):
        # A calibration mode needs the run of `Model.quantize`; one layer
        # on its own is refused and stays float.
        case = MODES[mode_name]
        layer = build_layer(kind)
        before = _snapshot(layer)
        with self.assertRaisesRegex(ValueError, r"model\.quantize"):
            layer.quantize(case.mode, config=case.make_config())
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(len(layer.weights), len(before))
        for variable, (expected, value) in zip(layer.weights, before):
            self.assertIs(variable, expected)
            self.assertAllEqual(variable, value)

    @parameterized.named_parameters(PROJECTION_PAIRS)
    @pytest.mark.requires_trainable_backend
    def test_input_gradient(self, mode_name, kind):
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        if case.mode == "float8" and backend.backend() == "tensorflow":
            self.skipTest(
                "The float8 custom gradient fails under an eager TensorFlow "
                "gradient tape."
            )
        strategy = strategy_registry.get_strategy(case.mode)
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        reference = _twin(kind, layer)
        x = layer_inputs(kind, rng)
        _quantize(layer, case, x)
        gradient = input_gradient(layer, x)

        if strategy.owns_weight_storage and kind != "ternary_dense":
            # The gradient flows through the dequantized weight, whatever
            # the forward pass does with the inputs.
            weight = strategy.quantized_weights(layer)[0]
            reference._kernel.assign(weight.dequantize("float32"))
            self.assertAllClose(
                gradient,
                input_gradient(reference, x),
                atol=1e-5,
                rtol=1e-5,
                tpu_atol=1e-2,
            )
        elif kind == "ternary_dense":
            # The float layer's straight-through kernel is the frozen one.
            self.assertAllClose(
                gradient, input_gradient(reference, x), atol=1e-5, rtol=1e-5
            )
        else:
            # float8 rounds the inputs and the kernel to 3 mantissa bits.
            self.assertLess(
                _relative_error(gradient, input_gradient(reference, x)), 0.1
            )

    @parameterized.named_parameters(PAIRS)
    def test_tf_saved_model_export(self, mode_name, kind):
        if backend.backend() != "tensorflow":
            self.skipTest("The TF SavedModel export needs TensorFlow.")
        if testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on Tensorflow GPU")
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        x = layer_inputs(kind, rng)
        _quantize(layer, case, x)
        # The export signature takes its dtype from the input.
        inputs = layers.Input(x.shape[1:], dtype=str(x.dtype))
        model = models.Sequential([inputs, layer])
        y = model(x)
        path = os.path.join(self.get_temp_dir(), "exported_model")
        model.export(path, format="tf_saved_model")
        reloaded = export.TFSMLayer(path)
        self.assertAllClose(reloaded(x), y)

    @parameterized.named_parameters(PAIRS)
    def test_from_config_and_set_weights(self, mode_name, kind):
        _skip_known_gaps(self, mode_name, kind)
        case = MODES[mode_name]
        rng = np.random.default_rng(0)
        layer = build_layer(kind)
        x = layer_inputs(kind, rng)
        _quantize(layer, case, x)
        new = policy_built(layer)
        new.set_weights(layer.get_weights())
        self.assertAllClose(new(x), layer(x))

    @parameterized.named_parameters(
        _pairs(
            lambda m, k: k in SUPPORTED[m]
            and MODES[m].mode in CALIBRATION_MODES
            and not LAYERS[k].is_lookup
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_calibration_through_model_quantize(self, mode_name, kind):
        case = MODES[mode_name]
        input_shape = LAYERS[kind].input_shape
        sequence_length = input_shape[1] if len(input_shape) == 3 else 3
        layer = LAYERS[kind].make()
        # Every layer of the table has 6 outputs per position.
        model, structure = tiny_calibration_model(
            [layer, layers.Reshape((sequence_length, 6))],
            sequence_length=sequence_length,
            embed_dim=input_shape[-1],
        )
        rng = np.random.default_rng(0)
        dataset = token_dataset(4, sequence_length, 48, rng)
        x = np.concatenate(dataset, axis=0)
        y_float = _numpy(model(x))
        config = case.make_config()
        config.dataset = dataset
        config.tokenizer = lambda text: text
        config.num_samples = 4
        config.sequence_length = sequence_length
        config.quantization_layer_structure = structure

        model.quantize(case.mode, config=config, verbose=False)
        self.assertEqual(layer.quantization_mode, case.mode)
        # Calibrated: the layer holds codes, not its float kernel.
        self.assertIsNotNone(layer._quantized_weight())
        y = _numpy(model(x))
        self.assertLess(_relative_error(y, y_float), case.max_error)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded = saving.load_model(path, custom_objects=CUSTOM_OBJECTS)
        self.assertAllClose(loaded(x), y)


class ReleasedProtocolTest(testing.TestCase):
    """A layer that implements `quantize` itself, as released layers do."""

    def test_quantize_call_setter_save_load(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8)).astype("float32")
        layer = ReleasedProtocolLayer()
        layer.build((None, 8))
        y_float = _numpy(layer(x))

        layer.quantize("int8")
        self.assertEqual(layer.quantization_mode, "int8")
        self.assertEqual(layer.dtype_policy.name, "int8_from_float32")
        # The forward pass is the layer's own `quantized_call`.
        layer.scale.assign(np.array([1.0, 2.0, 3.0], "float32"))
        y = _numpy(layer(x))
        self.assertAllClose(y, y_float * np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "already quantized"):
            layer.quantize("int8")

        # The dtype policy setter quantizes a built float layer.
        other = ReleasedProtocolLayer()
        other.build((None, 8))
        other.dtype_policy = "int8_from_float32"
        self.assertEqual(other.quantization_mode, "int8")
        self.assertLen(other.weights, 2)

        # A layer built from the policy reads the store, and the saved
        # model files load.
        restored = policy_built(layer)
        restored.load_own_variables(_store(layer))
        self.assertAllClose(restored(x), y)
        model = models.Sequential([layer])
        model(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded = saving.load_model(path, custom_objects=CUSTOM_OBJECTS)
        self.assertEqual(
            loaded.layers[0].dtype_policy.name, "int8_from_float32"
        )
        self.assertAllClose(loaded(x), y)
        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)
        fresh = models.Sequential([policy_built(layer)])
        fresh(x)
        fresh.load_weights(path)
        self.assertAllClose(fresh(x), y)

        # `Model.quantize` calls the layer's own `quantize`.
        model = models.Sequential([layers.Input((8,)), ReleasedProtocolLayer()])
        report = model.quantize("int8", verbose=False)
        self.assertEqual(model.layers[0].quantization_mode, "int8")
        self.assertLen(report.quantized, 1)
