"""Conformance of every quantizable layer to every mode it supports.

`SUPPORTED` is the written (mode, layer) table: a change to the support
matrix is a change to this table. Every supported pair runs the same
stages: the life cycle, the scale convention of the stored variables,
LoRA, the dtype policy setter, the input gradient, the TF SavedModel
export and `from_config` with `set_weights`.
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
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_test_utils import CALIBRATION_MODES
from keras.src.quantizers.quantization_test_utils import CUSTOM_OBJECTS
from keras.src.quantizers.quantization_test_utils import LAYERS
from keras.src.quantizers.quantization_test_utils import ReleasedProtocolLayer
from keras.src.quantizers.quantization_test_utils import StandardProjection
from keras.src.quantizers.quantization_test_utils import TokenTable
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
    """Skips a pair with a known defect."""
    if (
        mode_name == "int8"
        and kind == "einsum_batched"
        and backend.backend() == "jax"
        and not testing.jax_uses_tpu()
    ):
        test.skipTest(
            "`jnp.einsum` on int8 operands with "
            "`preferred_element_type=int32` gives wrong values on XLA:CPU "
            "and XLA:GPU for an equation with a batch axis."
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


def _assert_same_store(test, store, expected):
    """`store` holds the keys, dtypes, shapes and bytes of `expected`."""
    test.assertEqual(list(store), list(expected))
    for key, value in expected.items():
        test.assertEqual(store[key].dtype, value.dtype, key)
        test.assertEqual(store[key].shape, value.shape, key)
        test.assertEqual(store[key].tobytes(), value.tobytes(), key)


def _spec_variable(layer, entry, weight_name):
    """The variable a spec entry names (`kernel` is held at `_kernel`)."""
    if entry == weight_name:
        return getattr(layer, f"_{entry}", None)
    return getattr(layer, entry, None)


def _variable_table(layer):
    return sorted(
        (v.name, tuple(v.shape), standardize_dtype(v.dtype))
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


def _code_step(view):
    """The real value of one code step at each entry of a view's weight."""
    shape = view.layout.unpacked_shape(view.codes.shape)
    others = view.read_tensors()[1:]
    views = [
        view.with_tensors((view.pack_image(ops.full(shape, code)),) + others)
        for code in (0.0, 1.0)
    ]
    zero, one = (_numpy(v.dequantize("float32")) for v in views)
    return one - zero


def _assert_round_trips(test, layer, x, y, name):
    """A model of `layer` loads back from `.keras` and `.weights.h5`."""
    model = models.Sequential([layer])
    model(x)
    path = os.path.join(test.get_temp_dir(), "model.keras")
    model.save(path)
    loaded = saving.load_model(path, custom_objects=CUSTOM_OBJECTS)
    test.assertEqual(loaded.layers[0].dtype_policy.name, name)
    test.assertAllClose(loaded(x), y)
    path = os.path.join(test.get_temp_dir(), "model.weights.h5")
    model.save_weights(path)
    fresh = models.Sequential([policy_built(layer)])
    fresh(x)
    fresh.load_weights(path)
    test.assertAllClose(fresh(x), y)


# The scale convention of each row: how the stored codes, scale and zero
# point give the weight. float8 stores no codes.
CONVENTIONS = {
    "int8": "divisor",
    "int8_weight_only": "divisor",
    "int4_per_channel": "divisor",
    "int4_grouped": "grouped",
    "float8": None,
    "ternary": "scalar",
    "gptq": "grouped",
    "awq": "grouped",
}

# `(scale_form, has_zero_point, grouped)` of the scheme of each convention.
DECLARED_SCHEMES = {
    "divisor": ("divisor", False, False),
    "grouped": ("multiplier", True, True),
    "scalar": ("multiplier", False, False),
}

# The range of the unpacked codes of each row that stores codes.
CODE_RANGES = {
    "int8": (-127, 127),
    "int8_weight_only": (-127, 127),
    "int4_per_channel": (-8, 7),
    "int4_grouped": (-8, 7),
    "ternary": (-1, 1),
    "gptq": (0, 15),
    "awq": (0, 15),
}


def _broadcast_scale(scale, scale_axes, shape):
    """Reshapes a stored scale to broadcast against codes of `shape`."""
    kept = [axis for axis in scale_axes if axis is not None]
    # An axis of size one that runs along no axis of the codes goes.
    sizes = [n for n, axis in zip(scale.shape, scale_axes) if axis is not None]
    scale = np.transpose(scale.reshape(sizes), np.argsort(kept))
    target = [1] * len(shape)
    for axis in kept:
        target[axis] = shape[axis]
    return scale.reshape(target)


def _documented_weight(view, convention):
    """The weight that a view's stored tensors give under `convention`.

    Computed in NumPy from the stored codes, scale, zero point, group
    index and input scales, without the view's `dequantize`:

    - `"divisor"`: `codes / scale`, the scale laid out by `scale_axes`.
    - `"grouped"`: `(codes - zero_point) * scale`, the zero point and the
      scale of each position's group (`g_idx`) along `axis`, divided by
      `input_scales` when the view has them.
    - `"scalar"`: `codes * scale`, one scalar scale.
    """
    codes = _numpy(view.layout.unpack(view.codes)).astype("float32")
    scale = _numpy(view.scale).astype("float32")
    if convention == "divisor":
        weight = codes / _broadcast_scale(scale, view.scale_axes, codes.shape)
    elif convention == "scalar":
        weight = codes * scale
    else:
        if view.g_idx is None:
            # The mode fixes the groups: position `i` is in group `i // G`.
            length = codes.shape[view.axis]
            groups = np.arange(length) // view.scheme.group_size
        else:
            groups = _numpy(view.g_idx).astype("int64")
        zero_point = _numpy(view.zero_point).astype("float32")
        weight = np.take(scale, groups, axis=view.axis) * (
            codes - np.take(zero_point, groups, axis=view.axis)
        )
        if view.input_scales is not None:
            shape = [1] * codes.ndim
            shape[view.axis] = -1
            weight = weight / _numpy(view.input_scales).reshape(shape)
    if view.permutation is None:
        return weight.reshape(view.shape)
    permuted = [view.shape[axis] for axis in view.permutation]
    return np.transpose(weight.reshape(permuted), np.argsort(view.permutation))


@pytest.mark.skipif(
    testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
)
class QuantizationConformanceTest(testing.TestCase):
    def _assert_unchanged(self, layer, before):
        """`layer` holds the variables of `before`, with the same values."""
        self.assertEqual(len(layer.weights), len(before))
        for variable, (expected, value) in zip(layer.weights, before):
            self.assertIs(variable, expected)
            self.assertAllEqual(variable, value)

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

        _assert_round_trips(self, layer, x, y, name)

    @parameterized.named_parameters(PAIRS)
    def test_stored_variables_follow_the_scale_convention(
        self, mode_name, kind
    ):
        case = MODES[mode_name]
        convention = CONVENTIONS[mode_name]
        weight_name = LAYERS[kind].weight_name
        strategy = strategy_registry.get_strategy(case.mode)
        layer = build_layer(kind)
        float_weights = [_numpy(getattr(layer, f"_{weight_name}")).copy()]
        if kind == "reversible_untied":
            float_weights.append(_numpy(layer.reverse_embeddings).copy())
        _quantize(layer, case, layer_inputs(kind, np.random.default_rng(0)))

        views = strategy.quantized_weights(layer)
        if convention is None:
            # float8 keeps the float kernel and stores no codes.
            self.assertEqual(views, ())
            self.assertAllEqual(
                getattr(layer, f"_{weight_name}"), float_weights[0]
            )
            return
        for view, float_weight in zip(views, float_weights, strict=True):
            # The view reads the layer's stored variables themselves.
            for name in (
                "codes",
                "scale",
                "zero_point",
                "g_idx",
                "input_scales",
            ):
                variable = getattr(view, name)
                if variable is not None:
                    self.assertTrue(
                        any(variable is v for v in layer.weights), msg=name
                    )
            scheme = view.scheme
            self.assertEqual(
                (
                    scheme.scale_form,
                    scheme.has_zero_point,
                    scheme.group_size is not None,
                ),
                DECLARED_SCHEMES[convention],
            )
            self.assertEqual(scheme.code_range, CODE_RANGES[mode_name])
            codes = _numpy(view.layout.unpack(view.codes))
            low, high = CODE_RANGES[mode_name]
            self.assertGreaterEqual(codes.min(), low)
            self.assertLessEqual(codes.max(), high)
            if convention == "scalar":
                self.assertEqual(tuple(view.scale.shape), ())
            weight = _documented_weight(view, convention)
            self.assertAllClose(
                weight,
                view.dequantize("float32"),
                atol=1e-6,
                rtol=1e-6,
                tpu_atol=1e-3,
                tpu_rtol=1e-3,
            )
            if case.max_error is not None:
                # `max_error` bounds the output error; on these rows it
                # also bounds the weight error (at most 0.12, for gptq).
                self.assertLess(
                    _relative_error(weight, float_weight), case.max_error
                )

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
            self._assert_unchanged(lora_first, before)
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
        # carries the update. A calibration mode rounds onto its calibrated
        # grid, which clips the update at the range of each group; this
        # update is large against a group of 4, so the save warns.
        merged = policy_built(layer)
        if case.mode in CALIBRATION_MODES:
            with self.assertWarnsRegex(UserWarning, "clipped"):
                store = _store(layer)
        else:
            store = _store(layer)
        merged.load_own_variables(store)
        self.assertFalse(merged.lora_enabled)
        weight = _dequantized(strategy, layer)
        update = _lora_update(layer, weight_name)
        error = _dequantized(strategy, merged) - weight - update
        if case.mode in CALIBRATION_MODES:
            # The merged weight is still closer to the quantized weight
            # plus the update than the quantized weight is.
            self.assertLess(np.linalg.norm(error), np.linalg.norm(update))
        else:
            # The merge encodes the merged weight with a fresh scale, so
            # each entry is within half a code step of it.
            step = _code_step(strategy.quantized_weights(merged)[0])
            atol = 1e-2 if testing.uses_tpu() else 1e-5
            self.assertLessEqual(np.max(np.abs(error) - 0.5 * step), atol)

    @parameterized.named_parameters(PAIRS)
    def test_lora_save_is_a_fixed_point(self, mode_name, kind):
        case = MODES[mode_name]
        strategy = strategy_registry.get_strategy(case.mode)
        layer = build_layer(kind)
        if not hasattr(layer, "enable_lora") or not strategy.supports_lora:
            self.skipTest("The pair has no LoRA.")
        _quantize(layer, case, layer_inputs(kind, np.random.default_rng(0)))

        def assert_fixed_point(expected):
            # A load leaves a zero update, so a save, a load into a layer
            # with LoRA and a save again give the same store.
            store = expected
            for _ in range(3):
                reloaded = policy_built(layer)
                reloaded.enable_lora(2)
                reloaded.load_own_variables(store)
                store = _store(reloaded)
                _assert_same_store(self, store, expected)

        # A zero update saves the stored values.
        expected = _store(layer)
        layer.enable_lora(2)
        _assert_same_store(self, _store(layer), expected)
        assert_fixed_point(expected)

        # The first save of a nonzero update merges it.
        _set_lora_factors(layer, LAYERS[kind].weight_name, scale=0.3)
        assert_fixed_point(_store(layer))

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
        self._assert_unchanged(layer, before)

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
            self.assertAllClose(
                logits, hidden @ table, atol=1e-4, rtol=1e-5, tpu_atol=1e-2
            )
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
        self._assert_unchanged(layer, before)
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
        self._assert_unchanged(layer, before)

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


@pytest.mark.skipif(
    testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
)
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
        _assert_round_trips(self, layer, x, y, "int8_from_float32")

        # `Model.quantize` calls the layer's own `quantize`.
        model = models.Sequential([layers.Input((8,)), ReleasedProtocolLayer()])
        report = model.quantize("int8", verbose=False)
        self.assertEqual(model.layers[0].quantization_mode, "int8")
        self.assertLen(report.quantized, 1)


class FixedGroupsTest(testing.TestCase):
    """A third-party layer gets the fixed group index of its mode from
    `add_group_index` and the calibration build."""

    @parameterized.named_parameters(("int4", "int4"), ("awq", "awq"))
    def test_third_party_layer_gets_fixed_groups(self, mode):
        """int4 and AWQ in groups of 64: the `g_idx` of the layer holds
        `k // 64`, its view has no `g_idx`, and a write of other groups
        changes no output."""
        layer = StandardProjection(32, use_bias=False)
        layer.build((None, 256))
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 256)).astype("float32")
        if mode == "int4":
            config = Int4QuantizationConfig(block_size=64)
            layer.quantize("int4", config=config)
        else:
            batch = rng.standard_normal((16, 256)).astype("float32")
            calibrate_layer(
                layer, calibration_config("awq", group_size=64), batch
            )
        self.assertAllEqual(layer.g_idx, np.arange(256) // 64)
        self.assertIsNone(layer._quantized_weight().g_idx)
        expected = _numpy(layer(x))
        weights = layer.get_weights()
        index = [v.name for v in layer.weights].index("g_idx")
        weights[index] = rng.permutation(np.arange(256) // 64)
        layer.set_weights(weights)
        self.assertAllEqual(_numpy(layer(x)), expected)

    def test_third_party_lookup_gets_fixed_groups(self):
        """A grouped int4 `TokenTable` holds `d // G`, and its view has no
        `g_idx`: column `d` is in group `d // G`."""
        layer = TokenTable(10, 8)
        layer.build()
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=4))
        self.assertAllEqual(layer.g_idx, np.arange(8) // 4)
        self.assertIsNone(layer._quantized_weight().g_idx)
        self.assertAllEqual(
            layer._quantized_weight().group_index(), [0] * 4 + [1] * 4
        )
