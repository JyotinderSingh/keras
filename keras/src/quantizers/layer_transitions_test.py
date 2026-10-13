import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantizers import AbsMaxQuantizer


class _DenseSubclass(layers.Dense):
    pass


class _FailingFinalizeDense(layers.Dense):
    """A `Dense` that raises once the swap has built the mode's variables.

    The policy step comes after the strategy's `quantize`, or after the
    write-back of a calibration, so the layer is changed when it raises.
    """

    error = RuntimeError

    def _finalize_quantization_policy(self, strategy, config):
        raise self.error("Injected failure.")


class _InterruptedDense(_FailingFinalizeDense):
    error = KeyboardInterrupt


class _OwnLoraDense(layers.Dense):
    """A `Dense` whose own `enable_lora` does not check the mode.

    A custom layer's `enable_lora` need not call `_check_lora_supported`.
    """

    def enable_lora(self, rank):
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a", shape=(self.kernel_shape[0], rank)
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b", shape=(rank, self.units), initializer="ones"
        )
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = rank


class _FlagFirstLayer(layers.Layer):
    """A layer on the per-layer protocol of v3.12 to v3.15.

    Its own `quantized_build` sets `_is_quantized`, and its `quantize`
    then assigns the quantized policy to a float layer.
    """

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 3), name="w")

    def call(self, inputs):
        return ops.matmul(inputs, self.w)

    def quantized_build(self, input_shape, mode, config=None):
        self.scale = self.add_weight(
            shape=(3,), initializer="ones", trainable=False, name="scale"
        )
        self._is_quantized = True

    def quantized_call(self, inputs):
        return ops.matmul(inputs, self.w) * self.scale

    def quantize(self, mode, type_check=True, config=None):
        self._check_quantize_args(mode, self.compute_dtype)
        self._tracker.unlock()
        self.quantized_build(None, mode)
        self._tracker.lock()
        self.dtype_policy = f"{mode}_from_{self.dtype_policy.name}"


_DENSE_CLASSES = {
    "dense": layers.Dense,
    "lora_dense": layers.Dense,
    "frozen_dense": layers.Dense,
    "subclass": _DenseSubclass,
    "failing_dense": _FailingFinalizeDense,
    "interrupted_dense": _InterruptedDense,
}


def _built(kind):
    if kind in _DENSE_CLASSES:
        layer, inputs = _DENSE_CLASSES[kind](6), np.ones((2, 8), "float32")
        layer.build((None, 8))
        if kind == "lora_dense":
            layer.enable_lora(2)
        if kind == "frozen_dense":
            layer.trainable = False
    elif kind == "einsum":
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 6), bias_axes="d"
        )
        inputs = np.ones((2, 3, 8), "float32")
        layer.build((None, 3, 8))
    elif kind == "embedding":
        layer, inputs = layers.Embedding(10, 8), np.array([[1, 2]], "int32")
        layer.build()
    else:
        layer = layers.ReversibleEmbedding(10, 8, tie_weights=False)
        inputs = np.array([[1, 2]], "int32")
        layer.build()
    return layer, inputs


# The scale of a weight quantizer that reduces every kernel axis fits no
# scale variable, so the mode fails after it replaced the float weight.
_UNFIT_INT8 = Int8QuantizationConfig(
    weight_quantizer=AbsMaxQuantizer(axis=(0, 1))
)
_UNFIT_INT4 = Int4QuantizationConfig(
    block_size=None,
    weight_quantizer=AbsMaxQuantizer(
        axis=(0, 1), value_range=(-8, 7), output_dtype="int8"
    ),
)


class LayerTransitionsTest(testing.TestCase):
    def _snapshot(self, layer, inputs):
        # The first call sets attributes of its own, so it comes first.
        outputs = ops.convert_to_numpy(layer(inputs))
        return {
            "outputs": outputs,
            "attributes": dict(vars(layer)),
            "trainable": list(layer._trainable_variables),
            "non_trainable": list(layer._non_trainable_variables),
            "values": [ops.convert_to_numpy(v) for v in layer.weights],
        }

    def assertUnchanged(self, layer, inputs, snapshot):
        attributes = snapshot["attributes"]
        self.assertEqual(sorted(vars(layer)), sorted(attributes))
        for name, value in vars(layer).items():
            self.assertIs(value, attributes[name], name)
        self.assertEqual(
            list(map(id, layer._trainable_variables)),
            list(map(id, snapshot["trainable"])),
        )
        self.assertEqual(
            list(map(id, layer._non_trainable_variables)),
            list(map(id, snapshot["non_trainable"])),
        )
        for variable, value in zip(layer.weights, snapshot["values"]):
            self.assertAllEqual(variable, value)
        self.assertAllEqual(layer(inputs), snapshot["outputs"])
        self.assertTorchParametersLive(layer)

    def assertTorchParametersLive(self, module):
        # Torch lists only the live variables of `module` as parameters.
        if backend.backend() != "torch":
            return
        live = {id(v.value) for v in module.variables}
        parameters = [p for _, p in module.named_parameters()]
        self.assertLen(parameters, len(module.variables))
        self.assertTrue(all(id(p) in live for p in parameters))

    # --- The dtype policy setter ------------------------------------------

    @parameterized.named_parameters(
        ("calibration_mode", "dense", "gptq/4/128_from_float32", ValueError),
        ("unsupported_mode", "embedding", "float8_from_float32", None),
        ("lora", "lora_dense", "float8_from_float32", None),
        ("subclass", "subclass", "int8_from_float32", None),
    )
    def test_refused_policy_leaves_the_layer_unchanged(
        self, kind, policy, error
    ):
        layer, inputs = _built(kind)
        snapshot = self._snapshot(layer, inputs)
        with self.assertRaises(error or NotImplementedError):
            layer.dtype_policy = policy
        self.assertEqual(layer.dtype_policy.name, "float32")
        self.assertIsNone(layer.quantization_mode)
        self.assertUnchanged(layer, inputs, snapshot)

    @parameterized.named_parameters(("no_entry", None), ("entry", "float32"))
    def test_refused_policy_restores_the_dtype_policy_map(self, entry):
        policy_map = dtype_policies.DTypePolicyMap("float32")
        if entry is not None:
            policy_map["d"] = entry
            original = policy_map["d"]
        layer = layers.Dense(4, name="d", dtype=policy_map)
        layer.build((None, 8))
        with self.assertRaises(ValueError):
            layer.dtype_policy = "awq/4/128_from_float32"
        self.assertIs(layer.dtype_policy, policy_map)
        if entry is None:
            self.assertNotIn("d", policy_map)
        else:
            self.assertIs(policy_map["d"], original)
        self.assertIsNone(layer.quantization_mode)

    def test_policy_resolves_an_ambiguous_dtype_policy_map(self):
        # Two keys of the map match the path. The exact key that the
        # assignment stores takes precedence over both.
        policy_map = dtype_policies.DTypePolicyMap("float32")
        layer = layers.Dense(4, name="d", dtype=policy_map)
        layer.build((None, 8))
        policy_map["d.*"] = "float32"
        policy_map[".*d"] = "float32"
        layer.dtype_policy = "bfloat16"
        self.assertEqual(layer.compute_dtype, "bfloat16")

    @parameterized.named_parameters(
        ("int8_to_float", "dense", "int8", "float32"),
        ("int8_to_int4", "dense", "int8", "int4/128_from_float32"),
        ("int4_block_size", "dense", "int4", "int4/-1_from_float32"),
        ("int4_lookup_to_int8", "embedding", "int4", "int8_from_float32"),
        ("float8_to_float", "dense", "float8", "float32"),
        (
            "float8_history_length",
            "dense",
            "float8",
            dtype_policies.QuantizedFloat8DTypePolicy(
                "float8", "float32", amax_history_length=16
            ),
        ),
    )
    def test_quantized_layer_keeps_its_mode_and_parameters(
        self, kind, mode, policy
    ):
        layer, inputs = _built(kind)
        layer.quantize(mode)
        name = layer.dtype_policy.name
        snapshot = self._snapshot(layer, inputs)
        with self.assertRaisesRegex(ValueError, "only its source dtype"):
            layer.dtype_policy = policy
        self.assertEqual(layer.dtype_policy.name, name)
        self.assertUnchanged(layer, inputs, snapshot)

    def test_quantized_layer_accepts_another_source_dtype(self):
        layer, inputs = _built("dense")
        layer.quantize("int4")
        layer.dtype_policy = "int4/128_from_mixed_bfloat16"
        self.assertEqual(layer.quantization_mode, "int4")
        self.assertEqual(layer.compute_dtype, "bfloat16")
        self.assertEqual(layer(inputs).shape, (2, 6))

    def test_parameter_check_reads_no_policy_name(self):
        # The check compares the mode and its parameters, not the policy
        # name.
        policy_map = dtype_policies.DTypePolicyMap()
        policy_map["d"] = "int4/4_from_float32"
        layer = layers.Dense(4, name="d", dtype=policy_map)
        layer.build((None, 8))
        policy_map.get_config()
        entry = policy_map["d"]
        with self.assertRaisesRegex(ValueError, "only its source dtype"):
            layer.dtype_policy = "int4/-1_from_float32"
        self.assertIs(policy_map["d"], entry)
        layer.dtype_policy = "int4/4_from_mixed_bfloat16"
        self.assertEqual(layer.compute_dtype, "bfloat16")
        self.assertEqual(tuple(layer.kernel_scale.shape), (2, 4))

    def test_layer_that_sets_its_flag_before_its_policy(self):
        layer = _FlagFirstLayer()
        layer.build((None, 8))
        layer.quantize("int8")
        self.assertTrue(layer._is_quantized)
        self.assertEqual(layer.dtype_policy.name, "int8_from_float32")
        self.assertEqual(layer(np.ones((2, 8), "float32")).shape, (2, 3))

    # --- A failed quantization --------------------------------------------

    @parameterized.named_parameters(
        ("dense_int8", "dense", _UNFIT_INT8),
        ("dense_int4", "dense", _UNFIT_INT4),
        ("einsum_int8", "einsum", _UNFIT_INT8),
        ("embedding_int4", "embedding", _UNFIT_INT4),
        ("untied_reversible_int8", "reversible", _UNFIT_INT8),
        ("frozen_dense_int8", "frozen_dense", _UNFIT_INT8),
        ("lora_dense_int8", "lora_dense", _UNFIT_INT8),
    )
    def test_failed_quantize_restores_the_float_layer(self, kind, config):
        layer, inputs = _built(kind)
        snapshot = self._snapshot(layer, inputs)
        with self.assertRaises(ValueError):
            layer.quantize(config=config)
        self.assertUnchanged(layer, inputs, snapshot)
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(layer._is_quantized)
        # Each variable is back in its own store, so unfreezing the layer
        # and moving its weights work as on a float layer.
        layer.trainable = True
        fresh, _ = _built(kind)
        fresh.trainable = True
        self.assertEqual(
            [v.name for v in layer.trainable_variables],
            [v.name for v in fresh.trainable_variables],
        )
        fresh.set_weights(layer.get_weights())
        layer.quantize(config.mode)
        self.assertEqual(layer.quantization_mode, config.mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
        ("ternary", "ternary"),
    )
    def test_failure_after_the_mode_built_its_variables(self, mode):
        layer, inputs = _built("failing_dense")
        snapshot = self._snapshot(layer, inputs)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            layer.quantize(mode, type_check=False)
        self.assertUnchanged(layer, inputs, snapshot)
        self.assertIsNone(layer.quantization_mode)

    def test_interrupted_quantize_restores_the_float_layer(self):
        layer, inputs = _built("interrupted_dense")
        snapshot = self._snapshot(layer, inputs)
        with self.assertRaises(KeyboardInterrupt):
            layer.quantize("int8", type_check=False)
        self.assertUnchanged(layer, inputs, snapshot)
        self.assertIsNone(layer.quantization_mode)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_failure_inside_the_calibration_swap(self, mode):
        # The calibrator's write-back swaps the float layer through the
        # same transaction as `Layer.quantize`. The failure comes after the
        # swap assigned the codes and deleted the float kernel.
        layer, inputs = _built("failing_dense")
        snapshot = self._snapshot(layer, inputs)
        config = calibration_config(mode, group_size=-1)
        calibrator = calibrate_layer(layer, config, inputs, solve=False)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            calibrator.quantize()
        self.assertUnchanged(layer, inputs, snapshot)
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="TensorFlow object-graph checkpoints.",
    )
    def test_restored_layer_checkpoints_like_a_float_layer(self):
        import tensorflow as tf

        def model(cls):
            inputs = layers.Input((8,))
            dense = cls(6, name="d")
            return models.Model(inputs, dense(inputs)), dense

        for mode in ("int8", "float8"):
            model_a, dense = model(_FailingFinalizeDense)
            with self.assertRaises(RuntimeError):
                dense.quantize(mode, type_check=False)
            path = tf.train.Checkpoint(model=model_a).save(
                os.path.join(self.get_temp_dir(), mode)
            )
            model_b, _ = model(layers.Dense)
            status = tf.train.Checkpoint(model=model_b).restore(path)
            status.assert_consumed()

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="TensorFlow SavedModel export.",
    )
    def test_restored_layer_exports_to_a_saved_model(self):
        # The int8 swap of an `EinsumDense` sets list attributes, whose
        # names TensorFlow records for the export. The restore deletes the
        # attributes and their names.
        inputs = layers.Input((3, 8))
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 6), bias_axes="d"
        )
        model = models.Model(inputs, layer(inputs))
        with self.assertRaises(ValueError):
            layer.quantize(config=_UNFIT_INT8)
        model.export(
            os.path.join(self.get_temp_dir(), "exported"),
            format="tf_saved_model",
        )

    # --- A failed `Model.quantize` ----------------------------------------

    def _failed_model_quantize(self, model):
        # `d2` raises after its swap. It is a `Dense` subclass, so the call
        # skips the type check.
        data = np.ones((2, 8), "float32")
        model.predict(data, verbose=0)
        second = model.get_layer("d2")
        snapshot = self._snapshot(second, data)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            model.quantize("int8", type_check=False)
        self.assertEqual(model.get_layer("d1").quantization_mode, "int8")
        self.assertUnchanged(second, data, snapshot)
        self.assertIsNone(model.predict_function)
        self.assertAllClose(model.predict(data, verbose=0), model(data))
        self.assertTorchParametersLive(model)

    def test_failed_model_quantize_resets_the_compiled_functions(self):
        inputs = layers.Input((8,))
        x = layers.Dense(8, name="d1")(inputs)
        outputs = _FailingFinalizeDense(8, name="d2")(x)
        model = models.Model(inputs, outputs)
        self._failed_model_quantize(model)

    def test_failed_sequential_quantize_resets_the_compiled_functions(self):
        model = models.Sequential(
            [
                layers.Input((8,)),
                layers.Dense(8, name="d1"),
                _FailingFinalizeDense(8, name="d2"),
            ]
        )
        self._failed_model_quantize(model)

    # --- LoRA in a mode without LoRA support ------------------------------

    @parameterized.named_parameters(
        ("float8", "float8"), ("ternary", "ternary")
    )
    def test_call_refuses_lora_in_a_mode_without_lora_support(self, mode):
        # The forward pass of the mode has no term for the LoRA update.
        layer = _OwnLoraDense(6)
        layer.build((None, 8))
        layer.quantize(mode, type_check=False)
        layer.enable_lora(2)
        with self.assertRaisesRegex(NotImplementedError, "lora is not"):
            layer(np.ones((2, 8), "float32"))

    # --- The load check ---------------------------------------------------

    @parameterized.named_parameters(
        ("float_lora_from_int8", "dense", None, "int8"),
        ("int8_lora_from_float", "dense", "int8", None),
        ("embedding_lora_from_int8", "embedding", None, "int8"),
        ("int4_lora_from_int8", "dense", "int4", "int8"),
    )
    def test_load_check_runs_with_lora(self, kind, mode, stored_mode):
        source, _ = _built(kind)
        if stored_mode is not None:
            source.quantize(stored_mode)
        store = {}
        source.save_own_variables(store)
        layer, _ = _built(kind)
        if mode is not None:
            layer.quantize(mode)
        layer.enable_lora(2)
        with self.assertRaisesRegex(ValueError, "expected .* variables"):
            layer.load_own_variables(store)

    @parameterized.named_parameters(
        ("float", "dense", None),
        ("int8", "dense", "int8"),
        ("int4", "dense", "int4"),
        ("embedding_int8", "embedding", "int8"),
    )
    def test_lora_layer_loads_its_merged_store(self, kind, mode):
        layer, inputs = _built(kind)
        if mode is not None:
            layer.quantize(mode)
        layer.enable_lora(2)
        store = {}
        layer.save_own_variables(store)
        target, _ = _built(kind)
        if mode is not None:
            target.quantize(mode)
        target.enable_lora(2)
        target.load_own_variables(store)
        self.assertAllClose(target(inputs), layer(inputs))
