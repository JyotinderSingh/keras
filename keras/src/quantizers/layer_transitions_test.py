import os
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer


class _DenseSubclass(layers.Dense):
    pass


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


def _built(kind):
    if kind in ("dense", "lora_dense", "frozen_dense", "subclass"):
        cls = _DenseSubclass if kind == "subclass" else layers.Dense
        layer, inputs = cls(6), np.ones((2, 8), "float32")
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


def _fail_after_quantize(mode, names=None):
    """Makes `mode`'s `quantize` raise after it changed the layer."""
    strategy_cls = type(strategy_registry.get_strategy(mode))
    original = strategy_cls.quantize

    def quantize(self, layer, config):
        original(self, layer, config)
        if names is None or layer.name in names:
            raise RuntimeError("Injected failure.")

    return mock.patch.object(strategy_cls, "quantize", quantize)


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
        if backend.backend() == "torch":
            live = {id(v.value) for v in layer.variables}
            parameters = [p for _, p in layer.named_parameters()]
            self.assertLen(parameters, len(layer.variables))
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
        # `DTypePolicyMap.get_config` can clear the names of the live
        # entries; the check compares the mode and its parameters.
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
        ("int8", "int8", None),
        ("int4", "int4", None),
        ("float8", "float8", None),
        ("ternary", "ternary", None),
        ("gptq", "gptq", GPTQConfig),
        ("awq", "awq", AWQConfig),
    )
    def test_failure_after_the_mode_built_its_variables(self, mode, cls):
        layer, inputs = _built("dense")
        snapshot = self._snapshot(layer, inputs)
        config = None
        if cls is not None:
            config = cls(dataset=["a"], tokenizer=lambda x: np.zeros((1, 4)))
        with _fail_after_quantize(mode):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                layer.quantize(mode, config=config)
        self.assertUnchanged(layer, inputs, snapshot)
        self.assertIsNone(layer.quantization_mode)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="TensorFlow object-graph checkpoints.",
    )
    def test_restored_layer_checkpoints_like_a_float_layer(self):
        import tensorflow as tf

        def model():
            inputs = layers.Input((8,))
            dense = layers.Dense(6, name="d")
            return models.Model(inputs, dense(inputs)), dense

        for mode in ("int8", "float8"):
            model_a, dense = model()
            with _fail_after_quantize(mode):
                with self.assertRaises(RuntimeError):
                    dense.quantize(mode)
            path = tf.train.Checkpoint(model=model_a).save(
                os.path.join(self.get_temp_dir(), mode)
            )
            model_b, _ = model()
            status = tf.train.Checkpoint(model=model_b).restore(path)
            status.assert_consumed()

    # --- A failed `Model.quantize` ----------------------------------------

    def _failed_model_quantize(self, model):
        data = np.ones((2, 8), "float32")
        model.predict(data, verbose=0)
        second = model.get_layer("d2")
        snapshot = self._snapshot(second, np.ones((2, 8), "float32"))
        with _fail_after_quantize("int8", names=("d2",)):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                model.quantize("int8")
        self.assertEqual(model.get_layer("d1").quantization_mode, "int8")
        self.assertUnchanged(second, np.ones((2, 8), "float32"), snapshot)
        self.assertIsNone(model.predict_function)
        self.assertAllClose(model.predict(data, verbose=0), model(data))

    def test_failed_model_quantize_resets_the_compiled_functions(self):
        inputs = layers.Input((8,))
        x = layers.Dense(8, name="d1")(inputs)
        outputs = layers.Dense(8, name="d2")(x)
        model = models.Model(inputs, outputs)
        self._failed_model_quantize(model)
        if backend.backend() == "torch":
            live = {id(v.value) for v in model.variables}
            parameters = [p for _, p in model.named_parameters()]
            self.assertLen(parameters, len(model.variables))
            self.assertTrue(all(id(p) in live for p in parameters))

    def test_failed_sequential_quantize_resets_the_compiled_functions(self):
        model = models.Sequential(
            [
                layers.Input((8,)),
                layers.Dense(8, name="d1"),
                layers.Dense(8, name="d2"),
            ]
        )
        self._failed_model_quantize(model)

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="torch parameters only."
    )
    @pytest.mark.skip(
        reason="`Model._post_quantize` does not re-track "
        "`Sequential._functional`; fixed by the branch "
        "`fix-torch-post-quantize-sequential`."
    )
    def test_failed_sequential_quantize_retracks_torch_params(self):
        model = models.Sequential(
            [
                layers.Input((8,)),
                layers.Dense(8, name="d1"),
                layers.Dense(8, name="d2"),
            ]
        )
        self._failed_model_quantize(model)
        live = {id(v.value) for v in model.variables}
        parameters = [p for _, p in model.named_parameters()]
        self.assertLen(parameters, len(model.variables))
        self.assertTrue(all(id(p) in live for p in parameters))

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
