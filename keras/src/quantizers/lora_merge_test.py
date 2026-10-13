import os

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.quantization_config import Int4QuantizationConfig


def _layer(kind, dtype=None, lora_rank=None):
    """A built layer of `kind` and its weight name."""
    kwargs = {"dtype": dtype, "lora_rank": lora_rank}
    if kind == "dense":
        layer = layers.Dense(6, **kwargs)
        layer.build((None, 8))
        return layer, "kernel"
    if kind == "einsum":
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 6), bias_axes="d", **kwargs
        )
        layer.build((None, 3, 8))
        return layer, "kernel"
    if kind == "conv":
        layer = layers.Conv2D(6, 3, **kwargs)
        layer.build((None, 5, 5, 4))
        return layer, "kernel"
    if kind == "reversible":
        layer = layers.ReversibleEmbedding(10, 8, **kwargs)
    else:
        layer = layers.Embedding(10, 8, **kwargs)
    layer.build()
    return layer, "embeddings"


def _set_random(variables, seed):
    rng = np.random.default_rng(seed)
    for variable in variables:
        if variable.path.endswith("g_idx"):
            # Group indices stay valid as built.
            continue
        if "int" in variable.dtype:
            value = rng.integers(0, 16, variable.shape)
        else:
            value = rng.uniform(0.5, 1.5, variable.shape)
        variable.assign(value.astype(variable.dtype))


def _store(layer):
    store = {}
    layer.save_own_variables(store)
    return {key: ops.convert_to_numpy(value) for key, value in store.items()}


def _lora_update(layer, name):
    a = ops.convert_to_numpy(getattr(layer, f"lora_{name}_a"))
    b = ops.convert_to_numpy(getattr(layer, f"lora_{name}_b"))
    return (layer.lora_alpha / layer.lora_rank) * np.matmul(a, b)


# (kind, float dtype policy, `quantize` arguments or a policy to build from)
_ZERO_UPDATE_CASES = (
    ("dense_float_mixed_bfloat16", "dense", "mixed_bfloat16", None),
    ("conv_float_mixed_bfloat16", "conv", "mixed_bfloat16", None),
    ("dense_int8", "dense", None, ("int8",)),
    ("dense_int8_bfloat16", "dense", "bfloat16", ("int8",)),
    ("dense_int4_per_channel", "dense", None, ("int4", -1)),
    ("dense_int4_grouped", "dense", None, ("int4", 4)),
    ("dense_int4_grouped_bfloat16", "dense", "bfloat16", ("int4", 4)),
    ("einsum_int8", "einsum", None, ("int8",)),
    ("einsum_int4_grouped", "einsum", None, ("int4", 4)),
    ("embedding_int8", "embedding", None, ("int8",)),
    ("embedding_int8_bfloat16", "embedding", "bfloat16", ("int8",)),
    ("embedding_int4_grouped", "embedding", None, ("int4", 2)),
    ("reversible_int8", "reversible", None, ("int8",)),
    ("dense_gptq", "dense", None, "gptq/4/4_from_float32"),
    ("dense_awq", "dense", None, "awq/4/4_from_float32"),
)


class LoRAMergedSaveTest(testing.TestCase):
    def assertStoresEqual(self, store, expected):
        self.assertEqual(sorted(store), sorted(expected))
        for key, value in expected.items():
            self.assertEqual(store[key].dtype, value.dtype, key)
            self.assertEqual(store[key].shape, value.shape, key)
            self.assertEqual(store[key].tobytes(), value.tobytes(), key)

    def _zero_update_layer(self, kind, dtype, quantization):
        if isinstance(quantization, str):
            # The calibration modes come from a policy: values at random.
            layer, name = _layer(kind, quantization)
            _set_random(layer.weights, seed=1)
            return layer, name
        layer, name = _layer(kind, dtype)
        _set_random(layer.weights, seed=1)
        if quantization is None:
            return layer, name
        if quantization[0] == "int4":
            layer.quantize(
                config=Int4QuantizationConfig(block_size=quantization[1])
            )
        else:
            layer.quantize(quantization[0])
        return layer, name

    @parameterized.named_parameters(*_ZERO_UPDATE_CASES)
    def test_zero_update_saves_the_stored_values(
        self, kind, dtype, quantization
    ):
        layer, _ = self._zero_update_layer(kind, dtype, quantization)
        expected = _store(layer)
        layer.enable_lora(2)
        self.assertStoresEqual(_store(layer), expected)

        # A save, load and save again of a layer with LoRA is a fixed point.
        store = _store(layer)
        policy = layer.dtype_policy.name
        for _ in range(3):
            reloaded, _ = _layer(kind, policy, lora_rank=2)
            reloaded.load_own_variables(store)
            store = _store(reloaded)
            self.assertStoresEqual(store, expected)

    @parameterized.named_parameters(
        ("dense", "dense"),
        ("einsum", "einsum"),
        ("embedding", "embedding"),
        ("conv", "conv"),
    )
    def test_nonzero_update_merges_into_the_variable_dtype(self, kind):
        for policy in ("mixed_bfloat16", "mixed_float16"):
            layer, name = _layer(kind, policy)
            layer.enable_lora(2)
            _set_random(
                [getattr(layer, f"lora_{name}_{f}") for f in ("a", "b")],
                seed=2,
            )
            weight = ops.convert_to_numpy(getattr(layer, f"_{name}"))
            expected = weight + _lora_update(layer, name)
            stored = _store(layer)["0"]
            self.assertEqual(stored.dtype, np.float32)
            self.assertAllClose(stored, expected, atol=1e-6, rtol=1e-6)

    @parameterized.named_parameters(
        ("dense", "dense"),
        ("embedding", "embedding"),
        ("conv", "conv"),
    )
    def test_nonzero_update_merges_as_the_forward_without_mixed_policy(
        self, kind
    ):
        # Variable dtype equals compute dtype: one rounding, the same values
        # as the weight property.
        for policy in ("bfloat16", "float16", "float32"):
            layer, name = _layer(kind, policy)
            layer.enable_lora(2)
            _set_random(
                [getattr(layer, f"lora_{name}_{f}") for f in ("a", "b")],
                seed=2,
            )
            stored = _store(layer)["0"]
            forward = ops.convert_to_numpy(getattr(layer, name))
            self.assertEqual(stored.dtype, forward.dtype)
            self.assertEqual(stored.tobytes(), forward.tobytes())

    def test_merged_weights_file_keeps_float32_under_mixed_policy(self):
        layer, _ = _layer("dense", "mixed_bfloat16")
        layer.enable_lora(2)
        _set_random([layer.lora_kernel_a, layer.lora_kernel_b], seed=2)
        expected = ops.convert_to_numpy(layer._kernel) + _lora_update(
            layer, "kernel"
        )
        model = models.Sequential([layers.Input((8,)), layer])
        path = os.path.join(self.get_temp_dir(), "merged.weights.h5")
        model.save_weights(path)

        reloaded = models.Sequential([layers.Input((8,)), layers.Dense(6)])
        reloaded.load_weights(path)
        self.assertAllClose(
            reloaded.layers[0].kernel, expected, atol=1e-6, rtol=1e-6
        )
