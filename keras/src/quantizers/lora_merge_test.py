import os

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.conformance_test import _assert_same_store
from keras.src.quantizers.conformance_test import _lora_update
from keras.src.quantizers.conformance_test import _set_lora_factors
from keras.src.quantizers.conformance_test import _store
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import LAYERS
from keras.src.quantizers.quantization_test_utils import build_layer


def _layer(kind, dtype=None, lora_rank=None):
    """A built layer of `kind` and its weight name."""
    if kind == "conv":
        layer = layers.Conv2D(6, 3, dtype=dtype, lora_rank=lora_rank)
        layer.build((None, 5, 5, 4))
        return layer, "kernel"
    layer = build_layer(kind, dtype=dtype, lora_rank=lora_rank)
    return layer, LAYERS[kind].weight_name


# Policies other than `float32`, the policy of the conformance test:
# (kind, dtype policy, `quantize` arguments).
_ZERO_UPDATE_CASES = (
    ("dense_float_mixed_bfloat16", "dense", "mixed_bfloat16", None),
    ("conv_float_mixed_bfloat16", "conv", "mixed_bfloat16", None),
    ("dense_int8_bfloat16", "dense", "bfloat16", ("int8",)),
    ("dense_int4_grouped_bfloat16", "dense", "bfloat16", ("int4", 4)),
    ("embedding_int8_bfloat16", "embedding", "bfloat16", ("int8",)),
)


class LoRAMergedSaveTest(testing.TestCase):
    @parameterized.named_parameters(*_ZERO_UPDATE_CASES)
    def test_zero_update_saves_the_stored_values(
        self, kind, dtype, quantization
    ):
        layer, _ = _layer(kind, dtype)
        if quantization is not None and quantization[0] == "int4":
            layer.quantize(
                config=Int4QuantizationConfig(block_size=quantization[1])
            )
        elif quantization is not None:
            layer.quantize(quantization[0])
        expected = _store(layer)
        layer.enable_lora(2)
        _assert_same_store(self, _store(layer), expected)

        # A save, load and save again of a layer with LoRA is a fixed point.
        store = _store(layer)
        policy = layer.dtype_policy.name
        for _ in range(3):
            reloaded, _ = _layer(kind, policy, lora_rank=2)
            reloaded.load_own_variables(store)
            store = _store(reloaded)
            _assert_same_store(self, store, expected)

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
            _set_lora_factors(layer, name, scale=1.0)
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
            _set_lora_factors(layer, name, scale=1.0)
            stored = _store(layer)["0"]
            forward = ops.convert_to_numpy(getattr(layer, name))
            self.assertEqual(stored.dtype, forward.dtype)
            self.assertEqual(stored.tobytes(), forward.tobytes())

    def test_conv_under_a_quantized_policy_saves_its_float_kernel(self):
        # A conv has no quantization geometry, so it holds no codes.
        layer = layers.Conv2D(6, 3, dtype="int8_from_float32")
        layer.build((None, 8, 8, 4))
        self.assertAllEqual(_store(layer)["0"], layer._kernel)

    def test_merged_weights_file_keeps_float32_under_mixed_policy(self):
        layer, _ = _layer("dense", "mixed_bfloat16")
        layer.enable_lora(2)
        _set_lora_factors(layer, "kernel", scale=1.0)
        expected = ops.convert_to_numpy(layer._kernel) + _lora_update(
            layer, "kernel"
        )
        model = models.Sequential([layers.Input((12,)), layer])
        path = os.path.join(self.get_temp_dir(), "merged.weights.h5")
        model.save_weights(path)

        reloaded = models.Sequential([layers.Input((12,)), layers.Dense(6)])
        reloaded.load_weights(path)
        self.assertAllClose(
            reloaded.layers[0].kernel, expected, atol=1e-6, rtol=1e-6
        )
