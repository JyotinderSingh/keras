"""int8 and int4 store an embeddings table through one `LookupHandlers`."""

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.testing.test_utils import named_product


class LookupHandlersTest(testing.TestCase):
    @parameterized.named_parameters(("embedding", False), ("untied", True))
    def test_int8_table_has_no_zero_point_or_group_index(self, untied):
        if untied:
            layer = layers.ReversibleEmbedding(10, 4, tie_weights=False)
        else:
            layer = layers.Embedding(10, 4)
        layer.build()
        layer.quantize("int8")
        self.assertIsNone(layer.embeddings_zero)
        self.assertIsNone(layer.g_idx)
        if untied:
            self.assertIsNone(layer.reverse_embeddings_zero)

    @parameterized.named_parameters(
        named_product(
            tie_weights=(True, False),
            reverse_dtype=("bfloat16", "float16"),
        )
    )
    def test_grouped_reverse_keeps_the_table_in_float32(
        self, tie_weights, reverse_dtype
    ):
        # Below the compute dtype, only the inputs and the logits are cast
        # to `reverse_dtype`: the dequantized table keeps the precision of
        # the float32 scale.
        rng = np.random.default_rng(0)
        layer = layers.ReversibleEmbedding(
            37, 13, tie_weights=tie_weights, reverse_dtype=reverse_dtype
        )
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if not tie_weights:
            layer.reverse_embeddings.assign(rng.normal(size=(13, 37)))
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=4))
        views = strategy_registry.get_strategy("int4").quantized_weights(layer)
        if tie_weights:
            table = ops.transpose(views[0].dequantize("float32"))
        else:
            table = views[1].dequantize("float32")
        inputs = rng.normal(size=(3, 13)).astype("float32") * 4.0

        logits = layer(inputs, reverse=True)

        inputs = ops.cast(ops.cast(inputs, reverse_dtype), "float32")
        expected = ops.cast(ops.matmul(inputs, table), reverse_dtype)
        self.assertDType(logits, reverse_dtype)
        self.assertAllEqual(logits, expected)
