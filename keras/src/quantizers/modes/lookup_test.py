"""int8 and int4 store an embeddings table through one `LookupHandlers`."""

import numpy as np
from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.modes.int8 import Int8Strategy
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.testing.test_utils import named_product


class ToyModeConfig(Int8QuantizationConfig):
    """The int8 config under the name of a toy mode."""

    def __init__(self, mode):
        super().__init__()
        self._mode = mode

    @property
    def mode(self):
        return self._mode


class MultiplierScaleInt8(Int8Strategy):
    """int8 with an ungrouped multiplier scale for the table."""

    name = "demo_multiplier"

    def _get_lookup_scheme(self, layer, config):
        return WeightScheme(code_range=(-127, 127), scale_form="multiplier")


class NoTableViewInt8(Int8Strategy):
    """int8 without a `QuantizedWeight` view of the table."""

    name = "demo_no_view"

    def _get_lookup_quantized_weight(self, layer, geometry):
        return None


class ToyModesEmbedding(layers.Embedding):
    """An `Embedding` that stores the toy modes as int8 does."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        for mode in (MultiplierScaleInt8.name, NoTableViewInt8.name):
            spec[mode] = spec["int8"]
        return spec


class LookupHandlersTest(testing.TestCase):
    @parameterized.named_parameters(
        ("multiplier_scale", MultiplierScaleInt8, "does not implement"),
        ("no_table_view", NoTableViewInt8, "cannot write its codes"),
    )
    def test_quantize_refuses_a_mode_the_handlers_cannot_serve(
        self, strategy_cls, message
    ):
        strategy_registry.register_quantization_strategy(strategy_cls)
        self.addCleanup(
            strategy_registry.unregister_quantization_strategy,
            strategy_cls.name,
        )
        layer = ToyModesEmbedding(10, 4)
        layer.build()
        before = ops.convert_to_numpy(layer.embeddings)
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.quantize(
                config=ToyModeConfig(strategy_cls.name), type_check=False
            )
        self.assertIsNone(layer.quantization_mode)
        self.assertAllEqual(layer.embeddings, before)

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

    @parameterized.named_parameters(
        named_product(
            block_size=(None, -1, 4),
            ids_shape=((), (5,), (2, 3), (2, 2, 2)),
            policy=("float32", "mixed_bfloat16"),
        )
    )
    def test_forward_is_the_rows_of_the_dequantized_table(
        self, block_size, ids_shape, policy
    ):
        # The lookup gathers the stored rows and dequantizes only those,
        # which gives the rows of the dequantized table bit for bit.
        rng = np.random.default_rng(0)
        layer = layers.Embedding(37, 13, dtype=policy)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if block_size is None:
            mode, config = "int8", None
        else:
            mode = "int4"
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        ids = rng.integers(0, 37, size=ids_shape)

        outputs = layer(ids)

        (view,) = strategy_registry.get_strategy(mode).quantized_weights(layer)
        with AutocastScope(layer.compute_dtype):
            table = view.dequantize(layer.compute_dtype)
        self.assertAllEqual(outputs, ops.take(table, ids, axis=0))

    @parameterized.named_parameters(
        ("tied", True),
        ("untied", False),
    )
    def test_weight_only_reverse_projects_through_the_dequantized_table(
        self, tie_weights
    ):
        # Per-channel int4 contracts with the codes and applies the scale
        # to the logits.
        rng = np.random.default_rng(0)
        layer = layers.ReversibleEmbedding(37, 13, tie_weights=tie_weights)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if not tie_weights:
            layer.reverse_embeddings.assign(rng.normal(size=(13, 37)))
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=-1))
        strategy = strategy_registry.get_strategy("int4")
        table = strategy._get_reverse_lookup_quantized_weight(layer, None)
        inputs = rng.normal(size=(3, 13)).astype("float32")

        logits = layer(inputs, reverse=True)

        self.assertAllClose(
            logits,
            ops.matmul(inputs, table.dequantize("float32")),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.named_parameters(
        named_product(block_size=(None, -1, 4), dtype=(None, "bfloat16"))
    )
    def test_forward_in_a_mixed_precision_model_reads_the_scale_dtype(
        self, block_size, dtype
    ):
        # An `Embedding` does not autocast, so inside a model under
        # `mixed_bfloat16` its scale reads in its own dtype (float32 for
        # the global policy, bfloat16 for a `bfloat16` layer), although
        # the variable reports `floatx()` there, and the rows are
        # dequantized in that dtype.
        dtype_policies.dtype_policy.set_dtype_policy("mixed_bfloat16")
        self.addCleanup(dtype_policies.dtype_policy.set_dtype_policy, "float32")
        rng = np.random.default_rng(0)
        layer = layers.Embedding(37, 13, dtype=dtype)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if block_size is None:
            mode, config = "int8", None
        else:
            mode = "int4"
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        inputs = layers.Input(shape=(None,), dtype="int32")
        model = models.Model(inputs, layer(inputs))
        ids = rng.integers(0, 37, size=(2, 3))

        outputs = model(ids)

        (view,) = strategy_registry.get_strategy(mode).quantized_weights(layer)
        scale_dtype = layer.variable_dtype
        self.assertDType(outputs, scale_dtype)
        self.assertAllEqual(
            outputs, ops.take(view.dequantize(scale_dtype), ids, axis=0)
        )
