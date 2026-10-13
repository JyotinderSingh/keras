import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import validate_and_resolve_config
from keras.src.quantizers.report import QuantizationReport


class _SubclassedDense(layers.Dense):
    pass


class _DenseWithOwnGeometry(layers.Dense):
    def _quantization_geometry(self):
        return super()._quantization_geometry()


def _model(*dense_layers):
    inputs = layers.Input((4,))
    x = inputs
    for layer in dense_layers:
        x = layer(x)
    return models.Model(inputs, x)


class QuantizeErrorsTest(testing.TestCase):
    @parameterized.named_parameters(
        (
            "einsum_dense",
            "einsum_dense",
            "ternary",
            "('int8', 'float8', 'int4', 'gptq', 'awq')",
        ),
        ("ternary_dense", "ternary_dense", "int8", "('ternary',)"),
        ("embedding", "embedding", "float8", "('int8', 'int4')"),
        ("reversible_embedding", "reversible", "ternary", "('int8', 'int4')"),
    )
    def test_mode_error_lists_the_modes_of_the_spec(self, kind, mode, modes):
        if kind == "einsum_dense":
            layer = layers.EinsumDense("ab,bc->ac", output_shape=3)
            layer.build((None, 4))
        elif kind == "ternary_dense":
            layer = layers.TernaryDense(3)
            layer.build((None, 4))
        elif kind == "embedding":
            layer = layers.Embedding(5, 4)
            layer.build()
        else:
            layer = layers.ReversibleEmbedding(5, 4)
            layer.build()
        with self.assertRaises(NotImplementedError) as context:
            layer.quantize(mode)
        self.assertEqual(
            str(context.exception),
            f"Invalid quantization mode. Expected one of {modes}. "
            f"Received: quantization_mode={mode}",
        )
        self.assertIsNone(layer.quantization_mode)

    def test_mode_error_of_a_layer_with_every_mode(self):
        layer = layers.Dense(3)
        layer.build((None, 4))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Expected one of \('int8', 'float8', 'int4', 'ternary', "
            r"'gptq', 'awq'\)\. Received: quantization_mode=float7",
        ):
            layer.quantized_build((None, 4), "float7")

    def test_type_check_refusal_names_the_owner_and_the_ways_out(self):
        layer = _SubclassedDense(3, name="sub")
        layer.build((None, 4))
        with self.assertRaises(NotImplementedError) as context:
            layer.quantize("int8")
        message = str(context.exception)
        self.assertIn("'_SubclassedDense'", message)
        self.assertIn("a subclass of `Dense`", message)
        self.assertIn("`type_check=False`", message)
        self.assertIn(
            "define `_quantization_geometry()` on `_SubclassedDense`", message
        )
        self.assertIsNone(layer.quantization_mode)

        # The first way out.
        layer.quantize("int8", type_check=False)
        self.assertEqual(layer.quantization_mode, "int8")

    def test_subclass_that_defines_its_geometry_quantizes(self):
        # The second way out: the subclass declares its own support.
        layer = _DenseWithOwnGeometry(3)
        layer.build((None, 4))
        x = np.random.default_rng(0).standard_normal((2, 4)).astype("float32")
        expected = ops.convert_to_numpy(layer(x))
        layer.quantize("int8")
        self.assertEqual(layer.quantization_mode, "int8")
        self.assertAllClose(layer(x), expected, atol=0.05, rtol=0.05)

    def test_model_quantize_skips_a_subclass_and_reports_it(self):
        model = _model(layers.Dense(4, name="a"), _SubclassedDense(4, name="b"))
        report = model.quantize("int8", verbose=False)
        self.assertEqual(model.get_layer("a").quantization_mode, "int8")
        self.assertIsNone(model.get_layer("b").quantization_mode)
        self.assertEqual(
            report.skipped_by_reason(QuantizationReport.SKIP_NO_SUPPORT),
            ["b"],
        )

    def test_int8_under_float16_gives_no_mixed_float16_advice(self):
        layer = layers.Dense(3, dtype="float16")
        layer.build((None, 4))
        with self.assertRaises(NotImplementedError) as context:
            layer.quantize("int8")
        message = str(context.exception)
        self.assertIn("compute_dtype='float16'", message)
        self.assertIn("'mixed_bfloat16'", message)
        self.assertNotIn("mixed_float16", message)
        self.assertIsNone(layer.quantization_mode)

    def test_model_quantize_skips_a_float16_layer_and_reports_it(self):
        model = _model(
            layers.Dense(4, name="a"),
            layers.Dense(4, dtype="float16", name="b"),
            layers.Dense(4, name="c"),
        )
        x = np.random.default_rng(0).standard_normal((2, 4)).astype("float32")
        model.predict(x, verbose=0)
        float16_kernel = ops.convert_to_numpy(model.get_layer("b").kernel)

        report = model.quantize("int8", verbose=False)
        self.assertEqual(model.get_layer("a").quantization_mode, "int8")
        self.assertIsNone(model.get_layer("b").quantization_mode)
        self.assertEqual(model.get_layer("c").quantization_mode, "int8")
        self.assertEqual(
            report.skipped_by_reason(QuantizationReport.SKIP_NO_SUPPORT),
            ["b"],
        )
        self.assertAllClose(model.get_layer("b").kernel, float16_kernel)
        self.assertAllClose(model.predict(x, verbose=0), model(x))

    def test_model_quantize_refuses_a_float16_model_before_any_change(self):
        model = _model(
            layers.Dense(4, dtype="float16"), layers.Dense(4, dtype="float16")
        )
        model.dtype_policy = "float16"
        with self.assertRaisesRegex(NotImplementedError, "float16"):
            model.quantize("int8", verbose=False)
        for layer in model.layers:
            self.assertIsNone(layer.quantization_mode)

    def test_string_config_that_contradicts_mode_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "Contradictory arguments: mode='int4' but config='int8'",
        ):
            validate_and_resolve_config("int4", "int8")
        self.assertIsInstance(
            validate_and_resolve_config(None, "int8"), Int8QuantizationConfig
        )
        self.assertIsInstance(
            validate_and_resolve_config("int8", "int8"), Int8QuantizationConfig
        )

        layer = layers.Dense(3)
        layer.build((None, 4))
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            layer.quantize("int4", config="int8")
        self.assertIsNone(layer.quantization_mode)

        model = _model(layers.Dense(4))
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            model.quantize("int4", config="int8", verbose=False)
        self.assertIsNone(model.layers[-1].quantization_mode)

        layer.quantize(config="int8")
        self.assertEqual(layer.quantization_mode, "int8")
