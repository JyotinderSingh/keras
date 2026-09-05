import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import mode_registry
from keras.src.quantizers import quantizers
from keras.src.quantizers.qtensor import Int2Quads
from keras.src.quantizers.qtensor import Int4Pairs
from keras.src.quantizers.qtensor import NoPack
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import TernaryTrits
from keras.src.quantizers.qtensor import WeightScheme
from keras.src.quantizers.quantization_config import Int4QuantizationConfig


def _int8_scheme(channel_axis=-1):
    return WeightScheme(
        bits=8,
        code_range=(-127, 127),
        channel_axis=channel_axis,
    )


class WeightSchemeTest(testing.TestCase):
    def test_fields_and_derived_properties(self):
        scheme = WeightScheme(
            bits=4,
            code_range=(-8, 7),
            zero_point_dtype="int8",
            group_size=32,
            group_axis=0,
        )
        self.assertEqual(scheme.bits, 4)
        self.assertEqual(scheme.code_range, (-8, 7))
        self.assertTrue(scheme.signed)
        self.assertFalse(scheme.symmetric)
        self.assertTrue(scheme.grouped)
        self.assertIn("group_size=32", repr(scheme))

        unsigned = WeightScheme(bits=4, code_range=(0, 15))
        self.assertFalse(unsigned.signed)
        self.assertTrue(unsigned.symmetric)
        self.assertFalse(unsigned.grouped)

    @parameterized.named_parameters(
        (
            "group_size_without_axis",
            dict(
                bits=4,
                code_range=(-8, 7),
                group_size=32,
            ),
            "given together",
        ),
        (
            "grouped_with_channel_axis",
            dict(
                bits=4,
                code_range=(-8, 7),
                group_size=32,
                group_axis=0,
                channel_axis=-1,
            ),
            "no per-channel axis",
        ),
    )
    def test_invalid_schemes_are_rejected(self, kwargs, message):
        with self.assertRaisesRegex(ValueError, message):
            WeightScheme(**kwargs)


class PackLayoutTest(testing.TestCase):
    @parameterized.named_parameters(
        ("int4_axis0", Int4Pairs, 0, (-8, 7), "int8"),
        ("int4_axis1", Int4Pairs, 1, (-8, 7), "int8"),
        ("int4_uint8", Int4Pairs, 0, (0, 15), "uint8"),
        ("int2_axis0", Int2Quads, 0, (-2, 1), "int8"),
        ("int2_uint8", Int2Quads, 1, (0, 3), "uint8"),
    )
    def test_pack_unpack_round_trip(self, layout_cls, axis, value_range, dtype):
        rng = np.random.default_rng(0)
        # Odd lengths exercise the padding on every axis.
        codes = rng.integers(value_range[0], value_range[1] + 1, (5, 7))
        codes = codes.astype(dtype)
        layout = layout_cls(axis=axis, orig_len=codes.shape[axis], dtype=dtype)
        packed = layout.pack(codes)
        self.assertEqual(
            packed.shape[axis], layout.packed_length(codes.shape[axis])
        )
        self.assertAllClose(layout.unpack(packed), codes)

    def test_ternary_round_trip(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(-1, 2, (11, 3)).astype("int8")
        layout = TernaryTrits(axis=0, orig_len=11)
        packed = layout.pack(codes)
        self.assertEqual(packed.shape, (3, 3))
        self.assertEqual(layout.packed_length(11), 3)
        self.assertAllClose(layout.unpack(packed), codes)

    def test_values_per_byte(self):
        self.assertEqual(NoPack().values_per_byte, 1)
        self.assertEqual(Int4Pairs(0, 4).values_per_byte, 2)
        self.assertEqual(Int2Quads(0, 4).values_per_byte, 4)
        self.assertEqual(TernaryTrits(0, 5).values_per_byte, 5)


class QTensorTest(testing.TestCase):
    def test_validates_zero_point_and_g_idx_against_scheme(self):
        codes = np.zeros((4, 2), "int8")
        scale = np.ones((2,), "float32")
        with self.assertRaisesRegex(ValueError, "zero_point"):
            QTensor(
                codes,
                scale,
                NoPack(),
                _int8_scheme(),
                (4, 2),
                zero_point=np.zeros((2,), "int8"),
            )
        grouped = WeightScheme(
            bits=4,
            code_range=(-8, 7),
            zero_point_dtype="int8",
            group_size=2,
            group_axis=0,
        )
        with self.assertRaisesRegex(ValueError, "g_idx"):
            QTensor(
                codes,
                scale,
                NoPack(),
                grouped,
                (4, 2),
                zero_point=np.zeros((2,), "int8"),
            )

    def test_per_channel_scale_divides(self):
        codes = np.array([[-127, 64], [3, 0]], "int8")
        scale = np.array([2.0, 4.0], "float32")
        view = QTensor(codes, scale, NoPack(), _int8_scheme(), (2, 2))
        self.assertAllClose(view.unpack(), codes)
        self.assertAllClose(view.dequantize(), codes / scale)

    def test_per_channel_scale_broadcasts_along_its_axis(self):
        codes = np.array([[2, 4], [6, 8]], "int8")
        scale = np.array([2.0, 4.0], "float32")
        # Scale runs along axis 0: rows are divided by 2 and 4.
        view = QTensor(codes, scale, NoPack(), _int8_scheme(0), (2, 2))
        self.assertAllClose(view.dequantize(), [[1.0, 2.0], [1.5, 2.0]])

    def test_scalar_scale(self):
        codes = np.array([[-1, 0, 1]], "int8")
        scheme = WeightScheme(bits=2, code_range=(-1, 1))
        view = QTensor(codes, np.float32(2.0), NoPack(), scheme, (1, 3))
        self.assertAllClose(view.dequantize(), [[-0.5, 0.0, 0.5]])

    def test_grouped_scheme_replays_group_map(self):
        # Two groups of two rows; group 1 has a non-zero zero point.
        codes = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], "int8")
        scale = np.array([[1.0, 10.0], [2.0, 20.0]], "float32")
        zero = np.array([[0, 0], [1, 1]], "int8")
        g_idx = np.array([0, 0, 1, 1], "float32")
        scheme = WeightScheme(
            bits=4,
            code_range=(-8, 7),
            zero_point_dtype="int8",
            group_size=2,
            group_axis=0,
        )
        view = QTensor(
            codes, scale, NoPack(), scheme, (4, 2), zero_point=zero, g_idx=g_idx
        )
        expected = quantizers.dequantize_grouped(
            codes, scale, zero, g_idx, group_axis=0
        )
        self.assertAllClose(view.dequantize(), expected)
        self.assertAllClose(
            view.dequantize(),
            [[1.0, 0.2], [3.0, 0.4], [2.0, 0.25], [3.0, 0.35]],
        )

    def test_input_scales_divide_each_input_row(self):
        codes = np.array([[1, 2], [3, 4], [5, 6]], "uint8")
        scheme = WeightScheme(bits=4, code_range=(0, 15))
        input_scales = np.array([1.0, 2.0, 4.0], "float32")
        view = QTensor(
            codes,
            np.float32(1.0),
            NoPack(),
            scheme,
            (3, 2),
            input_scales=input_scales,
        )
        self.assertAllClose(view.unpack(), codes)
        self.assertAllClose(view.dequantize(), codes / input_scales[:, None])

    def test_logical_shape_restores_an_nd_kernel(self):
        codes = np.arange(12, dtype="int8").reshape(4, 3)
        scale = np.ones((3,), "float32")
        view = QTensor(codes, scale, NoPack(), _int8_scheme(), (2, 2, 3))
        self.assertEqual(tuple(view.unpack().shape), (2, 2, 3))
        self.assertEqual(tuple(view.dequantize().shape), (2, 2, 3))
        self.assertEqual(view.num_values, 12)


class LayerViewTest(testing.TestCase):
    """The view built by each mode matches what the layers exposed before."""

    def _dense(self, mode, config=None, units=5, input_dim=7):
        layer = layers.Dense(units)
        layer.build((None, input_dim))
        layer.quantize(mode, config=config)
        return layer

    @parameterized.named_parameters(
        ("int8", "int8", None),
        ("int4_per_channel", "int4", Int4QuantizationConfig(block_size=-1)),
        ("int4_grouped", "int4", Int4QuantizationConfig(block_size=4)),
    )
    def test_dense_unpack_backs_the_kernel_property(self, mode, config):
        layer = self._dense(mode, config)
        view = layer._qtensor()
        self.assertEqual(view.logical_shape, (7, 5))
        self.assertAllClose(view.unpack(), layer.kernel)
        if mode == "int4":
            expected = quantizers.unpack_int4(
                layer._kernel, layer.units, axis=-1
            )
            self.assertAllClose(view.unpack(), expected)

    def test_dense_dequantize_matches_the_stored_scales(self):
        layer = self._dense("int4", Int4QuantizationConfig(block_size=4))
        view = layer._qtensor()
        unpacked = quantizers.unpack_int4(layer._kernel, layer.units, axis=-1)
        expected = quantizers.dequantize_grouped(
            ops.cast(unpacked, layer.compute_dtype),
            layer.kernel_scale,
            layer.kernel_zero,
            layer.g_idx,
            group_axis=0,
        )
        self.assertAllClose(view.dequantize(), expected)

    def test_dequantize_reconstructs_the_float_kernel(self):
        layer = layers.Dense(5)
        layer.build((None, 7))
        reference = ops.convert_to_numpy(layer._kernel)
        layer.quantize("int8")
        # int8 abs-max quantization is accurate to about one code step.
        self.assertAllClose(layer._qtensor().dequantize(), reference, atol=1e-2)

    @parameterized.named_parameters(
        ("int8", "int8", None),
        ("int4_per_channel", "int4", Int4QuantizationConfig(block_size=-1)),
        ("int4_grouped", "int4", Int4QuantizationConfig(block_size=4)),
    )
    def test_encode_is_what_quantize_stores(self, mode, config):
        # `encode` is the exact quantization math behind `quantize()`: fed
        # the float kernel, it must produce the stored form byte for byte.
        layer = layers.Dense(5)
        layer.build((None, 7))
        float_kernel = ops.convert_to_numpy(layer._kernel)
        descriptor = mode_registry.get_mode(mode)
        codes, scale, zero = descriptor.encode(layer, float_kernel, config)

        layer.quantize(mode, config=config)
        self.assertDType(codes, layer._kernel.dtype)
        self.assertAllClose(codes, layer._kernel)
        self.assertAllClose(scale, layer.kernel_scale)
        if zero is None:
            self.assertIsNone(layer._qtensor().zero_point)
        else:
            self.assertAllClose(zero, layer.kernel_zero)

    def test_einsum_dense_views(self):
        layer = layers.EinsumDense(
            "btd,dnh->btnh", output_shape=(None, 2, 3), bias_axes=None
        )
        layer.build((None, 4, 6))
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=2))
        view = layer._qtensor()
        self.assertEqual(view.logical_shape, (6, 2, 3))
        self.assertAllClose(view.unpack(), layer.kernel)
        self.assertEqual(tuple(view.dequantize().shape), (6, 2, 3))

    def test_embedding_views(self):
        layer = layers.Embedding(9, 4)
        layer.build()
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=2))
        view = layer._qtensor()
        self.assertEqual(view.logical_shape, (9, 4))
        self.assertAllClose(view.unpack(), layer.embeddings)
        unpacked = quantizers.unpack_int4(
            layer._embeddings, layer.output_dim, axis=-1
        )
        expected = quantizers.dequantize_grouped(
            ops.cast(unpacked, layer.compute_dtype),
            layer.embeddings_scale,
            layer.embeddings_zero,
            layer.g_idx,
            group_axis=-1,
        )
        self.assertAllClose(view.dequantize(), expected)

    def test_ternary_view(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 11))
        layer.quantize("ternary")
        view = layer._qtensor()
        self.assertIsInstance(view.layout, TernaryTrits)
        codes = ops.convert_to_numpy(view.unpack())
        self.assertEqual(codes.shape, (11, 4))
        self.assertTrue(np.isin(codes, [-1, 0, 1]).all())
        self.assertAllClose(
            view.dequantize(), codes / ops.convert_to_numpy(layer.kernel_scale)
        )

    def test_modes_without_codes_have_no_view(self):
        layer = layers.Dense(3)
        layer.build((None, 4))
        self.assertIsNone(layer._qtensor())
        layer.quantize("float8")
        self.assertIsNone(layer._qtensor())

    def test_uncalibrated_calibration_mode_has_no_view(self):
        from keras.src.quantizers.gptq_config import GPTQConfig

        layer = layers.Dense(3)
        layer.build((None, 4))
        layer.quantize(
            "gptq",
            config=GPTQConfig(dataset=None, tokenizer=None, group_size=2),
        )
        self.assertIsNone(layer._qtensor())
        # The float kernel is still what the property exposes.
        self.assertEqual(
            backend.standardize_dtype(layer.kernel.dtype), "float32"
        )


class QuantizationSummaryTest(testing.TestCase):
    def test_summary_counts_logical_parameters_exactly(self):
        # An odd output dim pads the packed int4 kernel; the summary must
        # count the 8 x 5 weights the codes stand for, not the padding.
        inputs = layers.Input([8])
        outputs = layers.Dense(5, name="d")(inputs)
        model = models.Model(inputs, outputs)
        model.quantize(
            "int4", config=Int4QuantizationConfig(block_size=4), verbose=False
        )
        summary = model.quantization_summary(verbose=False)
        self.assertIn("Quantized params : 40", summary)
        self.assertIn("weight store : int8 (24 bytes)", summary)
        self.assertIn("160 bytes float32", summary)
