import numpy as np

from keras.src import layers
from keras.src import testing
from keras.src.quantizers.awq import AWQCalibrator
from keras.src.quantizers.calibrator import Calibrator
from keras.src.quantizers.gptq import GPTQCalibrator


class _Bare(Calibrator):
    mode = "gptq"


class CalibratorTest(testing.TestCase):
    """The per-layer object of a calibration mode."""

    def test_resolves_the_two_d_view_from_the_geometry(self):
        dense = layers.Dense(32)
        dense.build((None, 16))
        for cls in (GPTQCalibrator, AWQCalibrator):
            calibrator = cls(dense)
            self.assertEqual((calibrator.rows, calibrator.columns), (16, 32))
            self.assertEqual(calibrator.num_samples, 0)
        # A 3-D einsum kernel is viewed as `(rows, columns)`.
        einsum = layers.EinsumDense("...h,hio->...io", output_shape=(4, 8))
        einsum.build((None, 16))
        calibrator = GPTQCalibrator(einsum)
        self.assertEqual((calibrator.rows, calibrator.columns), (16, 32))
        self.assertEqual(tuple(calibrator._kernel_view().shape), (16, 32))
        self.assertIs(calibrator.original_layer, einsum)
        # The rank guard lives on the geometry, once.
        four_d = layers.EinsumDense(
            "abc,cdef->abdef", output_shape=(3, 2, 3, 2)
        )
        four_d.build((None, 3, 4))
        with self.assertRaisesRegex(ValueError, "only supports 2D or 3D"):
            GPTQCalibrator(four_d)
        # An unbuilt layer reports the missing kernel, not an unsupported
        # type (the wording of the `AttributeError` varies by backend).
        with self.assertRaisesRegex(AttributeError, "kernel"):
            GPTQCalibrator(layers.Dense(4))

    def test_refuses_layers_outside_the_mode(self):
        ternary = layers.TernaryDense(4)
        ternary.build((None, 3))
        for cls, name in ((GPTQCalibrator, "GPTQ"), (AWQCalibrator, "AWQ")):
            with self.assertRaisesRegex(
                TypeError, f"Unsupported layer type for {name}"
            ):
                cls(layers.Layer())
            with self.assertRaisesRegex(
                TypeError, f"Unsupported layer type for {name}"
            ):
                cls(ternary)

    def test_protocol_methods_are_abstract(self):
        dense = layers.Dense(4)
        dense.build((None, 3))
        bare = _Bare(dense, None)
        x = np.ones((2, 3), "float32")
        with self.assertRaises(NotImplementedError):
            bare.observe(x)
        with self.assertRaises(NotImplementedError):
            bare.quantize()
        with self.assertRaises(NotImplementedError):
            bare.release()
        with self.assertRaises(NotImplementedError):
            _Bare.undersampling_warning([])

    def test_undersampling_is_declared_by_the_calibrator(self):
        self.assertEqual(GPTQCalibrator.warn_tokens_per_row, 4)
        self.assertIsNone(AWQCalibrator.warn_tokens_per_row)
        message = GPTQCalibrator.undersampling_warning(
            [("block/dense", 8, 256), ("block/other", 16, 256)]
        )
        self.assertIn("undersampled for 2 layer(s)", message)
        self.assertIn("block/dense (8 tokens for 256 input features)", message)
