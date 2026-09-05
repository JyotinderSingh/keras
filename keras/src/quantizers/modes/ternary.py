import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.mode_registry import require_geometry
from keras.src.quantizers.modes.common import add_matmul_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import TernaryTrits
from keras.src.quantizers.qtensor import WeightScheme
from keras.src.quantizers.quantization_config import TernaryQuantizationConfig
from keras.src.quantizers.quantizers import bitnet_ternary_values
from keras.src.quantizers.quantizers import pack_ternary


def _ternary_divisor(beta, dtype):
    """The stored scale for a BitNet beta: `1 / beta`, finite in `dtype`.

    An all-zero kernel has `beta == 0` and all-zero codes, so any finite
    scale reproduces it: store 1.0 rather than an infinite reciprocal.
    """
    if beta <= 0.0:
        return 1.0
    tiny = float(ml_dtypes.finfo(backend.standardize_dtype(dtype)).tiny)
    return 1.0 / max(float(beta), tiny)


class TernaryMode(QuantizationMode):
    """Ternary (BitNet b1.58) quantization: weights in `{-1, 0, +1}`.

    The ternarization rule (threshold and scale) is owned by the layer's
    geometry: the default is the BitNet b1.58 rule applied to the float
    kernel, and `TernaryDense` supplies its straight-through-estimator
    values instead.
    """

    name = "ternary"
    config_cls = TernaryQuantizationConfig

    def build(self, layer, input_shape, config):
        del config
        require_geometry(layer, self.name)
        input_dim, units = input_shape
        # Stored as `[in, packed(out)]` like every other packed projection:
        # five trits per byte (3^5 == 243 <= 256) along the output axis.
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(input_dim, (units + 4) // 5),
            # 121 = 1+3+9+27+81: byte whose five base-3 digits are all 0,
            # decoding to trit 0 (neutral). "zeros" (byte 0) has the same
            # digits but maps to trit -1, giving an all-minus-one kernel.
            initializer=initializers.Constant(121),
            dtype="uint8",
            trainable=False,
        )
        # Scalar divisor scale, `1 / beta` (BitNet b1.58); 1.0 with a
        # fixed threshold.
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=(),
            initializer="ones",
            trainable=False,
        )

    def qtensor(self, layer):
        geometry = require_geometry(layer, self.name)
        # Codes are exactly {-1, 0, +1}; the scalar scale is `1 / beta`, so
        # dividing by it applies the BitNet beta (the forward pass divides
        # the matmul output, which is the same product).
        return QTensor(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=TernaryTrits(axis=-1, orig_len=layer.units),
            scheme=WeightScheme(bits=2, code_range=(-1, 1)),
            logical_shape=geometry.weight_shape,
            compute_dtype=layer.compute_dtype,
        )

    def call(self, layer, inputs, **kwargs):
        # A storage format, not a compute win: the packed kernel is unpacked
        # to `{-1, 0, +1}` on every call and fed to a standard matmul, so
        # inference is slightly slower than a float `Dense` call. A native
        # ternary kernel reading the packed format would be needed for a
        # speedup.
        kernel = ops.cast(self.qtensor(layer).unpack(), layer.compute_dtype)
        x = ops.matmul(inputs, kernel)
        x = ops.divide(x, ops.cast(layer.kernel_scale, layer.compute_dtype))
        x = add_matmul_lora_delta(layer, inputs, x)
        return apply_bias_activation(layer, x)

    def encode(self, layer, weight, config=None):
        del config
        # The BitNet b1.58 rule on an arbitrary float weight (the LoRA-merged
        # kernel at save time). `quantize` instead applies the layer's own
        # rule through its geometry.
        codes, scale = bitnet_ternary_values(weight)
        packed, _, _ = pack_ternary(codes, axis=-1)
        scale = _ternary_divisor(scale, layer.variable_dtype)
        return packed, np.array(scale, dtype="float32"), None

    def quantize(self, layer, config):
        del config
        geometry = require_geometry(layer, self.name)
        kernel_shape = geometry.weight_shape
        # The geometry owns the ternarization rule: the BitNet b1.58 rule by
        # default, or the layer's own values (`TernaryDense` freezes exactly
        # the forward value of its straight-through kernel, so quantizing
        # does not change the layer's outputs).
        kernel_ternary, beta = geometry.ternary_values()
        packed_kernel, _, _ = pack_ternary(kernel_ternary, axis=-1)
        del layer._kernel
        layer.quantized_build(kernel_shape, "ternary")
        layer._kernel.assign(packed_kernel)
        layer.kernel_scale.assign(
            _ternary_divisor(beta, layer.kernel_scale.dtype)
        )
