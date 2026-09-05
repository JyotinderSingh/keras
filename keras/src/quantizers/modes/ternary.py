import ml_dtypes

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.mode_registry import require_geometry
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import TernaryTrits
from keras.src.quantizers.qtensor import WeightScheme
from keras.src.quantizers.quantization_config import TernaryQuantizationConfig
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
        # Five trits per byte (3^5 == 243 <= 256): ceil(input_dim / 5) rows.
        packed_rows = (input_dim + 4) // 5
        layer._packed_kernel = layer.add_weight(
            name="kernel",
            shape=(packed_rows, units),
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
        layer._orig_input_dim = input_dim

    def qtensor(self, layer):
        # Codes are exactly {-1, 0, +1}; the scalar scale is `1 / beta`, so
        # dividing by it applies the BitNet beta (the forward pass divides
        # the matmul output, which is the same product).
        return QTensor(
            codes=layer._packed_kernel,
            scale=layer.kernel_scale,
            layout=TernaryTrits(axis=0, orig_len=layer._orig_input_dim),
            scheme=WeightScheme(bits=2, code_range=(-1, 1)),
            logical_shape=(layer._orig_input_dim, layer.units),
            compute_dtype=layer.compute_dtype,
        )

    def call(self, layer, inputs, **kwargs):
        # Sparseskip inference path. Weights split into pos (+1) and neg (-1)
        # boolean masks so the matmul is structurally multiply-free — only
        # additions, subtractions, and zero-skips on kernel values.
        # Note: the packed kernel is unpacked to full float on every call and
        # fed to a standard matmul. Standard BLAS does not skip zero
        # multiplications, so there is no compute speedup over a plain Dense
        # call in this path; inference is slightly slower due to the unpack.
        # Realizing the full sparseskip speedup requires a native ternary
        # kernel that reads the packed format directly.
        k = self.qtensor(layer).unpack()
        pos = ops.cast(ops.equal(k, 1), layer.compute_dtype)
        neg = ops.cast(ops.equal(k, -1), layer.compute_dtype)
        x = ops.subtract(
            ops.matmul(inputs, pos),
            ops.matmul(inputs, neg),
        )
        x = ops.divide(x, ops.cast(layer.kernel_scale, layer.compute_dtype))
        if layer.bias is not None:
            x = ops.add(x, layer.bias)
        if layer.activation is not None:
            x = layer.activation(x)
        return x

    def quantize(self, layer, config):
        del config
        geometry = require_geometry(layer, self.name)
        kernel_shape = layer._kernel.shape
        # The geometry owns the ternarization rule: the BitNet b1.58 rule by
        # default, or the layer's own values (`TernaryDense` freezes exactly
        # the forward value of its straight-through kernel, so quantizing
        # does not change the layer's outputs).
        kernel_ternary, beta = geometry.ternary_values()
        packed_kernel, _, _ = pack_ternary(kernel_ternary, axis=0)
        del layer._kernel
        layer.quantized_build(kernel_shape, "ternary")
        layer._packed_kernel.assign(packed_kernel)
        layer.kernel_scale.assign(
            _ternary_divisor(beta, layer.kernel_scale.dtype)
        )
