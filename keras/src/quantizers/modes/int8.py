from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.common import (
    contract_with_float_kernel_gradient,
)
from keras.src.quantizers.modes.lookup import LookupHandlers
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.quantized_weight import lay_out_scale
from keras.src.quantizers.quantizers import AbsMaxQuantizer

# Symmetric int8 codes with a per-channel divisor scale.
_INT8_SCHEME = WeightScheme(code_range=(-127, 127), scale_form="divisor")


class Int8Strategy(LookupHandlers, GeometryDispatchStrategy):
    """W8A8 dynamic quantization (int8 weights times int8 activations).

    One projection implementation serves every kernel contracted against
    its inputs: the geometry says how to contract, which kernel axes the
    weight quantizer reduces over (`kernel_axes.contracted`), and the
    layout of the stored scale (`kernel_scale_axes`).

    An embeddings table goes through `LookupHandlers`, which this mode
    gives its scheme and the default activation quantizer of the reverse
    projection. The handlers derive the encoding from the scheme.
    """

    name = "int8"
    config_cls = Int8QuantizationConfig
    geometry_families = ("projection", "lookup")

    def check_compute_dtype(self, compute_dtype):
        if compute_dtype == "float16":
            raise NotImplementedError(
                f"Quantization mode='{self.name}' doesn't work well with "
                "compute_dtype='float16'. Use a dtype policy that computes "
                "in another dtype, such as 'mixed_bfloat16' or 'float32'."
            )

    # --- Projection (Dense, EinsumDense) ----------------------------------

    def _build_projection(self, layer, geometry, kernel_shape, config):
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config, AbsMaxQuantizer()
            )
        )
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=[
                1 if axis is None else kernel_shape[axis]
                for axis in geometry.kernel_scale_axes
            ],
            initializer="ones",
            trainable=False,
        )

    def _call_projection(self, layer, geometry, inputs, training=None):
        view = self._get_projection_quantized_weight(layer, geometry)

        def contract_codes(inputs, quantized_weight):
            # The int8 scale is stored in the outputs' layout, so it de-scales
            # the integer contraction directly.
            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                output_scale = ops.multiply(
                    geometry.align_inputs_scale(inputs_scale),
                    quantized_weight.scale,
                )
            else:
                # Weight-only: contract against the int8 kernel and de-scale
                # the outputs.
                output_scale = quantized_weight.scale
            x = geometry.contract(inputs, quantized_weight.codes)
            x = ops.cast(x, layer.compute_dtype)
            x = ops.divide(x, output_scale)
            return x

        x = contract_with_float_kernel_gradient(
            layer, geometry, view, inputs, contract_codes
        )
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=geometry.kernel_axes.contracted)
        )
        kernel_value, kernel_scale = weight_quantizer(weight, to_numpy=True)
        return (
            kernel_value,
            lay_out_scale(
                kernel_scale,
                range(len(weight.shape)),
                geometry.kernel_scale_axes,
            ),
            None,
        )

    def _get_projection_quantized_weight(self, layer, geometry):
        return QuantizedWeight(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=NoPack(),
            scheme=_INT8_SCHEME,
            shape=geometry.weight_shape,
            scale_axes=geometry.kernel_scale_axes,
        )

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, _ = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, self.name, config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)

    # --- Embeddings lookup (Embedding, ReversibleEmbedding) ---------------
    # The build, forward passes, views and conversion are `LookupHandlers`.

    def _get_lookup_scheme(self, layer, config):
        return _INT8_SCHEME

    def _get_default_reverse_inputs_quantizer(self):
        return AbsMaxQuantizer(axis=-1)
