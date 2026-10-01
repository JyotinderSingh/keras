from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.lookup import LookupHandlers
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.quantizers import AbsMaxQuantizer

# Symmetric int8 codes with a per-channel divisor scale.
_INT8_SCHEME = WeightScheme(code_range=(-127, 127), scale_form="divisor")


class Int8Strategy(LookupHandlers, GeometryDispatchStrategy):
    """W8A8 dynamic quantization (int8 weights times int8 activations).

    One projection implementation serves every kernel contracted against
    its inputs: the geometry says how to contract, which axes the
    quantizers reduce over, and how a scale lines up with the kernel and
    with the outputs. An embeddings table goes through `LookupHandlers`,
    which this mode gives its scheme, its encoding and the default
    activation quantizer of the reverse projection.
    """

    name = "int8"
    config_cls = Int8QuantizationConfig
    geometry_families = ("projection", "lookup")

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
            shape=geometry.kernel_scale_shape(kernel_shape),
            initializer="ones",
            trainable=False,
        )

    def _call_projection(self, layer, geometry, inputs, training=None):
        @ops.custom_gradient
        def contract_with_inputs_gradient(inputs, kernel, kernel_scale):
            """Contracts against the int8 kernel with a custom gradient.

            Autodiff cannot differentiate through the int8 kernel, so the
            gradient with respect to the inputs is taken through the
            dequantized kernel.
            """
            quantized_weight = self._projection_view(
                geometry, kernel, kernel_scale
            )

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = quantized_weight.dequantize(layer.compute_dtype)
                return (
                    geometry.contract_grad(upstream, float_kernel),
                    None,
                    None,
                )

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
            return x, grad_fn

        x = contract_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(layer._kernel),
            # Read inside the autocast scope: on TensorFlow eager the gradient
            # runs after it, and the variable itself would then read float32.
            ops.convert_to_tensor(layer.kernel_scale.value),
        )
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=geometry.kernel_reduced_axes)
        )
        kernel_value, kernel_scale = weight_quantizer(weight, to_numpy=True)
        return (
            kernel_value,
            geometry.kernel_scale_for_storage(kernel_scale),
            None,
        )

    def _projection_view(self, geometry, codes, scale):
        """The view over the layer's variables or over traced tensors."""
        # A matmul kernel's scale is shared along its input axis. An einsum
        # kernel's is stored in the outputs' layout, and the geometry lays
        # it back out against the kernel.
        axis = geometry.kernel_scale_axis
        return QuantizedWeight(
            codes=codes,
            scale=scale,
            layout=NoPack(),
            scheme=_INT8_SCHEME,
            shape=geometry.weight_shape,
            axis=axis,
            align_scale=(
                None if axis is not None else geometry.kernel_scale_for_dequant
            ),
        )

    def _quantized_weight_projection(self, layer, geometry):
        return self._projection_view(
            geometry, layer._kernel, layer.kernel_scale
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

    def _lookup_scheme(self, layer, config):
        return _INT8_SCHEME

    def _default_reverse_inputs_quantizer(self):
        return AbsMaxQuantizer(axis=-1)

    def _encode_lookup(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=-1),
        )
        embeddings_value, embeddings_scale = weight_quantizer(
            weight, to_numpy=True
        )
        return embeddings_value, ops.squeeze(embeddings_scale, axis=-1), None
