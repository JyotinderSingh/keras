from keras.src import backend
from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchMode
from keras.src.quantizers.modes.common import add_einsum_lora_delta
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import add_matmul_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.qtensor import NoPack
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import WeightScheme
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer


def _int8_scheme(channel_axis):
    """Symmetric int8 codes with a per-channel divisor scale."""
    return WeightScheme(
        bits=8,
        code_range=(-127, 127),
        channel_axis=channel_axis,
    )


class Int8Mode(GeometryDispatchMode):
    """W8A8 dynamic quantization (int8 weights times int8 activations)."""

    name = "int8"
    config_cls = Int8QuantizationConfig

    # --- Matmul projection (Dense) ----------------------------------------

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
            shape=(layer.units,),
            initializer="ones",
            trainable=False,
        )

    def _call_projection(self, layer, inputs, training=None):
        @ops.custom_gradient
        def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
            """Custom gradient function to handle the int8 quantized weights.

            Automatic differentiation will not know how to handle the int8
            quantized weights. So a custom gradient function is needed to
            handle the int8 quantized weights.

            The custom gradient function will use the dequantized kernel to
            compute the gradient.
            """

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=layer.compute_dtype),
                    kernel_scale,
                )
                inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
                return (inputs_grad, None, None)

            output_scale = kernel_scale
            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(inputs, axis=-1)
                output_scale = ops.multiply(output_scale, inputs_scale)

            x = ops.matmul(inputs, kernel)
            # De-scale outputs
            x = ops.cast(x, layer.compute_dtype)
            x = ops.divide(x, output_scale)
            return x, grad_fn

        x = matmul_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(layer._kernel),
            ops.convert_to_tensor(layer.kernel_scale),
        )
        x = add_matmul_lora_delta(layer, inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=0)
        )
        kernel_value, kernel_scale = weight_quantizer(weight, to_numpy=True)
        return kernel_value, ops.squeeze(kernel_scale, axis=0), None

    def _qtensor_projection(self, layer, geometry):
        return QTensor(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=NoPack(),
            scheme=_int8_scheme(channel_axis=-1),
            logical_shape=layer._kernel.shape,
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, _ = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        # Build variables for int8 mode
        layer.quantized_build(kernel_shape, "int8", config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)

    # --- Einsum projection (EinsumDense) ----------------------------------

    def _build_einsum(self, layer, geometry, kernel_shape, config):
        layer._set_quantization_info()
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config,
                AbsMaxQuantizer(),
            )
        )
        # If the config provided a default AbsMaxQuantizer, we need to
        # override the axis to match the equation's reduction axes.
        layer.quantization_axis = tuple(layer._input_reduced_axes)
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        kernel_scale_shape = layer._get_kernel_scale_shape(kernel_shape)
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=kernel_scale_shape,
            initializer="ones",
            trainable=False,
        )

    def _call_einsum(self, layer, inputs, training=None):
        @ops.custom_gradient
        def einsum_with_inputs_gradient(inputs, kernel, kernel_scale):
            """Performs int8 quantized einsum with a custom gradient.

            Computes the einsum operation with quantized inputs and a
            quantized kernel, then de-quantizes the result.

            Also computes the gradient with respect to the original,
            full-precision inputs by using a de-quantized kernel.

            Args:
                inputs: The full-precision input tensor.
                kernel: The int8 quantized kernel tensor.
                kernel_scale: The float32 scale factor for the kernel.

            Returns:
                A tuple `(output, grad_fn)`:
                    `output`: The de-quantized result of the einsum
                        operation.
                    `grad_fn`: The custom gradient function for the backward
                        pass.

            Raises:
                ValueError: If the quantization mode is not supported.
            """

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                # De-scale kernel
                _kernel_scale = kernel_scale
                _kernel_scale = layer._adjust_scale_for_dequant(_kernel_scale)
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=layer.compute_dtype),
                    _kernel_scale,
                )
                # From https://stackoverflow.com/a/47609896
                inputs_grad = ops.einsum(
                    layer._custom_gradient_equation, upstream, float_kernel
                )
                return (inputs_grad, None, None)

            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=layer.quantization_axis
                )
                # Align `inputs_scale` axes with the output
                # for correct broadcasting
                inputs_scale = layer._adjust_scale_for_quant(
                    inputs_scale, "input"
                )
                x = ops.einsum(layer.equation, inputs, kernel)
                # De-scale outputs
                x = ops.cast(x, layer.compute_dtype)
                x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            else:
                # Weight-only quantization: dequantize kernel and use float
                # einsum. This is a workaround for PyTorch's einsum which
                # doesn't support mixed-precision inputs (float input,
                # int8 kernel).
                if backend.backend() == "torch":
                    kernel_scale = layer._adjust_scale_for_dequant(kernel_scale)
                    float_kernel = ops.divide(
                        ops.cast(kernel, dtype=layer.compute_dtype),
                        kernel_scale,
                    )
                    x = ops.einsum(layer.equation, inputs, float_kernel)
                else:
                    x = ops.einsum(layer.equation, inputs, kernel)
                    # De-scale outputs
                    x = ops.cast(x, layer.compute_dtype)
                    x = ops.divide(x, kernel_scale)
            return x, grad_fn

        x = einsum_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(layer._kernel),
            ops.convert_to_tensor(layer.kernel_scale),
        )
        x = add_einsum_lora_delta(layer, inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_einsum(self, layer, geometry, weight, config):
        layer._set_quantization_info()
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=layer._kernel_reduced_axes),
        )
        kernel_value, kernel_scale = weight_quantizer(weight, to_numpy=True)
        kernel_scale = layer._adjust_scale_for_quant(kernel_scale, "kernel")
        return kernel_value, kernel_scale, None

    def _qtensor_einsum(self, layer, geometry):
        # The stored scale has the equation's squeezed/transposed layout;
        # the layer knows how to align it with the N-D kernel again.
        return QTensor(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            layout=NoPack(),
            scheme=_int8_scheme(channel_axis=None),
            logical_shape=layer._kernel.shape,
            align_scale=layer._adjust_scale_for_dequant,
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_einsum(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, _ = self._encode_einsum(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, "int8", config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)

    # --- Embeddings lookup (Embedding, ReversibleEmbedding) ---------------

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        layer._embeddings = layer.add_weight(
            name="embeddings",
            shape=embeddings_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        # We choose to reduce the axis of `output_dim` because, typically,
        # `input_dim` is larger than `output_dim`. This reduces quantization
        # error.
        layer.embeddings_scale = layer.add_weight(
            name="embeddings_scale",
            shape=(layer.input_dim,),
            initializer="ones",
            trainable=False,
        )
        if geometry.reversible:
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(
                    config, AbsMaxQuantizer(axis=-1)
                )
            )
            if not layer.tie_weights:
                layer.reverse_embeddings = layer.add_weight(
                    name="reverse_embeddings",
                    shape=(layer.output_dim, layer.input_dim),
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )
                layer.reverse_embeddings_scale = layer.add_weight(
                    name="reverse_embeddings_scale",
                    shape=(layer.input_dim,),
                    initializer="ones",
                    trainable=False,
                )

    def _call_lookup(self, layer, inputs, training=None):
        # We cannot update quantized layer._embeddings, so the custom
        # gradient is not needed
        inputs = cast_lookup_inputs(inputs)
        embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
        outputs = ops.take(layer._embeddings, inputs, axis=0)
        # De-scale outputs
        outputs = ops.divide(
            ops.cast(outputs, dtype=layer.compute_dtype),
            ops.expand_dims(embeddings_scale, axis=-1),
        )
        return add_lookup_lora_delta(layer, inputs, outputs)

    def _call_reversible_lookup(self, layer, inputs, reverse=False):
        if not reverse:
            return self._call_lookup(layer, inputs)
        else:
            if layer.tie_weights:
                kernel = ops.transpose(layer._embeddings)
                scale = ops.transpose(layer.embeddings_scale)
            else:
                kernel = layer.reverse_embeddings
                scale = layer.reverse_embeddings_scale
            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(inputs)
            else:
                inputs_scale = ops.ones((1,), dtype=layer.compute_dtype)
            logits = ops.matmul(inputs, kernel)
            # De-scale outputs
            logits = ops.cast(logits, layer.compute_dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
            return apply_logit_soft_cap(layer, logits)

    def _encode_lookup(self, layer, geometry, weight, config):
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=-1),
        )
        embeddings_value, embeddings_scale = weight_quantizer(
            weight, to_numpy=True
        )
        return embeddings_value, ops.squeeze(embeddings_scale, axis=-1), None

    def _qtensor_lookup(self, layer, geometry):
        return QTensor(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            layout=NoPack(),
            scheme=_int8_scheme(channel_axis=0),
            logical_shape=(layer.input_dim, layer.output_dim),
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = (layer.input_dim, layer.output_dim)
        embeddings_value, embeddings_scale, _ = self._encode_lookup(
            layer, geometry, layer._embeddings, config
        )
        del layer._embeddings
        untied = geometry.reversible and not layer.tie_weights
        if untied:
            reverse_weight_quantizer = (
                QuantizationConfig.weight_quantizer_or_default(
                    config,
                    AbsMaxQuantizer(axis=0),
                )
            )
            reverse_embeddings_value, reverse_embeddings_scale = (
                reverse_weight_quantizer(
                    layer.reverse_embeddings, to_numpy=True
                )
            )
            reverse_embeddings_scale = ops.squeeze(
                reverse_embeddings_scale, axis=0
            )
            del layer.reverse_embeddings
        layer.quantized_build(embeddings_shape, "int8", config)
        layer._embeddings.assign(embeddings_value)
        layer.embeddings_scale.assign(embeddings_scale)
        if untied:
            layer.reverse_embeddings.assign(reverse_embeddings_value)
            layer.reverse_embeddings_scale.assign(reverse_embeddings_scale)
