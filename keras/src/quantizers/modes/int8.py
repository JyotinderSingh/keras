from keras.src import backend
from keras.src import ops
from keras.src.quantizers.modes.common import GeometryDispatchMode
from keras.src.quantizers.modes.common import add_einsum_lora_delta
from keras.src.quantizers.modes.common import add_matmul_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer


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

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config, AbsMaxQuantizer(axis=0)
        )
        kernel_value, kernel_scale = weight_quantizer(
            layer._kernel, to_numpy=True
        )
        kernel_scale = ops.squeeze(kernel_scale, axis=0)
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

    def _quantize_einsum(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        layer._set_quantization_info()
        # Quantize `layer._kernel` to int8 and compute corresponding scale
        weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
            config,
            AbsMaxQuantizer(axis=layer._kernel_reduced_axes),
        )
        kernel_value, kernel_scale = weight_quantizer(
            layer._kernel, to_numpy=True
        )
        kernel_scale = layer._adjust_scale_for_quant(kernel_scale, "kernel")
        del layer._kernel
        layer.quantized_build(kernel_shape, "int8", config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)
