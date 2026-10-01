"""int4 handlers for the projection family (`Dense`, `EinsumDense`)."""

import math

from keras.src import ops
from keras.src.quantizers.modes.common import add_group_index
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.int4.block_size import is_per_channel
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)


class Int4ProjectionHandlers:
    """The int4 build, forward, encode and view of a projection kernel.

    The kernel is stored as its 2-D `[rows, columns]` matrix
    (`KernelAxes.matrix` with `batch_in="columns"`): the contracted axes
    are the rows, and every other axis, in the kernel's order, makes up
    the columns. The codes are packed two per byte along the columns, and
    the scale runs per column (per-channel) or per group of rows (grouped,
    with a zero point and a group index). The forward pass dequantizes
    through the `QuantizedWeight` view and contracts in float.
    """

    def _build_projection(self, layer, geometry, kernel_shape, config):
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        _, rows, columns = geometry.kernel_axes.matrix(
            kernel_shape, batch_in="columns"
        )
        block_size = self.resolve_block_size(layer, config)

        # Codes packed two per byte along the columns.
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(rows, Int4Pairs.packed_length(columns)),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        if is_per_channel(block_size):
            scale_shape = (columns,)
        else:
            scale_shape = (math.ceil(rows / block_size), columns)
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )
        layer.kernel_zero = None
        layer.g_idx = None
        if not is_per_channel(block_size):
            # Grouped quantization is asymmetric: a zero point per group and
            # the row-to-group index.
            layer.kernel_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            layer.g_idx = add_group_index(
                layer,
                rows,
                lambda shape, dtype: ops.floor_divide(
                    ops.arange(rows, dtype=dtype), block_size
                ),
            )

    def _view(self, layer, geometry, codes, scale, zero_point, g_idx):
        """The view over the layer's variables or over traced tensors."""
        permutation, _, columns = geometry.kernel_axes.matrix(
            geometry.weight_shape, batch_in="columns"
        )
        return QuantizedWeight(
            codes=codes,
            scale=scale,
            layout=Int4Pairs(axis=-1, orig_len=columns),
            scheme=int4_scheme(
                self.resolve_block_size(layer, layer.quantization_config)
            ),
            shape=geometry.weight_shape,
            # A grouped scale runs per group of rows, a per-channel scale
            # per column.
            axis=None if g_idx is None else 0,
            scale_axes=(1,) if g_idx is None else None,
            permutation=permutation,
            zero_point=zero_point,
            g_idx=g_idx,
        )

    def _quantized_weight_projection(self, layer, geometry):
        return self._view(
            layer,
            geometry,
            layer._kernel,
            layer.kernel_scale,
            layer.kernel_zero,
            layer.g_idx,
        )

    def _call_projection(self, layer, geometry, inputs, training=None):
        grouped = layer.g_idx is not None

        @ops.custom_gradient
        def contract_with_inputs_gradient(
            inputs, kernel, kernel_scale, *group_params
        ):
            """Dequantizes the int4 kernel and contracts in float.

            `group_params` is `(kernel_zero, g_idx)` for a grouped scheme
            and empty for per-channel. Autodiff cannot differentiate through
            the packed kernel, so the gradient with respect to the inputs is
            taken through the dequantized kernel.
            """
            kernel_zero, g_idx = group_params if group_params else (None, None)

            def dequantize():
                return self._view(
                    layer, geometry, kernel, kernel_scale, kernel_zero, g_idx
                ).dequantize(layer.compute_dtype)

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                inputs_grad = geometry.contract_grad(upstream, dequantize())
                return (inputs_grad, None, None) + (None,) * len(group_params)

            float_kernel = dequantize()
            if layer.inputs_quantizer:
                inputs_q, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                x = geometry.contract(inputs_q, float_kernel)
                x = ops.cast(x, layer.compute_dtype)
                x = ops.divide(x, geometry.align_inputs_scale(inputs_scale))
            else:
                x = geometry.contract(inputs, float_kernel)
            return x, grad_fn

        params = [
            inputs,
            ops.convert_to_tensor(layer._kernel),
            # Read inside the autocast scope: on TensorFlow eager the gradient
            # runs after it, and the variable itself would then read float32.
            ops.convert_to_tensor(layer.kernel_scale.value),
        ]
        if grouped:
            params += [
                ops.convert_to_tensor(layer.kernel_zero),
                ops.convert_to_tensor(layer.g_idx),
            ]
        x = contract_with_inputs_gradient(*params)
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        # `Int4Strategy.resolve_block_size` is the single source of truth for
        # the group size, shared with the build path and the dtype-policy
        # naming, so the quantized values, the built variables, and the saved
        # policy string can never disagree. A bare `quantize("int4")` reaches
        # here with the canonical `Int4QuantizationConfig()` (grouped,
        # block_size=128); a `block_size` of `None` or `-1` selects the
        # per-channel escape hatch.
        block_size = self.resolve_block_size(layer, config)
        permutation, rows, columns = geometry.kernel_axes.matrix(
            weight.shape, batch_in="columns"
        )
        flat_kernel = ops.reshape(
            ops.transpose(weight, permutation), (rows, columns)
        )

        if is_per_channel(block_size):
            # Symmetric codes with one scale per column.
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=0, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            kernel_value_int4, kernel_scale = weight_quantizer(
                flat_kernel, to_numpy=True
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            # Asymmetric codes per group of rows: scale and zero point are
            # `[n_groups, columns]`.
            kernel_value_int4, kernel_scale, kernel_zero = (
                abs_max_quantize_grouped_with_zero_point(
                    flat_kernel, block_size=block_size, to_numpy=True
                )
            )

        # Pack two int4 values per int8 byte along the columns.
        packed_kernel_value, _, _ = pack_int4(kernel_value_int4, axis=-1)
        return packed_kernel_value, kernel_scale, kernel_zero

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, kernel_zero = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, self.name, config)
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)
        if kernel_zero is not None:
            layer.kernel_zero.assign(kernel_zero)
