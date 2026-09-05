import math

from keras.src import backend
from keras.src import ops
from keras.src.dtype_policies.dtype_policy import Int4DTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.modes.common import GeometryDispatchMode
from keras.src.quantizers.modes.common import add_einsum_lora_delta
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import add_matmul_lora_delta
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.qtensor import Int4Pairs
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import WeightScheme
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import abs_max_quantize
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)
from keras.src.quantizers.quantizers import dequantize_grouped
from keras.src.quantizers.quantizers import divisor_scale
from keras.src.quantizers.quantizers import pack_int4
from keras.src.quantizers.quantizers import unpack_int4


def _is_per_channel(block_size):
    """Whether `block_size` selects per-channel (ungrouped) quantization.

    `block_size` is validated to be `None`, `-1`, or a positive integer by
    both `Int4QuantizationConfig` and the policy-string codec, so `None`
    and `-1` are the two spellings of per-channel.
    """
    return block_size is None or block_size == -1


def _is_grouped(block_size):
    """Whether `block_size` selects sub-channel (grouped) quantization."""
    return not _is_per_channel(block_size)


def _flatten_rows_columns(kernel_shape, reduced_axes):
    """Flattens an N-D kernel shape to 2D `(rows, columns)`.

    Rows are the product of the contracted (reduced) axes, columns the
    product of the rest.
    """
    rows = 1
    columns = 1
    for i, dim in enumerate(kernel_shape):
        if i in reduced_axes:
            rows *= dim
        else:
            columns *= dim
    return rows, columns


def _int4_scheme(block_size, channel_axis, group_axis):
    """The int4 scheme for a block size: per-channel or grouped."""
    if _is_per_channel(block_size):
        # Symmetric codes with a per-channel divisor scale.
        return WeightScheme(
            bits=4,
            code_range=(-8, 7),
            channel_axis=channel_axis,
        )
    # Asymmetric codes: `(code - zero_point) / scale` per group.
    return WeightScheme(
        bits=4,
        code_range=(-8, 7),
        zero_point_dtype="int8",
        group_size=block_size,
        group_axis=group_axis,
    )


class Int4Mode(GeometryDispatchMode):
    """W4A16 weight-only quantization (packed int4 weights)."""

    name = "int4"
    config_cls = Int4QuantizationConfig

    def resolve_block_size(self, layer, config):
        """Determine the block size for int4 quantization.

        The block size can be specified either through the `config` argument
        or through the `dtype_policy` if it is of type `Int4DTypePolicy`.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the block size should be specified in the
        `dtype_policy`.

        Args:
            layer: The layer being quantized.
            config: An optional configuration object that may contain the
                `block_size` attribute.
        Returns:
            int or None. The determined block size for int4 quantization.
            Returns `None` or `-1` for per-channel quantization.
        """
        if isinstance(config, Int4QuantizationConfig):
            return config.block_size
        elif isinstance(layer.dtype_policy, Int4DTypePolicy):
            block_size = layer.dtype_policy.block_size
            # Convert -1 to None for consistency
            return None if block_size == -1 else block_size
        elif isinstance(layer.dtype_policy, DTypePolicyMap):
            policy = layer.dtype_policy[layer.path]
            if isinstance(policy, Int4DTypePolicy):
                block_size = policy.block_size
                return None if block_size == -1 else block_size
            # Fall back to None for legacy QuantizedDTypePolicy
            return None
        else:
            # For backwards compatibility with models that don't have
            # Int4DTypePolicy (legacy per-channel mode)
            return None

    def policy_from_string(self, mode_str, source_name):
        # Legacy bare "int4" policies carry no block size and stay generic
        # (they resolve to per-channel quantization on reload).
        if "/" in mode_str:
            return Int4DTypePolicy(mode_str, source_name)
        else:
            return QuantizedDTypePolicy(mode_str, source_name)

    def config_from_policy(self, policy):
        if isinstance(policy, Int4DTypePolicy):
            return Int4QuantizationConfig(block_size=policy.block_size)
        return Int4QuantizationConfig()

    def policy_suffix(self, layer, config):
        # Include block_size in policy name for sub-channel quantization.
        block_size = self.resolve_block_size(layer, config)
        # Use -1 for per-channel, otherwise use block_size
        block_size_value = -1 if block_size is None else block_size
        return f"int4/{block_size_value}"

    # --- Matmul projection (Dense) ----------------------------------------

    def _build_projection(self, layer, geometry, kernel_shape, config):
        """Build variables for int4 quantization.

        The kernel is packed along the last axis,
        resulting in shape `(input_dim, ceil(units/2))`.

        Args:
            layer: The layer being built.
            kernel_shape: The original float32 kernel shape
                `(input_dim, units)`.
            config: Optional quantization config specifying block_size.
        """
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        input_dim, output_dim = kernel_shape

        # kernel is packed along last axis (output dimension)
        # Stored shape: [input_dim, ceil(output_dim/2)]
        packed_cols = (output_dim + 1) // 2

        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(input_dim, packed_cols),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )

        block_size = self.resolve_block_size(layer, config)
        layer._int4_block_size = block_size

        if _is_per_channel(block_size):
            # Per-channel: one scale per output unit
            scale_shape = (layer.units,)
        else:
            # Sub-channel: [n_groups, out_features]
            n_groups = math.ceil(input_dim / block_size)
            scale_shape = (n_groups, layer.units)

        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )

        # Sub-channel quantization uses asymmetric quantization
        if _is_grouped(block_size):

            def idx_initializer(shape, dtype):
                return ops.floor_divide(
                    ops.arange(input_dim, dtype=dtype), block_size
                )

            layer.kernel_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            # `g_idx` is stored as `float32` because TF has no GPU kernel for
            # int32 resource variables (would pin the variable to CPU and
            # break jit_compile on GPU); consumers cast to int32 on-device.
            layer.g_idx = layer.add_weight(
                name="g_idx",
                shape=(input_dim,),
                initializer=idx_initializer,
                dtype="float32",
                trainable=False,
            )

        # Record dimensions for unpacking and reshaping at runtime.
        layer._orig_input_dim = input_dim
        layer._orig_output_dim = output_dim

    def _call_projection(self, layer, inputs, training=None):
        """Forward pass for an int4 quantized matmul projection.

        Uses custom gradients to handle quantized weights since autodiff
        cannot differentiate through int4 operations.
        """
        block_size = getattr(layer, "_int4_block_size", None)

        if _is_per_channel(block_size):
            # Per-channel: symmetric quantization (no zero point needed)
            @ops.custom_gradient
            def matmul_per_channel_with_inputs_gradient(
                inputs, kernel, kernel_scale
            ):
                """Per-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [in, ceil(out/2)], unpack along last axis
                unpacked_kernel = unpack_int4(
                    kernel, layer._orig_output_dim, axis=-1
                )

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    # Per-channel: unpacked is [in, out]
                    float_kernel = ops.divide(
                        ops.cast(unpacked_kernel, dtype=layer.compute_dtype),
                        kernel_scale,
                    )
                    inputs_grad = ops.matmul(
                        upstream, ops.transpose(float_kernel)
                    )
                    return (inputs_grad, None, None)

                # Forward pass: per-channel dequantization
                output_scale = kernel_scale
                if layer.inputs_quantizer:
                    inputs, inputs_scale = layer.inputs_quantizer(
                        inputs, axis=-1
                    )
                    output_scale = ops.multiply(output_scale, inputs_scale)

                x = ops.matmul(inputs, unpacked_kernel)
                x = ops.cast(x, layer.compute_dtype)
                x = ops.divide(x, output_scale)
                return x, grad_fn

            x = matmul_per_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(layer._kernel),
                ops.convert_to_tensor(layer.kernel_scale),
            )
        else:
            # Sub-channel: asymmetric quantization (with zero point)
            @ops.custom_gradient
            def matmul_sub_channel_with_inputs_gradient(
                inputs, kernel, kernel_scale, kernel_zero, g_idx
            ):
                """Sub-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [in, ceil(out/2)], unpack along last axis
                unpacked_kernel = unpack_int4(
                    kernel, layer._orig_output_dim, axis=-1
                )

                def _dequantize_kernel():
                    # Scale/zero are [n_groups, out]; g_idx expands them back
                    # over the input axis.
                    float_kernel = dequantize_grouped(
                        unpacked_kernel,
                        kernel_scale,
                        kernel_zero,
                        g_idx,
                        group_axis=0,
                    )
                    return ops.cast(float_kernel, layer.compute_dtype)

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = _dequantize_kernel()
                    inputs_grad = ops.matmul(
                        upstream, ops.transpose(float_kernel)
                    )
                    return (inputs_grad, None, None, None, None)

                x = ops.matmul(inputs, _dequantize_kernel())
                return x, grad_fn

            x = matmul_sub_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(layer._kernel),
                ops.convert_to_tensor(layer.kernel_scale),
                ops.convert_to_tensor(layer.kernel_zero),
                ops.convert_to_tensor(layer.g_idx),
            )

        x = add_matmul_lora_delta(layer, inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_projection(self, layer, geometry, weight, config):
        # Resolve the group size from the (already-resolved) config or the
        # layer's dtype policy. `Int4Mode.resolve_block_size` is the single
        # source of truth shared with the build path and the dtype-policy
        # naming, so the quantized values, the built variables, and the
        # saved policy string can never disagree. A bare `quantize("int4")`
        # reaches here with the canonical `Int4QuantizationConfig()`
        # (grouped, block_size=128); a `block_size` of `None` or `-1`
        # selects the per-channel escape hatch.
        block_size = self.resolve_block_size(layer, config)

        if _is_per_channel(block_size):
            # Per-channel quantization
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=0, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            kernel_value_int4, kernel_scale = weight_quantizer(
                weight, to_numpy=True
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            # Sub-channel quantization with asymmetric zero point
            # Returns kernel [in, out], scale [n_groups, out], zero
            # [n_groups, out]
            kernel_value_int4, kernel_scale, kernel_zero = (
                abs_max_quantize_grouped_with_zero_point(
                    weight, block_size=block_size, to_numpy=True
                )
            )
            kernel_scale = divisor_scale(kernel_scale, layer.variable_dtype)

        # Pack two int4 values per int8 byte along last axis
        # Stored as [in, ceil(out/2)]
        packed_kernel_value, _, _ = pack_int4(kernel_value_int4, axis=-1)
        return packed_kernel_value, kernel_scale, kernel_zero

    def _qtensor_projection(self, layer, geometry):
        grouped = _is_grouped(layer._int4_block_size)
        return QTensor(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            zero_point=layer.kernel_zero if grouped else None,
            g_idx=layer.g_idx if grouped else None,
            layout=Int4Pairs(axis=-1, orig_len=layer._orig_output_dim),
            scheme=_int4_scheme(
                layer._int4_block_size, channel_axis=-1, group_axis=0
            ),
            logical_shape=(layer._orig_input_dim, layer._orig_output_dim),
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, kernel_zero = self._encode_projection(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, "int4", config)

        # Assign values to the newly created variables.
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)
        if kernel_zero is not None:
            layer.kernel_zero.assign(kernel_zero)

    # --- Einsum projection (EinsumDense) ----------------------------------

    def _build_einsum(self, layer, geometry, kernel_shape, config):
        """Build variables for int4 quantization of an einsum kernel.

        The kernel is flattened to 2D [rows, columns]
        and packed along last axis to [rows, ceil(columns/2)].

        Args:
            layer: The layer being built.
            kernel_shape: Original kernel shape (may be N-dimensional).
            config: Optional quantization config specifying block_size.
        """
        layer._set_quantization_info()

        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        layer.quantization_axis = tuple(layer._input_reduced_axes)
        layer.original_kernel_shape = kernel_shape

        # Flatten kernel to 2D: rows = reduced dims, columns = non-reduced dims
        rows, columns = _flatten_rows_columns(
            kernel_shape, layer._kernel_reduced_axes
        )

        block_size = self.resolve_block_size(layer, config)
        use_grouped = _is_grouped(block_size)
        layer._int4_block_size = block_size if use_grouped else None

        # Kernel packed along last axis (columns)
        # Stored shape: [rows, ceil(columns/2)]
        packed_cols = (columns + 1) // 2
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(rows, packed_cols),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )

        if use_grouped:
            # Sub-channel: [n_groups, columns]
            n_groups = math.ceil(rows / block_size)
            scale_shape = (n_groups, columns)
        else:
            scale_shape = (columns,)

        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )

        # Sub-channel quantization uses asymmetric quantization with zero point
        if use_grouped:
            layer.kernel_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            # `g_idx` is stored as `float32` because TF has no GPU kernel for
            # int32 resource variables (would pin the variable to CPU and
            # break jit_compile on GPU); consumers cast to int32 on-device.
            layer.g_idx = layer.add_weight(
                name="g_idx",
                shape=(rows,),
                initializer="zeros",
                dtype="float32",
                trainable=False,
            )
            layer.g_idx.assign(
                ops.floor_divide(ops.arange(rows, dtype="float32"), block_size)
            )

    @staticmethod
    def _einsum_columns(layer):
        """Unpacked column count of a built int4 einsum kernel.

        `_build_einsum` allocates the scale as `(columns,)` per-channel and
        `(n_groups, columns)` grouped, so the last axis of the stored scale
        is the unpadded length of the packed kernel axis.
        """
        return int(layer.kernel_scale.shape[-1])

    def _call_einsum(self, layer, inputs, training=None):
        """Forward pass for int4 quantized EinsumDense.

        Uses custom gradients to handle quantized weights since autodiff
        cannot differentiate through int4 operations.
        """
        block_size = getattr(layer, "_int4_block_size", None)
        columns = self._einsum_columns(layer)

        if _is_per_channel(block_size):

            @ops.custom_gradient
            def einsum_per_channel_with_inputs_gradient(
                inputs, packed_kernel, kernel_scale
            ):
                """Per-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [rows, ceil(columns/2)],
                # unpack along last axis
                unpacked_kernel = unpack_int4(
                    packed_kernel,
                    columns,
                    axis=-1,
                    dtype="int8",
                )

                def _dequantize_kernel(unpacked, scale):
                    # kernel is [rows, columns], scale is [columns]
                    float_kernel = ops.divide(
                        ops.cast(unpacked, dtype=layer.compute_dtype),
                        scale,
                    )
                    return ops.reshape(
                        float_kernel, layer.original_kernel_shape
                    )

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    inputs_grad = ops.einsum(
                        layer._custom_gradient_equation, upstream, float_kernel
                    )
                    return (inputs_grad, None, None)

                if layer.inputs_quantizer:
                    # Per-channel with input quantization
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    inputs_q, inputs_scale = layer.inputs_quantizer(
                        inputs, axis=layer.quantization_axis
                    )
                    inputs_scale = layer._adjust_scale_for_quant(
                        inputs_scale, "input"
                    )
                    # Cast inputs to float for einsum. This is a workaround
                    # for PyTorch's einsum which doesn't support
                    # mixed-precision inputs (int8 input, float kernel).
                    if backend.backend() == "torch":
                        x = ops.einsum(
                            layer.equation,
                            ops.cast(inputs_q, layer.compute_dtype),
                            float_kernel,
                        )
                        x = ops.divide(x, inputs_scale)
                    else:
                        x = ops.einsum(layer.equation, inputs_q, float_kernel)
                        x = ops.cast(x, layer.compute_dtype)
                        x = ops.divide(x, inputs_scale)
                else:
                    # Weight-only per-channel quantization
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    x = ops.einsum(layer.equation, inputs, float_kernel)
                return x, grad_fn

            x = einsum_per_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(layer._kernel),
                ops.convert_to_tensor(layer.kernel_scale),
            )
        else:

            @ops.custom_gradient
            def einsum_sub_channel_with_inputs_gradient(
                inputs, packed_kernel, kernel_scale, kernel_zero, g_idx
            ):
                """Sub-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [rows, ceil(columns/2)],
                # unpack along last axis
                unpacked_kernel = unpack_int4(
                    packed_kernel,
                    columns,
                    axis=-1,
                    dtype="int8",
                )

                def _dequantize_kernel(unpacked, scale, zero, g_idx_t):
                    # Dequantize with group_axis=0 since
                    # scale is [n_groups, columns]
                    float_kernel = dequantize_grouped(
                        unpacked, scale, zero, g_idx_t, group_axis=0
                    )
                    float_kernel = ops.cast(float_kernel, layer.compute_dtype)
                    return ops.reshape(
                        float_kernel, layer.original_kernel_shape
                    )

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale, kernel_zero, g_idx
                    )
                    inputs_grad = ops.einsum(
                        layer._custom_gradient_equation, upstream, float_kernel
                    )
                    return (inputs_grad, None, None, None, None)

                float_kernel = _dequantize_kernel(
                    unpacked_kernel, kernel_scale, kernel_zero, g_idx
                )
                x = ops.einsum(layer.equation, inputs, float_kernel)
                return x, grad_fn

            x = einsum_sub_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(layer._kernel),
                ops.convert_to_tensor(layer.kernel_scale),
                ops.convert_to_tensor(layer.kernel_zero),
                ops.convert_to_tensor(layer.g_idx),
            )

        x = add_einsum_lora_delta(layer, inputs, x)
        return apply_bias_activation(layer, x)

    def _encode_einsum(self, layer, geometry, weight, config):
        layer._set_quantization_info()
        # `Int4Mode.resolve_block_size` is the single source of truth for the
        # group size, shared with the build path and the dtype-policy naming.
        # A bare `quantize("int4")` resolves to the canonical
        # `Int4QuantizationConfig()` (grouped, block_size=128); `None`/`-1`
        # selects per-channel.
        block_size = self.resolve_block_size(layer, config)

        # Flatten kernel to 2D: rows = reduced dims, columns = non-reduced
        rows, columns = _flatten_rows_columns(
            weight.shape, layer._kernel_reduced_axes
        )
        flat_kernel = ops.reshape(weight, (rows, columns))

        if _is_per_channel(block_size):
            # Per-channel quantization
            kernel_value_int4, kernel_scale = abs_max_quantize(
                flat_kernel,
                axis=0,
                value_range=(-8, 7),
                dtype="int8",
                to_numpy=True,
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            # Sub-channel quantization with asymmetric zero point
            # Returns kernel [rows, columns], scale [n_groups, columns]
            kernel_value_int4, kernel_scale, kernel_zero = (
                abs_max_quantize_grouped_with_zero_point(
                    flat_kernel, block_size=block_size, to_numpy=True
                )
            )
            kernel_scale = divisor_scale(kernel_scale, layer.variable_dtype)

        # Pack two int4 values per int8 byte along last axis
        # Stored as [rows, ceil(columns/2)]
        packed_kernel_value, _, _ = pack_int4(kernel_value_int4, axis=-1)
        return packed_kernel_value, kernel_scale, kernel_zero

    def _qtensor_einsum(self, layer, geometry):
        grouped = _is_grouped(layer._int4_block_size)
        return QTensor(
            codes=layer._kernel,
            scale=layer.kernel_scale,
            zero_point=layer.kernel_zero if grouped else None,
            g_idx=layer.g_idx if grouped else None,
            layout=Int4Pairs(axis=-1, orig_len=self._einsum_columns(layer)),
            scheme=_int4_scheme(
                layer._int4_block_size, channel_axis=-1, group_axis=0
            ),
            logical_shape=layer.original_kernel_shape,
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_einsum(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        kernel_value, kernel_scale, kernel_zero = self._encode_einsum(
            layer, geometry, layer._kernel, config
        )
        del layer._kernel
        layer.quantized_build(kernel_shape, "int4", config)

        # Assign values to the newly created variables.
        layer._kernel.assign(kernel_value)
        layer.kernel_scale.assign(kernel_scale)
        if kernel_zero is not None:
            layer.kernel_zero.assign(kernel_zero)

    # --- Embeddings lookup (Embedding, ReversibleEmbedding) ---------------

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        """Build variables for int4 quantization of an embeddings table.

        Args:
            layer: The layer being built.
            geometry: The layer's `LookupGeometry`.
            embeddings_shape: Original shape `(input_dim, output_dim)`.
            config: Optional quantization config specifying block_size.
        """
        input_dim, output_dim = embeddings_shape
        packed_rows = (output_dim + 1) // 2

        # Embeddings are stored packed: each int8 byte contains two
        # int4 values.
        layer._embeddings = layer.add_weight(
            name="embeddings",
            shape=(input_dim, packed_rows),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )

        block_size = self.resolve_block_size(layer, config)
        layer._int4_block_size = block_size

        if _is_per_channel(block_size):
            scale_shape = (layer.input_dim,)
        else:
            n_groups = math.ceil(output_dim / block_size)
            scale_shape = (layer.input_dim, n_groups)

        layer.embeddings_scale = layer.add_weight(
            name="embeddings_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )

        # Sub-channel quantization uses asymmetric quantization with
        # zero point
        if _is_grouped(block_size):
            layer.embeddings_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            # `g_idx` is stored as `float32` because TF has no GPU kernel for
            # int32 resource variables (would pin the variable to CPU and
            # break jit_compile on GPU); consumers cast to int32 on-device.
            layer.g_idx = layer.add_weight(
                name="g_idx",
                shape=(output_dim,),
                initializer="zeros",
                dtype="float32",
                trainable=False,
            )
            layer.g_idx.assign(
                ops.floor_divide(
                    ops.arange(output_dim, dtype="float32"), block_size
                )
            )

        layer._orig_output_dim = output_dim

        if geometry.reversible:
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(
                    config, AbsMaxQuantizer(axis=-1)
                )
            )
            if not layer.tie_weights:
                packed_reverse_rows = (
                    layer.output_dim + 1
                ) // 2  # ceil, odd dims
                layer.reverse_embeddings = layer.add_weight(
                    name="reverse_embeddings",
                    shape=(packed_reverse_rows, layer.input_dim),
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )

                if _is_per_channel(block_size):
                    # Per-channel: one scale per output unit (input_dim)
                    reverse_scale_shape = (layer.input_dim,)
                else:
                    # Grouped: scale per group along output_dim (axis=0)
                    reverse_n_groups = math.ceil(layer.output_dim / block_size)
                    reverse_scale_shape = (reverse_n_groups, layer.input_dim)

                layer.reverse_embeddings_scale = layer.add_weight(
                    name="reverse_embeddings_scale",
                    shape=reverse_scale_shape,
                    initializer="ones",
                    trainable=False,
                )

                # Zero point for asymmetric grouped quantization
                if _is_grouped(block_size):
                    layer.reverse_embeddings_zero = layer.add_weight(
                        name="reverse_zero_point",
                        shape=reverse_scale_shape,
                        initializer="zeros",
                        trainable=False,
                    )

    def _call_lookup(self, layer, inputs, training=None):
        """Forward pass for an int4 quantized embeddings lookup."""
        inputs = cast_lookup_inputs(inputs)

        unpacked_embeddings = unpack_int4(
            layer._embeddings, layer._orig_output_dim, axis=-1
        )
        outputs = ops.take(unpacked_embeddings, inputs, axis=0)

        block_size = getattr(layer, "_int4_block_size", None)

        if _is_per_channel(block_size):
            embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
            outputs = ops.divide(
                ops.cast(outputs, dtype=layer.compute_dtype),
                ops.expand_dims(embeddings_scale, axis=-1),
            )
        else:
            # Sub-channel: look up scale/zero for each input token,
            # then dequantize using g_idx to expand groups
            embeddings_scale = ops.take(layer.embeddings_scale, inputs, axis=0)
            embeddings_zero = ops.take(layer.embeddings_zero, inputs, axis=0)

            # Scale/zero are [batch..., n_groups], g_idx is [output_dim]
            outputs = dequantize_grouped(
                ops.cast(outputs, dtype=layer.compute_dtype),
                embeddings_scale,
                embeddings_zero,
                layer.g_idx,
                group_axis=-1,
            )

        return add_lookup_lora_delta(layer, inputs, outputs)

    def _call_reversible_lookup(self, layer, inputs, reverse=False):
        if not reverse:
            return self._call_lookup(layer, inputs)
        else:
            block_size = getattr(layer, "_int4_block_size", None)

            if layer.tie_weights:
                embeddings = ops.transpose(layer._embeddings)
                scale = layer.embeddings_scale
                # For tied weights, scale shape is (input_dim,) or
                # (input_dim, n_groups). For per-channel, transpose scale.
                if _is_per_channel(block_size):
                    scale = ops.transpose(scale)
            else:
                embeddings = layer.reverse_embeddings
                scale = layer.reverse_embeddings_scale

            unpacked_embeddings = unpack_int4(
                embeddings, layer.output_dim, axis=0
            )

            if layer.inputs_quantizer:
                inputs, inputs_scale = layer.inputs_quantizer(inputs)
            else:
                inputs_scale = ops.ones((1,), dtype=layer.compute_dtype)

            if _is_per_channel(block_size):
                # Per-channel: do matmul then dequantize
                logits = ops.matmul(inputs, unpacked_embeddings)
                logits = ops.cast(logits, layer.compute_dtype)
                logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
            elif layer.tie_weights:
                # Sub-channel with asymmetric quantization (tied weights)
                # Must dequantize embeddings before matmul for correctness
                # unpacked_embeddings shape: (output_dim, input_dim)
                # scale shape: (input_dim, n_groups)
                # embeddings_zero shape: (input_dim, n_groups)
                # g_idx shape: (output_dim,)

                # Transpose scale/zero for dequantization:
                # [input_dim, n_groups] -> [n_groups, input_dim]
                scale_t = ops.transpose(scale)
                zero_t = ops.transpose(layer.embeddings_zero)

                float_embeddings = dequantize_grouped(
                    ops.cast(unpacked_embeddings, layer.compute_dtype),
                    scale_t,
                    zero_t,
                    layer.g_idx,
                    group_axis=0,
                )

                # inputs shape: (batch, output_dim)
                # float_embeddings shape: (output_dim, input_dim)
                logits = ops.matmul(inputs, float_embeddings)
                logits = ops.divide(logits, inputs_scale)
            else:
                # Untied weights with asymmetric grouped quantization
                # Must dequantize embeddings before matmul for correctness
                # unpacked_embeddings shape: (output_dim, input_dim)
                # scale shape: (n_groups, input_dim)
                # reverse_embeddings_zero shape: (n_groups, input_dim)
                # g_idx shape: (output_dim,) - reuse from forward pass

                float_embeddings = dequantize_grouped(
                    ops.cast(unpacked_embeddings, layer.compute_dtype),
                    scale,
                    layer.reverse_embeddings_zero,
                    layer.g_idx,
                    group_axis=0,
                )

                # inputs shape: (batch, output_dim)
                # float_embeddings shape: (output_dim, input_dim)
                logits = ops.matmul(inputs, float_embeddings)
                logits = ops.divide(logits, inputs_scale)

            return apply_logit_soft_cap(layer, logits)

    def _encode_lookup(self, layer, geometry, weight, config):
        # `Int4Mode.resolve_block_size` is the single source of truth for the
        # group size, shared with the build path and the dtype-policy naming.
        # A bare `quantize("int4")` resolves to the canonical
        # `Int4QuantizationConfig()` (grouped, block_size=128); `None`/`-1`
        # selects per-channel.
        block_size = self.resolve_block_size(layer, config)

        if _is_per_channel(block_size):
            # Per-channel quantization
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=-1, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                weight, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            embeddings_zero = None
        else:
            # Sub-channel quantization with asymmetric zero point
            # Transpose to put output_dim first for grouped quantization
            embeddings_t = ops.transpose(weight)

            embeddings_value_t, scale_t, zero_t = (
                abs_max_quantize_grouped_with_zero_point(
                    embeddings_t,
                    block_size=block_size,
                    value_range=(-8, 7),
                    dtype="int8",
                    to_numpy=True,
                )
            )
            # Transpose back to (input_dim, output_dim) layout
            embeddings_value = ops.transpose(embeddings_value_t)
            embeddings_scale = divisor_scale(
                ops.transpose(scale_t), layer.variable_dtype
            )
            embeddings_zero = ops.transpose(zero_t)

        packed_embeddings_value, _, _ = pack_int4(embeddings_value, axis=-1)
        return packed_embeddings_value, embeddings_scale, embeddings_zero

    def _qtensor_lookup(self, layer, geometry):
        grouped = _is_grouped(layer._int4_block_size)
        return QTensor(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            zero_point=layer.embeddings_zero if grouped else None,
            g_idx=layer.g_idx if grouped else None,
            layout=Int4Pairs(axis=-1, orig_len=layer._orig_output_dim),
            scheme=_int4_scheme(
                layer._int4_block_size, channel_axis=0, group_axis=-1
            ),
            logical_shape=(layer.input_dim, layer.output_dim),
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = (layer.input_dim, layer.output_dim)
        block_size = self.resolve_block_size(layer, config)
        use_grouped = _is_grouped(block_size)
        packed_embeddings_value, embeddings_scale, embeddings_zero = (
            self._encode_lookup(layer, geometry, layer._embeddings, config)
        )
        del layer._embeddings

        # Quantize reverse embeddings if not tied
        untied = geometry.reversible and not layer.tie_weights
        if untied:
            if not use_grouped:
                reverse_weight_quantizer = (
                    QuantizationConfig.weight_quantizer_or_default(
                        config,
                        AbsMaxQuantizer(
                            axis=0, value_range=(-8, 7), output_dtype="int8"
                        ),
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
            else:
                reverse_value, reverse_scale, reverse_zero = (
                    abs_max_quantize_grouped_with_zero_point(
                        layer.reverse_embeddings,
                        block_size=block_size,
                        value_range=(-8, 7),
                        dtype="int8",
                        to_numpy=True,
                    )
                )
                reverse_embeddings_value = reverse_value
                reverse_embeddings_scale = divisor_scale(
                    reverse_scale, layer.variable_dtype
                )
                reverse_embeddings_zero = reverse_zero

            packed_reverse_embeddings_value, _, _ = pack_int4(
                reverse_embeddings_value, axis=0
            )
            del layer.reverse_embeddings

        layer.quantized_build(embeddings_shape, "int4", config)
        layer._embeddings.assign(packed_embeddings_value)
        layer.embeddings_scale.assign(embeddings_scale)
        if use_grouped:
            layer.embeddings_zero.assign(embeddings_zero)
        if untied:
            layer.reverse_embeddings.assign(packed_reverse_embeddings_value)
            layer.reverse_embeddings_scale.assign(reverse_embeddings_scale)
            if use_grouped:
                layer.reverse_embeddings_zero.assign(reverse_embeddings_zero)
