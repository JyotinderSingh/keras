"""int4 handlers for the lookup family (`Embedding`, `ReversibleEmbedding`)."""

import math

from keras.src import ops
from keras.src.quantizers.modes.common import add_lookup_lora_delta
from keras.src.quantizers.modes.common import apply_logit_soft_cap
from keras.src.quantizers.modes.common import cast_lookup_inputs
from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.int4.block_size import is_grouped
from keras.src.quantizers.modes.int4.block_size import is_per_channel
from keras.src.quantizers.qtensor import Int4Pairs
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)
from keras.src.quantizers.quantizers import dequantize_grouped
from keras.src.quantizers.quantizers import divisor_scale
from keras.src.quantizers.quantizers import pack_int4
from keras.src.quantizers.quantizers import unpack_int4


class Int4LookupHandlers:
    """`_build_lookup` / `_call_lookup` / `_call_reversible_lookup` /
    `_encode_lookup` / `_qtensor_lookup` / `_quantize_lookup`."""

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

        if is_per_channel(block_size):
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
        if is_grouped(block_size):
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

                if is_per_channel(block_size):
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
                if is_grouped(block_size):
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

        if is_per_channel(block_size):
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
                if is_per_channel(block_size):
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

            if is_per_channel(block_size):
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
        # `Int4Strategy.resolve_block_size` is the single source of truth for
        # the group size, shared with the build path and the dtype-policy
        # naming. A bare `quantize("int4")` resolves to the canonical
        # `Int4QuantizationConfig()` (grouped, block_size=128); `None`/`-1`
        # selects per-channel.
        block_size = self.resolve_block_size(layer, config)

        if is_per_channel(block_size):
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
        grouped = is_grouped(layer._int4_block_size)
        return QTensor(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            zero_point=layer.embeddings_zero if grouped else None,
            g_idx=layer.g_idx if grouped else None,
            layout=Int4Pairs(axis=-1, orig_len=layer._orig_output_dim),
            scheme=int4_scheme(
                layer._int4_block_size, channel_axis=0, group_axis=-1
            ),
            logical_shape=(layer.input_dim, layer.output_dim),
            compute_dtype=layer.compute_dtype,
        )

    def _quantize_lookup(self, layer, geometry, config):
        embeddings_shape = (layer.input_dim, layer.output_dim)
        block_size = self.resolve_block_size(layer, config)
        use_grouped = is_grouped(block_size)
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
