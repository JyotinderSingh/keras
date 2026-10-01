"""int4 handlers for the lookup family (`Embedding`, `ReversibleEmbedding`)."""

from keras.src import ops
from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.int4.block_size import is_per_channel
from keras.src.quantizers.modes.lookup import LookupHandlers
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)


class Int4LookupHandlers(LookupHandlers):
    """What int4 gives `LookupHandlers`: its scheme, layout and encoding.

    The table is packed two codes per byte along `output_dim`. The scale
    runs per row (per-channel) or per row and group of columns (grouped,
    with a zero point and a group index). The reverse projection is
    weight-only unless a config adds an activation quantizer.
    """

    def _lookup_scheme(self, layer, config):
        return int4_scheme(self.resolve_block_size(layer, config))

    def _lookup_layout(self, axis, length):
        return Int4Pairs(axis=axis, orig_len=length)

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
            embeddings_scale = ops.transpose(scale_t)
            embeddings_zero = ops.transpose(zero_t)

        packed_embeddings_value, _, _ = pack_int4(embeddings_value, axis=-1)
        return packed_embeddings_value, embeddings_scale, embeddings_zero
