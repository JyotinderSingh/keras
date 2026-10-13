"""int4 handlers for the lookup family (`Embedding`, `ReversibleEmbedding`)."""

from keras.src.quantizers.modes.int4.block_size import int4_scheme
from keras.src.quantizers.modes.lookup import LookupHandlers
from keras.src.quantizers.quantized_weight import Int4Pairs


class Int4LookupHandlers(LookupHandlers):
    """What int4 gives `LookupHandlers`: its scheme and layout.

    The table is packed two codes per byte along `output_dim`. The scale
    runs per row (per-channel) or per row and group of columns (grouped,
    with a zero point and a group index). The reverse projection is
    weight-only unless a config adds an activation quantizer.
    """

    def _get_lookup_scheme(self, layer, config):
        return int4_scheme(self.resolve_block_size(layer, config))

    def _get_lookup_layout(self, axis, length):
        return Int4Pairs(axis=axis, orig_len=length)
