"""What an int4 block size implies: per-channel or grouped, and the scheme."""

from keras.src.quantizers.qtensor import WeightScheme


def is_per_channel(block_size):
    """Whether `block_size` selects per-channel (ungrouped) quantization.

    `block_size` is validated to be `None`, `-1`, or a positive integer by
    both `Int4QuantizationConfig` and the policy-string codec, so `None`
    and `-1` are the two spellings of per-channel.
    """
    return block_size is None or block_size == -1


def is_grouped(block_size):
    """Whether `block_size` selects sub-channel (grouped) quantization."""
    return not is_per_channel(block_size)


def int4_scheme(block_size, channel_axis, group_axis):
    """The int4 scheme for a block size: per-channel or grouped."""
    if is_per_channel(block_size):
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
