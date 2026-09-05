"""Quantized weight views: `WeightScheme`, `PackLayout`, `QTensor`.

A quantization mode stores one weight as a handful of variables: integer
codes (often packed several to a byte), a scale, and for asymmetric
schemes a zero point and a group index. Reading a real-valued weight back
out of those variables used to be re-derived at every consumer, each with
its own ladder of mode branches. A `QTensor` is a read-only view that
gathers the variables of one weight together with two descriptions of how
to read them:

- `PackLayout` says how the codes are stored: how many fit in a byte, along
  which axis, and how to unpack them. It is a sum type, one subclass per
  storage format, because the formats differ in behavior (a base-3 trit
  pack cannot be described by a bitfield mask) and not only in data.
- `WeightScheme` says what the codes mean: their bit-width and range,
  whether there is a zero point and in which dtype, and how groups run.
  It is one flat record, because every consumer reads the fields as a
  unit.

The view has two surfaces, split by what they return. `unpack()` returns
the integer codes in the weight's logical shape and orientation, which is
what a layer's `kernel` or `embeddings` property exposes for a quantized
layer. `dequantize()` returns the real-valued weight, which is what the
LoRA-merged save path, the calibration forward pass, and any exporter
need. Neither surface caches: a view is built on demand by
`QuantizationMode.qtensor(layer)` and holds references to the layer's
variables, so it always reflects their current values.

Every mode stores the scale it divides by, the form the abs-max
quantizer produces (`qmax / amax`); the grouped and calibration
quantizers compute the multiplier form and store its reciprocal. So
`dequantize()` is one formula, `(code - zero_point) / scale`, with the
per-group values gathered through the group index for a grouped scheme.

`float8` has no view: it keeps the float kernel and stores dynamic-range
state, not a codebook, so `QuantizationMode.qtensor` returns `None` for
it, as it does for a calibration mode before its calibration pass.
"""

import dataclasses
import math

from keras.src import ops
from keras.src.quantizers.quantizers import dequantize_grouped
from keras.src.quantizers.quantizers import pack_int2
from keras.src.quantizers.quantizers import pack_int4
from keras.src.quantizers.quantizers import pack_ternary
from keras.src.quantizers.quantizers import unpack_int2
from keras.src.quantizers.quantizers import unpack_int4
from keras.src.quantizers.quantizers import unpack_ternary


@dataclasses.dataclass(frozen=True, kw_only=True)
class WeightScheme:
    """How the integer codes of one quantized weight map to real values.

    A flat, immutable record. Every field describes the stored variables
    as they are, so a scheme can be written down for any of the shipped
    modes without changing what they store.

    Args:
        bits: Bit-width of one code in the code domain (the storage
            density is the `PackLayout`'s business). Ternary codes take
            two bits each even though five of them pack into a byte.
        code_range: `(min, max)` of the codes, e.g. `(-127, 127)` for int8
            or `(0, 15)` for unsigned 4-bit calibration codes.
        zero_point_dtype: Dtype of the stored zero point, or `None` for a
            symmetric scheme with no zero point. The real value is
            `(code - zero_point) / scale`.
        channel_axis: For a per-channel divisor scale, the axis of the
            unpacked codes the scale runs along. `None` means the stored
            scale already broadcasts against the codes as it is, which
            covers per-tensor scalars and the pre-aligned N-D scale of an
            einsum kernel.
        group_size: Number of codes per group for a grouped scheme, or
            `None` for per-channel or per-tensor scaling.
        group_axis: For a grouped scheme, the axis of the unpacked codes
            along which the groups run. The scale and zero point hold one
            entry per group along that same axis, and the group index maps
            each position on it to its group.
    """

    bits: int
    code_range: tuple
    zero_point_dtype: str = None
    channel_axis: int = None
    group_size: int = None
    group_axis: int = None

    def __post_init__(self):
        if (self.group_size is None) != (self.group_axis is None):
            raise ValueError(
                "`group_size` and `group_axis` must be given together. "
                f"Received: group_size={self.group_size!r}, "
                f"group_axis={self.group_axis!r}"
            )
        if self.group_size is not None and self.channel_axis is not None:
            raise ValueError(
                "A grouped scheme has no per-channel axis. Received: "
                f"group_size={self.group_size!r}, "
                f"channel_axis={self.channel_axis!r}"
            )
        # Normalize the two numeric fields; the record is frozen, so go
        # through `object.__setattr__`.
        object.__setattr__(self, "bits", int(self.bits))
        object.__setattr__(
            self,
            "code_range",
            (int(self.code_range[0]), int(self.code_range[1])),
        )

    @property
    def signed(self):
        """Whether the codes are signed."""
        return self.code_range[0] < 0

    @property
    def symmetric(self):
        """Whether the scheme has no zero point."""
        return self.zero_point_dtype is None

    @property
    def grouped(self):
        """Whether the scale runs per group rather than per channel."""
        return self.group_size is not None


class PackLayout:
    """How the codes of one weight are laid out in their storage variable.

    Each subclass owns one storage format: the number of codes per byte,
    the axis the codes are packed along, and the exact pack and unpack
    ops. `values_per_byte` is what `Model.quantization_summary` uses to
    turn a stored byte count into a logical parameter count.
    """

    values_per_byte = 1

    def unpack(self, codes):
        """Returns the unpacked codes, one per element."""
        raise NotImplementedError

    def pack(self, codes):
        """Returns the packed codes for storage."""
        raise NotImplementedError

    def packed_length(self, length):
        """Stored length of an axis holding `length` codes."""
        return length

    def __repr__(self):
        return f"{type(self).__name__}()"


class NoPack(PackLayout):
    """One code per stored element."""

    def unpack(self, codes):
        return codes

    def pack(self, codes):
        return codes


class _AxisPack(PackLayout):
    """A layout that packs several codes per byte along one axis."""

    def __init__(self, axis, orig_len):
        self.axis = axis
        self.orig_len = orig_len

    def packed_length(self, length):
        return math.ceil(length / self.values_per_byte)

    def __repr__(self):
        return (
            f"{type(self).__name__}(axis={self.axis}, orig_len={self.orig_len})"
        )


class Int4Pairs(_AxisPack):
    """Two 4-bit codes per byte along `axis`, as `pack_int4` writes them."""

    values_per_byte = 2

    def __init__(self, axis, orig_len, dtype="int8"):
        super().__init__(axis, orig_len)
        self.dtype = dtype

    def unpack(self, codes):
        return unpack_int4(
            codes, self.orig_len, axis=self.axis, dtype=self.dtype
        )

    def pack(self, codes):
        packed, _, _ = pack_int4(codes, axis=self.axis, dtype=self.dtype)
        return packed


class Int2Quads(_AxisPack):
    """Four 2-bit codes per byte along `axis`, as `pack_int2` writes them."""

    values_per_byte = 4

    def __init__(self, axis, orig_len, dtype="int8"):
        super().__init__(axis, orig_len)
        self.dtype = dtype

    def unpack(self, codes):
        return unpack_int2(
            codes, self.orig_len, axis=self.axis, dtype=self.dtype
        )

    def pack(self, codes):
        packed, _, _ = pack_int2(codes, axis=self.axis, dtype=self.dtype)
        return packed


class TernaryTrits(_AxisPack):
    """Five ternary codes per byte along `axis`, in base 3.

    The unpack is arithmetic (`mod(floor_divide(byte, 3**k), 3) - 1`), not
    a mask and shift, which is why the layout is a subclass rather than a
    bit-width flag.
    """

    values_per_byte = 5

    def unpack(self, codes):
        return unpack_ternary(codes, self.orig_len, axis=self.axis)

    def pack(self, codes):
        packed, _, _ = pack_ternary(codes, axis=self.axis)
        return packed


class QTensor:
    """A read-only view over the stored variables of one quantized weight.

    Args:
        codes: The stored code variable, packed as `layout` describes.
        scale: The stored divisor scale: the real value is
            `(code - zero_point) / scale`.
        layout: The `PackLayout` of `codes`.
        scheme: The `WeightScheme` of the weight.
        logical_shape: Shape of the real-valued weight the codes stand for,
            in the weight's own orientation.
        zero_point: The stored zero point, or `None` for a symmetric scheme.
        g_idx: The stored group index of a grouped scheme (one entry per
            position along `scheme.group_axis`), or `None`. It is stored as
            `float32` because TensorFlow has no GPU kernel for int32
            resource variables; the view casts it on read.
        input_scales: Optional per-input-row scales divided out of the
            dequantized weight (AWQ's `awq_scales`).
        align_scale: Optional callable that aligns the stored scale with
            the unpacked codes before it is applied, for a scale whose
            layout the layer defines (an einsum kernel's equation analysis
            squeezes and transposes it). `scale` itself stays the stored
            variable, so consumers that serialize it see the stored form.
        compute_dtype: Dtype of the dequantized weight.
    """

    def __init__(
        self,
        codes,
        scale,
        layout,
        scheme,
        logical_shape,
        *,
        zero_point=None,
        g_idx=None,
        input_scales=None,
        align_scale=None,
        compute_dtype="float32",
    ):
        if scheme.symmetric != (zero_point is None):
            raise ValueError(
                "`zero_point` must be given exactly when the scheme has a "
                f"zero point. Received: scheme={scheme!r}, "
                f"zero_point={'given' if zero_point is not None else None}"
            )
        if scheme.grouped != (g_idx is not None):
            raise ValueError(
                "`g_idx` must be given exactly when the scheme is grouped. "
                f"Received: scheme={scheme!r}, "
                f"g_idx={'given' if g_idx is not None else None}"
            )
        self.codes = codes
        self.scale = scale
        self.layout = layout
        self.scheme = scheme
        self.logical_shape = tuple(int(d) for d in logical_shape)
        self.zero_point = zero_point
        self.g_idx = g_idx
        self.input_scales = input_scales
        self.align_scale = align_scale
        self.compute_dtype = compute_dtype

    @property
    def num_values(self):
        """Number of real-valued weights the codes stand for."""
        return math.prod(self.logical_shape)

    def unpack(self):
        """Returns the integer codes in the weight's logical shape."""
        return self._to_logical_shape(self.layout.unpack(self.codes))

    def dequantize(self):
        """Returns the real-valued weight in its logical shape."""
        codes = ops.cast(self.layout.unpack(self.codes), self.compute_dtype)
        if self.g_idx is not None:
            weight = dequantize_grouped(
                codes,
                self.scale,
                self.zero_point,
                self.g_idx,
                group_axis=self.scheme.group_axis,
            )
            weight = ops.cast(weight, self.compute_dtype)
        else:
            if self.zero_point is not None:
                codes = ops.subtract(
                    codes, ops.cast(self.zero_point, self.compute_dtype)
                )
            weight = ops.divide(codes, self._broadcast_scale(self.scale, codes))
        if self.input_scales is not None:
            # Per-input-row scales apply to the 2-D `[in, out]` codes, before
            # an einsum kernel is folded back to N-D.
            weight = ops.divide(weight, ops.expand_dims(self.input_scales, -1))
        return self._to_logical_shape(weight)

    def _broadcast_scale(self, scale, codes):
        """Aligns the stored scale with the codes it applies to."""
        if self.align_scale is not None:
            scale = self.align_scale(scale)
        axis = self.scheme.channel_axis
        if axis is None:
            return scale
        rank = len(codes.shape)
        axis = axis % rank
        if axis == rank - 1:
            return scale
        return ops.expand_dims(scale, axis=list(range(axis + 1, rank)))

    def _to_logical_shape(self, tensor):
        """Restores the weight's own shape (an einsum kernel is N-D)."""
        if tuple(tensor.shape) != self.logical_shape:
            tensor = ops.reshape(tensor, self.logical_shape)
        return tensor

    def __repr__(self):
        return (
            f"QTensor(logical_shape={self.logical_shape}, "
            f"layout={self.layout!r}, scheme={self.scheme!r})"
        )
