"""Quantized weight views: `WeightScheme`, `PackLayout`, `QVariable`.

A quantization mode stores one weight as a handful of variables: integer
codes (often packed several to a byte), a scale, and for asymmetric
schemes a zero point and a group index. A `QVariable` is a read-only view
that gathers the variables of one weight together with two descriptions
of how to read them:

- `PackLayout` says how the codes are stored: how many fit in a byte, along
  which axis, and how to unpack them. It is a sum type, one subclass per
  storage format, because the formats differ in behavior (a base-3 trit
  pack cannot be described by a bitfield mask) and not only in data.
- `WeightScheme` says what the codes mean: their range, whether there is
  a zero point, and how groups run. It is one flat record, because every
  consumer reads the fields as a unit.

The view has two surfaces, split by what they return. `unpack()` returns
the integer codes in the weight's own shape and orientation, which is
what a layer's `kernel` or `embeddings` property exposes for a quantized
layer. `dequantize()` returns the real-valued weight, which is what the
LoRA-merged save path, the calibration forward pass, and any exporter
need. Neither surface caches: a view is built on demand by
`QuantizationStrategy.qvariable(layer)` and holds references to the
layer's variables, so it always reflects their current values.

Every mode stores the scale it divides by, the form the abs-max
quantizers and `ternarize` produce; the calibration quantizers compute
the multiplier form of their references and store its reciprocal. So
`dequantize()` is one formula, `(code - zero_point) / scale`, with the
per-group values gathered through the group index for a grouped scheme
and, when `input_scales` is set, each input row divided by its scale.

`float8` has no view: it keeps the float kernel and stores dynamic-range
state, not a codebook, so `QuantizationStrategy.qvariable` returns `None`
for it, as it does for a calibration mode before its calibration pass.
"""

import dataclasses
import math

from keras.src import backend
from keras.src import ops
from keras.src.quantizers.packing import unpack_int2
from keras.src.quantizers.packing import unpack_int4
from keras.src.quantizers.packing import unpack_ternary
from keras.src.quantizers.quantizers import dequantize_grouped


@dataclasses.dataclass(frozen=True, kw_only=True)
class WeightScheme:
    """How the integer codes of one quantized weight map to real values.

    A flat, immutable record. Every field describes the stored variables
    as they are, so a scheme can be written down for any of the shipped
    modes without changing what they store.

    Args:
        code_range: `(min, max)` of the codes, e.g. `(-127, 127)` for int8
            or `(0, 15)` for unsigned 4-bit calibration codes.
        has_zero_point: Whether a zero point is stored. The real value is
            `(code - zero_point) / scale`; a symmetric scheme has none.
        channel_axis: For a per-channel divisor scale, the axis of the
            unpacked codes the scale runs along. `None` means the stored
            scale broadcasts against the codes as it is: a per-tensor
            scalar, or a scale the view's `align_scale` lays out.
        group_size: Number of codes per group for a grouped scheme, or
            `None` for per-channel or per-tensor scaling.
        group_axis: For a grouped scheme, the axis of the unpacked codes
            along which the groups run. The scale and zero point hold one
            entry per group along that same axis, and the group index maps
            each position on it to its group.
    """

    code_range: tuple
    has_zero_point: bool = False
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

    @property
    def bits(self):
        """Bit-width of the code domain: the smallest width holding every
        code. Ternary codes take two bits even though five pack into a
        byte; the storage density is the `PackLayout`'s business.
        """
        low, high = self.code_range
        return (high - low).bit_length()

    @property
    def signed(self):
        """Whether the codes are signed."""
        return self.code_range[0] < 0

    @property
    def grouped(self):
        """Whether the scale runs per group rather than per channel."""
        return self.group_size is not None


class PackLayout:
    """How the codes of one weight are laid out in their storage variable.

    Each subclass owns one storage format: the number of codes per byte
    (`values_per_byte`, from which `packed_length` derives the stored
    length of an axis when a mode builds its code variable), the axis the
    codes are packed along, and the unpack op. A mode's `encode` packs
    through the `quantizers` functions directly.
    """

    values_per_byte = 1

    def unpack(self, codes):
        """Returns the unpacked codes, one per element."""
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
    """Two 4-bit codes per byte along `axis`, as `pack_int4` writes them.

    The codes' own dtype (`int8` or `uint8`) says whether a nibble is
    sign-extended on unpack.
    """

    values_per_byte = 2

    def unpack(self, codes):
        dtype = backend.standardize_dtype(codes.dtype)
        return unpack_int4(codes, self.orig_len, axis=self.axis, dtype=dtype)


class Int2Quads(_AxisPack):
    """Four 2-bit codes per byte along `axis`, as `pack_int2` writes them."""

    values_per_byte = 4

    def unpack(self, codes):
        dtype = backend.standardize_dtype(codes.dtype)
        return unpack_int2(codes, self.orig_len, axis=self.axis, dtype=dtype)


class TernaryTrits(_AxisPack):
    """Five ternary codes per byte along `axis`, in base 3.

    The unpack is arithmetic (`mod(floor_divide(byte, 3**k), 3) - 1`), not
    a mask and shift, which is why the layout is a subclass rather than a
    bit-width flag.
    """

    values_per_byte = 5

    def unpack(self, codes):
        return unpack_ternary(codes, self.orig_len, axis=self.axis)


class QVariable:
    """A read-only view over the stored variables of one quantized weight.

    It is not a `keras.Variable`. It holds references to the tensors a
    mode stores for one weight (normally the layer's code, scale, zero
    point and group index variables; the int4 projection mode also builds
    it over traced tensors) and reads them through a `PackLayout` and a
    `WeightScheme`. `QuantizationStrategy.qvariable(layer)` builds it on
    demand and never caches it; it is `None` for `float8` and for a
    calibration mode before its calibration pass.

    Args:
        codes: The stored code variable, packed as `layout` describes.
            `unpack()` returns one code per weight, in `shape`.
        scale: The stored divisor scale: the real value is
            `(code - zero_point) / scale`.
        layout: The `PackLayout` of `codes`.
        scheme: The `WeightScheme` of the weight.
        shape: Shape of the weight the codes stand for, in the weight's
            own orientation; `codes.shape` is the stored, possibly packed,
            shape.
        zero_point: The stored zero point, or `None` for a scheme without
            one.
        g_idx: The stored group index of a grouped scheme (one entry per
            position along `scheme.group_axis`), or `None`. It is stored
            as `float32` because TensorFlow has no GPU kernel for int32
            resource variables; `dequantize_grouped` casts it on read.
        input_scales: Optional per-input-row scales divided out of the
            dequantized weight (AWQ's `awq_scales`). The codes, scale and
            zero point describe the weight after each input row was
            multiplied by these scales, so a reader that writes the
            stored triple to another format must carry `input_scales`
            with it.
        align_scale: Optional callable that lays the stored scale out
            against the unpacked codes, for a scale whose layout the
            layer defines (an einsum kernel's equation analysis squeezes
            and transposes it). It replaces `scheme.channel_axis`, which
            must then be `None`; `scale` itself stays the stored variable,
            so consumers that serialize it see the stored form.
        permutation: Optional axis order in which the stored codes
            flatten `shape`: the unpacked codes are the weight transposed
            by `permutation` and flattened to 2-D, as the calibration
            modes store an einsum kernel whose contracted axes do not
            lead. `None` (or the identity) keeps the weight's own order.
        compute_dtype: Dtype the unpacked codes are cast to before the
            arithmetic. A grouped scheme returns this dtype; a per-channel
            or per-tensor scheme returns its promotion with the scale's
            dtype (`float32` under a mixed policy), which the int8 and
            per-channel int4 consumers rely on bit for bit.
    """

    def __init__(
        self,
        *,
        codes,
        scale,
        layout,
        scheme,
        shape,
        zero_point=None,
        g_idx=None,
        input_scales=None,
        align_scale=None,
        permutation=None,
        compute_dtype="float32",
    ):
        if scheme.has_zero_point != (zero_point is not None):
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
        if align_scale is not None and (
            scheme.grouped or scheme.channel_axis is not None
        ):
            raise ValueError(
                "`align_scale` lays the stored scale out against the codes "
                "itself, so it needs a scheme with no `channel_axis` and "
                f"no groups. Received: scheme={scheme!r}"
            )
        shape = tuple(int(d) for d in shape)
        if permutation is not None:
            permutation = tuple(int(axis) for axis in permutation)
            if sorted(permutation) != list(range(len(shape))):
                raise ValueError(
                    "`permutation` must permute the axes of `shape`. "
                    f"Received: permutation={permutation}, shape={shape}"
                )
            if permutation == tuple(range(len(shape))):
                permutation = None
        self.codes = codes
        self.scale = scale
        self.layout = layout
        self.scheme = scheme
        self.shape = shape
        self.permutation = permutation
        self.zero_point = zero_point
        self.g_idx = g_idx
        self.input_scales = input_scales
        self.align_scale = align_scale
        self.compute_dtype = compute_dtype

    @property
    def num_values(self):
        """Number of real-valued weights the codes stand for."""
        return math.prod(self.shape)

    def unpack(self):
        """Returns the integer codes in `shape`."""
        return self._restore_shape(self.layout.unpack(self.codes))

    def dequantize(self):
        """Returns the real-valued weight in `shape`.

        See `compute_dtype` for the result dtype.
        """
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
        return self._restore_shape(weight)

    def _broadcast_scale(self, scale, codes):
        """Aligns the stored scale with the codes it applies to."""
        if self.align_scale is not None:
            return self.align_scale(scale)
        axis = self.scheme.channel_axis
        if axis is None:
            return scale
        shape = [1] * len(codes.shape)
        shape[axis] = -1
        return ops.reshape(scale, shape)

    def _restore_shape(self, tensor):
        """Restores `shape`; an einsum kernel is N-D."""
        if self.permutation is not None:
            permuted = tuple(self.shape[axis] for axis in self.permutation)
            inverse = [0] * len(self.permutation)
            for position, axis in enumerate(self.permutation):
                inverse[axis] = position
            return ops.transpose(ops.reshape(tensor, permuted), inverse)
        if tuple(tensor.shape) != self.shape:
            tensor = ops.reshape(tensor, self.shape)
        return tensor

    def __repr__(self):
        return (
            f"{type(self).__name__}(shape={self.shape}, "
            f"layout={self.layout!r}, scheme={self.scheme!r})"
        )
