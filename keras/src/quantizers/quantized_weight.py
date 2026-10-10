"""A read-only view over the stored variables of one quantized weight.

A quantization mode stores a weight as integer codes (often packed several
to a byte), a scale, and for a grouped scheme a zero point and a group
index. `QuantizedWeight` gathers those variables with a `PackLayout`, which
says how the codes are packed, and a `WeightScheme`, which says what they
mean. `dequantize` reads the real-valued weight; `code_image` and
`pack_image` write a real-valued weight back onto the stored grid.
"""

import copy
import dataclasses
import math

from keras.src import backend
from keras.src import ops
from keras.src.quantizers.packing import pack_int2
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.packing import pack_ternary
from keras.src.quantizers.packing import unpack_int2
from keras.src.quantizers.packing import unpack_int4
from keras.src.quantizers.packing import unpack_ternary
from keras.src.quantizers.quantizers import _take_group_params
from keras.src.quantizers.quantizers import dequantize_with_sz_map

# The stored tensors of a view, in the order `read_tensors` returns them.
_TENSOR_FIELDS = ("codes", "scale", "zero_point", "g_idx", "input_scales")


@dataclasses.dataclass(frozen=True, kw_only=True)
class WeightScheme:
    """How the integer codes of one quantized weight map to real values.

    Args:
        code_range: `(min, max)` of the unpacked codes, e.g. `(-127, 127)`
            for int8 or `(0, 15)` for unsigned 4-bit codes.
        scale_form: `"divisor"` when the real value is
            `(code - zero_point) / scale` (int8, per-channel int4), or
            `"multiplier"` when it is `(code - zero_point) * scale`
            (grouped int4, GPTQ, AWQ, ternary). A grouped scale is a
            multiplier.
        has_zero_point: Whether a zero point is stored.
        group_size: Length of a group along the view's `axis` for a grouped
            scale, or `None` for a per-channel or per-tensor scale. `g_idx`
            is always authoritative: a group may be shorter, under GPTQ's
            activation order the groups are not contiguous, and on a view
            whose stored rows stack several problems the groups restart at
            each problem, so each problem's last group may be shorter.
    """

    code_range: tuple
    scale_form: str
    has_zero_point: bool = False
    group_size: int = None

    def __post_init__(self):
        if self.scale_form not in ("divisor", "multiplier"):
            raise ValueError(
                "`scale_form` must be 'divisor' or 'multiplier'. "
                f"Received: scale_form={self.scale_form!r}"
            )
        if self.group_size is not None and (
            self.scale_form != "multiplier" or not self.has_zero_point
        ):
            raise ValueError(
                "A grouped scheme stores a multiplier scale and a zero point. "
                f"Received: scale_form={self.scale_form!r}, "
                f"has_zero_point={self.has_zero_point}, "
                f"group_size={self.group_size}"
            )


class PackLayout:
    """How the codes of one weight are packed in their storage variable.

    One subclass per storage format: `values_per_byte` codes share a
    stored element along the layout's axis, `packed_length` gives the
    stored length of that axis, `unpack` restores one code per element
    (in `unpacked_shape`) and `pack` stores them.
    """

    values_per_byte = 1

    def unpack(self, codes):
        """Returns the unpacked codes, one per element."""
        raise NotImplementedError

    def pack(self, codes):
        """Returns `codes`, one per element, in their stored form."""
        raise NotImplementedError

    @classmethod
    def packed_length(cls, length):
        """Stored length of an axis holding `length` codes."""
        return length

    def unpacked_shape(self, packed_shape):
        """Shape of the codes a stored tensor of `packed_shape` holds."""
        return tuple(packed_shape)

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

    @classmethod
    def packed_length(cls, length):
        return math.ceil(length / cls.values_per_byte)

    def unpacked_shape(self, packed_shape):
        shape = list(packed_shape)
        shape[self.axis] = self.orig_len
        return tuple(shape)

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

    def pack(self, codes):
        dtype = backend.standardize_dtype(codes.dtype)
        packed, _, _ = pack_int4(codes, axis=self.axis, dtype=dtype)
        return packed


class Int2Quads(_AxisPack):
    """Four 2-bit codes per byte along `axis`, as `pack_int2` writes them."""

    values_per_byte = 4

    def unpack(self, codes):
        dtype = backend.standardize_dtype(codes.dtype)
        return unpack_int2(codes, self.orig_len, axis=self.axis, dtype=dtype)

    def pack(self, codes):
        dtype = backend.standardize_dtype(codes.dtype)
        packed, _, _ = pack_int2(codes, axis=self.axis, dtype=dtype)
        return packed


class TernaryTrits(_AxisPack):
    """Five ternary codes per byte along `axis`, in base 3.

    The unpack is arithmetic, not a mask and shift, so a bit-width cannot
    describe it.
    """

    values_per_byte = 5

    def unpack(self, codes):
        return unpack_ternary(codes, self.orig_len, axis=self.axis)

    def pack(self, codes):
        packed, _, _ = pack_ternary(codes, axis=self.axis)
        return packed


def lay_out_scale(scale, source, target):
    """Lays a scale with the axes `source` out against the axes `target`.

    `source` and `target` give one label per axis of the scale and of the
    tensor it broadcasts against. An axis of `source` whose label is not
    in `target` has size one and is dropped (`None` marks such an axis).
    The other axes take the order of `target`, and an axis of `target`
    that is not in `source` becomes a broadcast axis of size one.
    """
    source, target = list(source), list(target)
    kept = [label for label in source if label in target]
    dropped = [i for i, label in enumerate(source) if label not in target]
    if dropped:
        scale = ops.squeeze(scale, axis=dropped)
    order = [kept.index(label) for label in target if label in kept]
    if order != sorted(order):
        scale = ops.transpose(scale, order)
    added = [i for i, label in enumerate(target) if label not in source]
    if added:
        scale = ops.expand_dims(scale, axis=added)
    return scale


class QuantizedWeight:
    """A read-only view over the stored variables of one quantized weight.

    The view holds references to the tensors a mode stores for the weight
    (the layer's variables, or traced tensors inside a forward pass) and
    does not cache, so it reads their current values.
    `QuantizationStrategy.quantized_weight(layer)` builds it on demand. A
    forward pass that passes the tensors to `ops.custom_gradient` reads
    them with `read_tensors` and gets the same view over them with
    `with_tensors`.

    The codes are stored as the weight transposed and reshaped:
    `layout.unpack(codes)` is `reshape(transpose(W, permutation),
    layout.unpacked_shape(codes.shape))`, where `W` is the weight in
    `shape` (an int4, GPTQ or AWQ einsum kernel is stored as 2-D). `axis`,
    `scale_axes` and the stored scale refer to these stored coordinates;
    `unpack()` and `dequantize()` return `shape`.

    The scheme sets how the stored scale lines up with the codes. A
    grouped weight gives the one `axis` its groups run along. Any other
    weight gives `scale_axes`.

    Args:
        codes: The stored codes, packed as `layout` describes.
        scale: The stored scale, applied as `scheme.scale_form` says.
        layout: The `PackLayout` of `codes`.
        scheme: The `WeightScheme` of the weight.
        shape: Shape of the weight the codes stand for.
        axis: For a grouped weight only: the axis of the unpacked codes
            its groups run along. The scale and zero point have one entry
            per group along it, and `g_idx` maps each position on it to
            its group. When the stored rows stack independent problems
            (the batch axis of a GPTQ or AWQ einsum kernel), the groups
            restart at each problem and are numbered across the problems,
            so the scale holds `batch * n_groups` rows and each problem's
            last group may be shorter; `g_idx` is authoritative. An int4
            einsum kernel never stacks problems: its batch axes are in the
            columns.
        permutation: Axis order of `shape` in which the codes are stored,
            or `None` for the weight's own order.
        zero_point: The stored zero point, given exactly when
            `scheme.has_zero_point`.
        g_idx: The stored group index, one entry per position along
            `axis`, given exactly when `scheme.group_size` is set.
        input_scales: Optional scales with one entry per position along
            `axis`, divided out of the weight (AWQ's `awq_scales`). The
            codes, scale and zero point describe the weight multiplied by
            them.
        scale_axes: For a weight that is not grouped only: for each axis
            of the stored scale, the axis of the unpacked codes it runs
            along, or `None` for an axis of size one. `()` is a per-tensor
            scalar. A per-channel scale of a matmul kernel is `(1,)`; an
            int8 einsum kernel's scale is stored in the outputs' layout.
    """

    def __init__(
        self,
        *,
        codes,
        scale,
        layout,
        scheme,
        shape,
        axis=None,
        permutation=None,
        zero_point=None,
        g_idx=None,
        input_scales=None,
        scale_axes=None,
    ):
        if scheme.has_zero_point != (zero_point is not None):
            raise ValueError(
                "`zero_point` must be given exactly when the scheme has a "
                f"zero point. Received: scheme={scheme!r}, "
                f"zero_point={'given' if zero_point is not None else None}"
            )
        if (scheme.group_size is not None) != (g_idx is not None):
            raise ValueError(
                "`g_idx` must be given exactly when the scheme is grouped. "
                f"Received: scheme={scheme!r}, "
                f"g_idx={'given' if g_idx is not None else None}"
            )
        if g_idx is not None:
            if not isinstance(axis, int) or scale_axes is not None:
                raise ValueError(
                    "A grouped weight needs the one `axis` its groups run "
                    "along, and no `scale_axes`. Received: "
                    f"axis={axis}, scale_axes={scale_axes}"
                )
        elif axis is not None or scale_axes is None or input_scales is not None:
            raise ValueError(
                "A weight that is not grouped lines its scale up by "
                "`scale_axes` (`()` for a scalar), and takes no `axis` and "
                "no `input_scales`. Received: "
                f"axis={axis}, scale_axes={scale_axes}, "
                f"input_scales={'given' if input_scales is not None else None}"
            )
        self.codes = codes
        self.scale = scale
        self.layout = layout
        self.scheme = scheme
        self.shape = tuple(int(d) for d in shape)
        self.axis = axis
        if permutation is not None:
            permutation = tuple(permutation)
            if permutation == tuple(range(len(self.shape))):
                permutation = None
        self.permutation = permutation
        self.zero_point = zero_point
        self.g_idx = g_idx
        self.input_scales = input_scales
        self.scale_axes = None if scale_axes is None else tuple(scale_axes)

    def read_tensors(self):
        """Returns the stored tensors, each variable through its `value`.

        The order is `codes` and `scale`, then `zero_point`, `g_idx` and
        `input_scales` when given. A float variable that autocasts reads in
        the dtype of the current autocast scope.
        """
        tensors = [getattr(self, name) for name in self._tensor_names()]
        return tuple(
            ops.convert_to_tensor(
                tensor.value if isinstance(tensor, backend.Variable) else tensor
            )
            for tensor in tensors
        )

    def with_tensors(self, tensors):
        """Returns this view over `tensors`, in the order of `read_tensors`."""
        view = copy.copy(self)
        for name, tensor in zip(self._tensor_names(), tensors, strict=True):
            setattr(view, name, tensor)
        return view

    def _tensor_names(self):
        """Names of the stored tensors that are given, in order."""
        return [
            name for name in _TENSOR_FIELDS if getattr(self, name) is not None
        ]

    def unpack(self):
        """Returns the integer codes in `shape`."""
        return self._restore_shape(self.layout.unpack(self.codes))

    def dequantize(self, dtype):
        """Returns the real-valued weight in `shape`, as `dtype`.

        The codes are cast to `dtype` before the arithmetic, and the result
        is cast to `dtype` after it.
        """
        codes = ops.cast(self.layout.unpack(self.codes), dtype)
        if self.g_idx is not None:
            weight = dequantize_with_sz_map(
                codes,
                self.scale,
                self.zero_point,
                self.g_idx,
                group_axis=self.axis,
            )
        else:
            if self.zero_point is not None:
                zero_point = ops.cast(self._align(self.zero_point), dtype)
                codes = ops.subtract(codes, zero_point)
            if self.scheme.scale_form == "multiplier":
                weight = ops.multiply(codes, self._align(self.scale))
            else:
                weight = ops.divide(codes, self._align(self.scale))
        if self.input_scales is not None:
            weight = ops.divide(
                weight, self._along_axis(self.input_scales, weight)
            )
        return self._restore_shape(ops.cast(weight, dtype))

    def code_image(self, weight):
        """Returns the unrounded codes of a real-valued `weight`.

        The inverse of `dequantize` under the stored parameters of a
        grouped multiplier scheme: `weight`, in `shape`, is laid out in
        stored coordinates, multiplied by `input_scales`, divided by its
        group's scale and shifted by its zero point, in `float32`. An entry
        that rounds outside `scheme.code_range` is a weight the stored
        parameters do not cover.
        """
        if self.g_idx is None or self.scheme.scale_form != "multiplier":
            raise NotImplementedError(
                "`code_image` supports grouped multiplier schemes only. "
                f"Received: scheme={self.scheme!r}"
            )
        image = self._as_stored(ops.cast(weight, "float32"))
        if self.input_scales is not None:
            input_scales = ops.cast(self.input_scales, "float32")
            image = ops.multiply(image, self._along_axis(input_scales, image))
        scales, zeros = _take_group_params(
            self.scale, self.zero_point, self.g_idx, self.axis
        )
        image = ops.divide(image, ops.cast(scales, "float32"))
        return ops.add(image, ops.cast(zeros, "float32"))

    def pack_image(self, image):
        """Returns the stored codes of a rounded code image.

        `image` is `code_image(weight)` rounded. It is clipped to
        `scheme.code_range`, cast to the dtype of `codes` and packed.
        """
        low, high = self.scheme.code_range
        codes = ops.cast(ops.clip(image, low, high), self.codes.dtype)
        return self.layout.pack(codes)

    def _align(self, tensor):
        """Lays a scale or zero point out against the unpacked codes."""
        return lay_out_scale(
            tensor, self.scale_axes, range(len(self.codes.shape))
        )

    def _along_axis(self, tensor, like):
        """Reshapes a 1-D `tensor` to broadcast along `axis` of `like`."""
        shape = [1] * len(like.shape)
        shape[self.axis] = -1
        return ops.reshape(tensor, shape)

    def _as_stored(self, weight):
        """Lays `weight`, in `shape`, out in stored coordinates."""
        if self.permutation is not None:
            weight = ops.transpose(weight, self.permutation)
        return ops.reshape(weight, self.layout.unpacked_shape(self.codes.shape))

    def _restore_shape(self, tensor):
        """Reshapes and transposes stored coordinates to `shape`."""
        if self.permutation is not None:
            permuted = [self.shape[axis] for axis in self.permutation]
            inverse = sorted(
                range(len(self.shape)), key=self.permutation.__getitem__
            )
            return ops.transpose(ops.reshape(tensor, permuted), inverse)
        if tuple(tensor.shape) != self.shape:
            tensor = ops.reshape(tensor, self.shape)
        return tensor

    def __repr__(self):
        return (
            f"{type(self).__name__}(shape={self.shape}, axis={self.axis}, "
            f"scale_axes={self.scale_axes}, "
            f"permutation={self.permutation}, layout={self.layout!r}, "
            f"scheme={self.scheme!r})"
        )
