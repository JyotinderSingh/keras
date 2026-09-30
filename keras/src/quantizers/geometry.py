"""Quantization geometry: the layer-side protocol behind the strategy registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The strategies in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

Two geometry families exist today:

- Projection: a float kernel contracted against the inputs. A strategy writes
  one projection implementation and the geometry supplies what differs per
  layer: how to contract (a plain matmul for `Dense` and `TernaryDense`,
  `ProjectionGeometry`; an einsum for `EinsumDense`,
  `EinsumProjectionGeometry`, whose axis analysis lives on the layer
  itself and is reached through the geometry's hooks), which axes the
  quantizers reduce over, how a scale lines up with the kernel and with
  the outputs, the 2D `(contracted, rest)` matrix of an N-D kernel, and
  the `ContractionView` the calibration modes derive from the equation.
- Lookup: a float embeddings table indexed by the inputs. `Embedding` is the
  plain case (`LookupGeometry`); `ReversibleEmbedding` adds a reverse
  projection (`ReversibleLookupGeometry`).

Making a layer quantizable
--------------------------

Return a geometry, and list the modes the layer supports:

```python
class MyProjection(Layer):
    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        # Doubles as the capability declaration: a mode absent from this
        # mapping is rejected for this layer.
        return {
            None: ["kernel", "bias"],
            "int8": ["kernel", "bias", "kernel_scale"],
        }
```

The geometry is a thin adapter, so the strategies still read
state directly off the layer. Beyond what `Layer` already provides, a
quantizable layer must define:

- Projections: `_kernel` (the float kernel variable), `kernel_shape` (its
  shape, recorded in `build()`), `units`, `bias` and `activation` (either
  may be `None`). `EinsumProjectionGeometry` additionally relies on the
  `einsum_axes` record `EinsumDense` derives from its equation in `build()`.
- Lookups: `_embeddings`, `input_dim` and `output_dim`. A reversible
  lookup adds `tie_weights`, `logit_soft_cap`, and, when untied, the
  `reverse_embeddings` variables.

LoRA is optional: `Layer` defines `lora_enabled = False`. A layer that
supports it sets `lora_enabled` in `enable_lora()` and defines
`lora_kernel_a`, `lora_kernel_b`, `lora_alpha` and `lora_rank` (a lookup
defines `lora_embeddings_a` and `lora_embeddings_b` in place of the kernel
factors).

The rest comes from `Layer` itself: strategies read `compute_dtype`,
`dtype_policy` and `path`, create their quantized variables through
`add_weight`, and re-enter through `Layer.quantized_build`, which routes
straight back to the strategy. A layer never needs to know which mode is
running, and implements none of these itself.

Defining `_quantization_geometry()` on a subclass also makes that subclass
the owner of its quantization support: `Layer.quantize`'s type check
accepts instances of the exact class that defines the method. A `Dense`
subclass therefore opts in by defining it; without it the subclass is
skipped by `Model.quantize` and remains reachable through
`quantize(..., type_check=False)`.

Customizing what a strategy does to a layer
-------------------------------------------

Override a geometry hook rather than a strategy method: the hooks on the
classes below are the only points at which strategies vary per
layer. `TernaryDense` is the in-tree example: its geometry supplies its
own straight-through ternarization values, and the ternary strategy needs no
knowledge of the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one strategy's math for itself alone, because that surface lives
on the strategy; a layer that contracts its kernel differently overrides
the geometry hooks, and anything beyond that means replacing the strategy (by
subclassing it, overriding the one handler, and registering it under a
new mode name). A new geometry family, on the other hand, needs no dispatcher
change at all: declare its `family` and implement the strategy's
`_<verb>_<family>` handlers, which `GeometryDispatchStrategy`
(`keras.src.quantizers.modes.common`) lists.
"""

import math
import string

from keras.src import ops
from keras.src.quantizers.quantizers import ternarize


class ContractionView:
    """A projection kernel as independent `(rows, columns)` matrices.

    The rows are the kernel's contracted axes and the columns the axes
    that reach the output from the kernel alone, each flattened in the
    kernel's order. An axis shared by the inputs, the kernel and the
    output (the expert axis of a mixture-of-experts down projection) is a
    batch axis: each of its `batch` indices is an independent problem with
    its own slice of the inputs.

    `kernel_to_view` lays the kernel out as `(batch, rows, columns)`, which
    is the kernel transposed by `kernel_permutation` and reshaped, and
    `inputs_to_view` lays the inputs out as `(batch, samples, rows)`, or
    `(samples, rows)` when `batch` is 1, so a row index means the same
    contracted position on both sides.

    Args:
        kernel_shape: The kernel's own shape.
        kernel_batch_axes: Kernel axes shared with the inputs and output.
        kernel_contracted_axes: Kernel axes contracted with the inputs.
        kernel_free_axes: Kernel axes that reach the output alone.
        input_batch_axes: Input axes matching `kernel_batch_axes`, in the
            same order.
        input_contracted_axes: Input axes matching
            `kernel_contracted_axes`, in the same order.
    """

    def __init__(
        self,
        kernel_shape,
        *,
        kernel_batch_axes,
        kernel_contracted_axes,
        kernel_free_axes,
        input_batch_axes,
        input_contracted_axes,
    ):
        self.kernel_shape = tuple(int(d) for d in kernel_shape)
        self.kernel_permutation = (
            tuple(kernel_batch_axes)
            + tuple(kernel_contracted_axes)
            + tuple(kernel_free_axes)
        )
        if sorted(self.kernel_permutation) != list(
            range(len(self.kernel_shape))
        ):
            raise ValueError(
                "The batch, contracted and free axes must partition the "
                f"kernel axes. Received: kernel_shape={self.kernel_shape}, "
                f"kernel_batch_axes={tuple(kernel_batch_axes)}, "
                f"kernel_contracted_axes={tuple(kernel_contracted_axes)}, "
                f"kernel_free_axes={tuple(kernel_free_axes)}"
            )
        self.batch = math.prod(self.kernel_shape[i] for i in kernel_batch_axes)
        self.rows = math.prod(
            self.kernel_shape[i] for i in kernel_contracted_axes
        )
        self.columns = math.prod(self.kernel_shape[i] for i in kernel_free_axes)
        self.input_batch_axes = tuple(input_batch_axes)
        self.input_contracted_axes = tuple(input_contracted_axes)

    @property
    def permuted(self):
        """Whether the view reorders the kernel axes."""
        return self.kernel_permutation != tuple(range(len(self.kernel_shape)))

    def kernel_to_view(self, kernel):
        """Lays the kernel out as `(batch, rows, columns)`."""
        if self.permuted:
            kernel = ops.transpose(kernel, self.kernel_permutation)
        return ops.reshape(kernel, (self.batch, self.rows, self.columns))

    def input_features(self, inputs):
        """Number of contracted features the inputs carry per sample."""
        rank = len(inputs.shape)
        return math.prod(
            inputs.shape[axis % rank] for axis in self.input_contracted_axes
        )

    def inputs_to_view(self, inputs):
        """Lays the layer's inputs out as `(batch, samples, rows)`.

        The result is `(samples, rows)` when the kernel has no batch axis
        (or a batch axis of size one).
        """
        rank = len(inputs.shape)
        batch = [axis % rank for axis in self.input_batch_axes]
        contracted = [axis % rank for axis in self.input_contracted_axes]
        free = [
            axis
            for axis in range(rank)
            if axis not in batch and axis not in contracted
        ]
        order = batch + free + contracted
        if order != list(range(rank)):
            inputs = ops.transpose(inputs, order)
        if self.batch > 1:
            return ops.reshape(inputs, (self.batch, -1, self.rows))
        return ops.reshape(inputs, (-1, self.rows))


class QuantizationGeometry:
    """Base class for a layer's quantization geometry.

    A geometry names the *family* it belongs to. A strategy built on
    `GeometryDispatchStrategy` implements one `_<verb>_<family>` handler
    per verb for each family it supports, so introducing a family is a
    declaration plus those handlers, with no dispatch chain to edit
    anywhere.
    """

    # Dispatch key: each `GeometryDispatchStrategy` verb resolves to
    # `_<verb>_<family>`.
    family = None
    # Whether the layer also projects back through its weight. Not a
    # dispatch key: strategies branch on it where the reverse table
    # matters, and the forward handler takes the layer's `reverse` argument.
    reversible = False
    # Attributes the layer's `build()` records and the strategies read.
    # `Layer.quantize` checks them before it changes the layer.
    build_attributes = ()

    def __init__(self, layer):
        self.layer = layer

    @property
    def weight_shape(self):
        """Shape of the float weight that quantization replaces."""
        raise NotImplementedError(
            f"{type(self).__name__} must define `weight_shape`."
        )


class ProjectionGeometry(QuantizationGeometry):
    """Geometry of a 2D kernel `(input_dim, units)` contracted by matmul."""

    family = "projection"
    build_attributes = ("kernel_shape",)

    @property
    def weight_shape(self):
        """Shape of the float kernel, as the layer recorded it in `build()`.

        Quantized storage may be packed or flattened, so this is the
        logical shape every strategy reads rather than a variable's shape.
        """
        return tuple(self.layer.kernel_shape)

    def contraction_view(self):
        """The kernel's `ContractionView`, derived from the contraction.

        GPTQ and AWQ calibrate on it. Unlike `kernel_matrix`, it stacks the
        problems of a batch axis along the rows, because act-order `g_idx`
        and AWQ's input scales differ per problem and lie along the rows.
        """
        return ContractionView(
            self.weight_shape,
            kernel_batch_axes=(),
            kernel_contracted_axes=(0,),
            kernel_free_axes=(1,),
            input_batch_axes=(),
            input_contracted_axes=(-1,),
        )

    def contract(self, inputs, kernel):
        """Contracts `inputs` against a kernel in the contraction shape."""
        return ops.matmul(inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        """Gradient of `contract` with respect to its inputs."""
        return ops.matmul(upstream, ops.transpose(float_kernel))

    def kernel_matrix(self, kernel_shape):
        """The kernel's 2D `(contracted, rest)` matrix, as int4 stores it.

        Returns `(permutation, rows, columns)`: the kernel transposed by
        `permutation` and reshaped to `(rows, columns)` has its contracted
        axes as the rows and every other axis, in the kernel's order, as
        the columns. A batch axis (the experts of a mixture-of-experts down
        projection) lands in the columns, so every column belongs to one
        problem. That keeps the released int4 bytes of a layout whose
        contracted axes lead, and a per-channel scale per problem and
        column. Without a batch axis it is the `contraction_view` matrix.
        """
        return (0, 1), kernel_shape[0], kernel_shape[1]

    @property
    def kernel_reduced_axes(self):
        """Kernel axes a weight quantizer reduces over."""
        return 0

    @property
    def inputs_quantization_axis(self):
        """Input axes an activation quantizer reduces over."""
        return -1

    def align_inputs_scale(self, scale):
        """Aligns an activation scale with the contraction's outputs."""
        return scale

    def kernel_scale_shape(self, kernel_shape):
        """Shape of a per-channel scale stored alongside the kernel."""
        return (kernel_shape[1],)

    @property
    def kernel_scale_axis(self):
        """Kernel axis a per-channel scale is shared along.

        `None` when the stored scale is laid out for the outputs and
        `kernel_scale_for_dequant` lays it out against the kernel instead.
        """
        return 0

    def kernel_scale_for_storage(self, scale):
        """Aligns a freshly computed kernel scale with its stored layout."""
        return ops.squeeze(scale, axis=0)

    def kernel_scale_for_dequant(self, scale):
        """Aligns the stored kernel scale with the kernel for dequantization."""
        return scale

    def add_lora_delta(self, inputs, x):
        """Adds the LoRA update to the contraction's output, when enabled."""
        layer = self.layer
        if layer.lora_enabled:
            lora_x = ops.matmul(inputs, layer.lora_kernel_a)
            lora_x = ops.matmul(lora_x, layer.lora_kernel_b)
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, layer.compute_dtype)
        return x

    def ternary_values(self):
        """Returns `(ternary_kernel, scale)` for ternary quantization.

        The default applies the BitNet b1.58 rule to the float kernel
        (`quantizers.ternarize`): `threshold = 0.5 * mean(|W|)` and
        `scale = mean(|W|)`. A layer that owns its own ternarization rule
        (`TernaryDense` and its straight-through estimator) overrides this
        in its geometry.
        """
        return ternarize(self.layer._kernel)


def _lora_equations(equation):
    """The two einsums that apply a LoRA update to `equation` in low-rank
    form, contracting the rank axis by name.

    `lora_kernel_a` carries the rank on the kernel's last axis and
    `lora_kernel_b` maps it to that axis's size. Contracting the rank with
    a matmul would only work when the kernel's last axis is also the last
    axis of the output; naming it works for every equation (an ellipsis in
    the output, a permuted output, or a kernel whose last axis is
    contracted away).

    Returns:
        `(first, second, a_first)`: `first` contracts the inputs against
        the factor that shares their subscripts, `second` contracts the
        rank axis against the other factor; `a_first` is whether that
        order is `(lora_kernel_a, lora_kernel_b)`, which holds when the
        kernel's last axis survives in the output.
    """
    inputs_spec, rest = equation.split(",")
    kernel_spec, output_spec = rest.split("->")
    last = kernel_spec[-1]
    rank = next(c for c in string.ascii_letters if c not in equation)
    if last in output_spec:
        # A last axis the inputs share too (a batch axis) is carried
        # through the first einsum rather than replaced by the rank.
        if last in inputs_spec:
            mid = f"{output_spec}{rank}"
        else:
            mid = output_spec.replace(last, rank)
        return (
            f"{inputs_spec},{kernel_spec[:-1]}{rank}->{mid}",
            f"{mid},{rank}{last}->{output_spec}",
            True,
        )
    mid = inputs_spec.replace(last, rank)
    return (
        f"{inputs_spec},{rank}{last}->{mid}",
        f"{mid},{kernel_spec[:-1]}{rank}->{output_spec}",
        False,
    )


class EinsumProjectionGeometry(ProjectionGeometry):
    """Geometry of an N-D einsum kernel (`EinsumDense`).

    The equation-derived axis analysis (`EinsumDense.einsum_axes`) is
    the layer's own geometry implementation; this class routes the
    strategies to it.
    """

    build_attributes = ("kernel_shape", "einsum_axes")

    def contraction_view(self):
        axes = self.layer.einsum_axes
        # The equation analysis already refuses a kernel axis absent from
        # both the inputs and the output; what is left to check is that the
        # kernel has something to contract and something to output, and
        # that the inputs are not summed over an axis on their own.
        if (
            not axes.kernel_reduced_axes
            or not axes.kernel_free_axes
            or sorted(axes.input_contracted_axes)
            != sorted(axes.input_reduced_axes)
        ):
            raise ValueError(
                "Cannot derive a contraction view for the `EinsumDense` "
                f"equation '{self.layer.equation}'. The kernel needs at "
                "least one axis contracted with the inputs and one axis "
                "that reaches the output, and the inputs must not be summed "
                "over an axis of their own. Exclude the layer with "
                "`filters`."
            )
        return ContractionView(
            self.weight_shape,
            kernel_batch_axes=axes.kernel_batch_axes,
            kernel_contracted_axes=axes.kernel_reduced_axes,
            kernel_free_axes=axes.kernel_free_axes,
            input_batch_axes=axes.input_batch_axes,
            input_contracted_axes=axes.input_contracted_axes,
        )

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        # From https://stackoverflow.com/a/47609896
        return ops.einsum(
            self.layer.einsum_axes.custom_gradient_equation,
            upstream,
            float_kernel,
        )

    def kernel_matrix(self, kernel_shape):
        contracted = self.layer.einsum_axes.kernel_reduced_axes
        rest = tuple(i for i in range(len(kernel_shape)) if i not in contracted)
        return (
            contracted + rest,
            math.prod(kernel_shape[i] for i in contracted),
            math.prod(kernel_shape[i] for i in rest),
        )

    @property
    def kernel_reduced_axes(self):
        return self.layer.einsum_axes.kernel_reduced_axes

    @property
    def inputs_quantization_axis(self):
        return tuple(self.layer.einsum_axes.input_reduced_axes)

    def align_inputs_scale(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "input")

    def kernel_scale_shape(self, kernel_shape):
        return self.layer._get_kernel_scale_shape(kernel_shape)

    @property
    def kernel_scale_axis(self):
        # The equation analysis may transpose or expand the stored scale
        # even for a 2-D kernel; `kernel_scale_for_dequant` lays it out.
        return None

    def kernel_scale_for_storage(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "kernel")

    def kernel_scale_for_dequant(self, scale):
        return self.layer._adjust_scale_for_dequant(scale)

    def add_lora_delta(self, inputs, x):
        layer = self.layer
        if layer.lora_enabled:
            first, second, a_first = _lora_equations(layer.equation)
            factors = (layer.lora_kernel_a, layer.lora_kernel_b)
            if not a_first:
                factors = factors[::-1]
            lora_x = ops.einsum(first, inputs, factors[0])
            lora_x = ops.einsum(second, lora_x, factors[1])
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, dtype=layer.compute_dtype)
        return x


class LookupGeometry(QuantizationGeometry):
    """Geometry of an embeddings table indexed by integer inputs."""

    family = "lookup"

    @property
    def weight_shape(self):
        return (self.layer.input_dim, self.layer.output_dim)


class ReversibleLookupGeometry(LookupGeometry):
    """Lookup geometry with a reverse projection (`ReversibleEmbedding`)."""

    reversible = True
