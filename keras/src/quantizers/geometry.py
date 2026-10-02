"""Quantization geometry: the layer-side protocol behind the strategy registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The strategies in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

Two geometry families exist:

- Projection: a float kernel contracted against the inputs. A strategy writes
  one projection implementation and the geometry supplies what differs per
  layer: how to contract (a plain matmul for `Dense` and `TernaryDense`,
  `ProjectionGeometry`; an einsum for `EinsumDense`,
  `EinsumProjectionGeometry`, which reads the equation's labels), the
  roles of the kernel's axes (`KernelAxes`), from which int8, int4, GPTQ
  and AWQ lay the kernel out, how an activation scale lines up with the
  outputs, and the layout of a stored per-channel scale.
- Lookup: a float embeddings table indexed by the inputs. `Embedding` is the
  plain case (`LookupGeometry`); `ReversibleEmbedding` adds a reverse
  projection (`ReversibleLookupGeometry`).

Making a layer quantizable
--------------------------

A layer, built-in or custom, becomes quantizable with the built-in modes by
the steps below. `ThirdPartyProjection` (with its subclasses `PermutedDense`
and `Pointwise1D`) and `TokenTable` in
`keras.src.quantizers.quantization_test_utils` follow them, and
`conformance_test.py` runs them through every mode they list.

1. Geometry. `_quantization_geometry()` returns the layer's geometry:

   - `ProjectionGeometry` for a 2-D `(input_dim, units)` kernel that a
     matmul contracts against the last axis of the inputs.
   - A `ProjectionGeometry` subclass for another kernel or contraction. It
     overrides `contract`, and `add_lora_delta` when the layer has LoRA.
     A kernel whose axes are not `(input_dim, units)` gives their roles in
     `kernel_axes` (`KernelAxes(contracted, free, batch)`). `contract_grad`
     is read by
     int8, and by int4 with an activation quantizer; the other modes take
     the input gradient through `contract`. `PermutedGeometry` and
     `PointwiseGeometry` in the fixture module are examples.
   - `LookupGeometry` for an embeddings table, and
     `ReversibleLookupGeometry` for a table that also projects back
     (`call(inputs, reverse=True)`).

   Two projection hooks derive from `kernel_axes` and need an override
   only in these cases. `kernel_scale_axes` (int8) is the free and batch
   axes in the kernel's order: override it when the outputs do not end
   with these axes in that order. `contraction_view` (GPTQ, AWQ) has no
   input batch axes and reads every contracted position from the last
   axis of the inputs: override it for a kernel with batch axes, or for
   inputs laid out another way. An activation quantizer (int8, and int4
   with one) reduces over `inputs_quantization_axis` (the last input
   axis) and lines its scale up with the outputs in `align_inputs_scale`
   (no change): override these when the inputs are contracted over other
   axes.

2. Spec. `variable_serialization_spec` maps `None` and each mode that the
   layer supports to the ordered names of the variables that the save and
   the load write. The modes in it are the layer's capability
   declaration: `quantize()` refuses a mode that is not in it. A mode also
   refuses a geometry family that it does not handle
   (`QuantizationStrategy.geometry_families`): int8 and int4 handle
   projections and lookups; float8, ternary, GPTQ and AWQ handle
   projections only, and ternary only a 2-D kernel. The names are those
   of `Dense` for a projection and of `Embedding` for a lookup
   (`PROJECTION_SPEC` and `LOOKUP_SPEC` in the fixture module). GPTQ and
   AWQ store no float kernel: their entries are `bias`,
   `quantized_kernel`, `kernel_scale`, `kernel_zero`, `g_idx`, and for AWQ
   `awq_scales` before `g_idx`. The order of the names is the checkpoint
   format of the mode.

3. Build. `build()` creates the weights in the order of `Dense.build`:

   - record `kernel_shape`, the float kernel's shape (a projection);
   - under a quantized policy (`self.quantization_mode` is set), call
     `self.quantized_build(shape, mode=self.quantization_mode,
     config=self.quantization_config)` with the float weight's shape,
     which creates the mode's variables;
   - create the float weight `_<name>` (`_kernel`, `_embeddings`) only
     when `self._strategy_owns_weight_storage()` is false: on a float
     layer, or under float8, which keeps the float kernel;
   - create the other float weights (`bias`, which may be `None`).

   `ThirdPartyProjection.build` instead creates the float kernel and the
   bias before it calls `quantized_build` for float8. A float8 layer
   built from its policy then lists `weights` in the order that
   `quantize()` gives, so `set_weights` works between the two. A
   projection also sets `activation`, which may be `None`. `quantize()`
   refuses a layer whose `build()` did not set the geometry's
   `build_attributes`.

4. Weight property. A public weight property (`kernel`, `embeddings`)
   reads `self._quantized_weight()`: the integer codes in the weight's
   shape (`unpack()`) when it returns a view, else the float `_<name>`.
   No mode reads the property.

5. Save and load. Two one-liners route the store through the spec:

   ```python
   def save_own_variables(self, store):
       self._save_serialized_variables(store, "kernel")

   def load_own_variables(self, store):
       self._load_serialized_variables(store, "kernel")
   ```

   A lookup passes `"embeddings"`. The save stores the entries in spec
   order under "0", "1", ... For `<name>` it reads the float weight at
   `_<name>`, or the codes of the mode's `QuantizedWeight` view.
   `quantized_<name>`, and `<name>_scale` and `<name>_zero` when the view
   has them, also come from the view. Every other entry is the layer
   attribute of that name, and an entry whose value is `None` is skipped.
   The load assigns the store back to the same variables in the same
   order.

6. Attributes. The geometry is a thin adapter, so the modes also read
   these attributes of the layer:

   - a projection: `_kernel`, `kernel_shape`, `bias` and `activation`
     (`EinsumProjectionGeometry` also reads `equation` and the
     `einsum_axes` that `EinsumDense.build` records);
   - a lookup: `_embeddings`, `input_dim` and `output_dim`;
   - a reversible lookup, in addition: `tie_weights`, `reverse_dtype`
     (the dtype of the reverse projection, or `None` for the compute
     dtype), `logit_soft_cap` (or `None`) and, when untied, the
     `reverse_embeddings` variable.

7. LoRA (optional). `Layer` defines `lora_enabled = False`. A layer with
   LoRA defines `enable_lora(rank, lora_alpha=None)`, which first calls
   `self._check_lora_supported(self.quantization_mode)` (float8 and
   ternary refuse LoRA), then creates `lora_kernel_a` (shape
   `kernel_shape[:-1] + (rank,)`) and `lora_kernel_b` (`(rank,
   kernel_shape[-1])`), and sets `lora_enabled`, `lora_rank` and
   `lora_alpha`. A lookup names its factors `lora_embeddings_a` and
   `lora_embeddings_b`. The save merges the update
   `lora_alpha / lora_rank * lora_<name>_a @ lora_<name>_b` into the
   stored weight, and the load sets the factors to zero.

8. Config (optional). `get_config` serializes `quantization_config` and
   `from_config` deserializes it, so that a `.keras` round trip keeps a
   quantizer that is not the mode's default.

The rest comes from `Layer` itself: the modes and the calibration run read
`compute_dtype`, `variable_dtype`, the layer's own policy
(`_own_dtype_policy`), `path` and `name`, create their variables through
`add_weight`, and re-enter through
`Layer.quantized_build`, which routes back to the strategy. The layer holds
no code for one mode: its build, weight property, save and load are the
same for every mode.

Defining `_quantization_geometry()` on a subclass also makes that subclass
the owner of its quantization support: `Layer.quantize`'s type check
accepts instances of the exact class that defines the method. `quantize()`
refuses an instance of a subclass that does not define it, and its error
names the two ways out: `quantize(..., type_check=False)`, or
`_quantization_geometry()` on the subclass. A `Dense` subclass therefore
opts in by defining it, for example to return `ProjectionGeometry(self)`;
without it, `Model.quantize` skips the subclass and reports it.

Customizing what a strategy does to a layer
-------------------------------------------

Override a geometry hook rather than a strategy method. A strategy varies
per layer through the hooks on the classes below and through the
attributes of step 6 above; AWQ also matches the layer's `name` against
`AWQConfig.clip_skip_patterns`. `TernaryDense` is the in-tree example of a
hook: its geometry supplies its own straight-through ternarization values,
and the ternary strategy needs no knowledge of the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one strategy's math for itself alone, because that surface lives
on the strategy; a layer that contracts its kernel differently overrides
the geometry hooks, and anything beyond that is a change to the mode itself.
A layer also cannot bring a new geometry family: each mode lists the
families it handles in `geometry_families`, and int8 and int4 implement one
`_<verb>_<family>` handler per verb and family (`GeometryDispatchStrategy`,
`keras.src.quantizers.modes.common`), so a new family is a change to the
built-in modes.
"""

import dataclasses
import math
import string

from keras.src import ops
from keras.src.quantizers.quantized_weight import lay_out_scale
from keras.src.quantizers.quantizers import ternarize


@dataclasses.dataclass(frozen=True)
class KernelAxes:
    """The roles of a projection kernel's axes.

    *Contracted* axes are summed against the inputs and *free* axes reach
    the output from the kernel alone. A *batch* axis is shared by the
    inputs, the kernel and the output (the experts of a
    mixture-of-experts down projection): each of its indices is an
    independent problem with its own slice of the inputs.

    The modes lay the kernel out from this record. With `B`, `K` and `N`
    the sizes of the batch, contracted and free axes:

    - int8 keeps the N-D kernel and reduces its scale over `contracted`.
    - int4 stores `matrix(shape, batch_in="columns")`, `(K, B * N)`: every
      column belongs to one problem, which keeps the released bytes of a
      kernel whose contracted axes lead.
    - GPTQ and AWQ store `matrix(shape, batch_in="rows")`, `(B * K, N)`:
      act-order `g_idx` and AWQ's input scales differ per problem and lie
      along the rows.

    Without a batch axis the two matrices are the same.
    """

    contracted: tuple
    free: tuple
    batch: tuple = ()

    def __post_init__(self):
        for name in ("contracted", "free", "batch"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def sizes(self, kernel_shape):
        """`(B, K, N)`: the sizes of the batch, contracted and free axes."""
        return tuple(
            math.prod(kernel_shape[i] for i in axes)
            for axes in (self.batch, self.contracted, self.free)
        )

    def matrix(self, kernel_shape, batch_in):
        """`(permutation, rows, columns)` of the kernel as a 2D matrix.

        The kernel transposed by `permutation` and reshaped to
        `(rows, columns)` has its contracted axes along the rows. The
        batch axes lead the rows (`batch_in="rows"`) or keep their place
        among the free axes in the columns (`batch_in="columns"`).
        """
        problems, rows, columns = self.sizes(kernel_shape)
        if batch_in == "rows":
            permutation = self.batch + self.contracted + self.free
            return permutation, problems * rows, columns
        if batch_in == "columns":
            rest = tuple(sorted(self.batch + self.free))
            return self.contracted + rest, rows, problems * columns
        raise ValueError(
            "`batch_in` must be 'rows' or 'columns'. "
            f"Received: batch_in={batch_in!r}"
        )


class ContractionView:
    """A projection kernel and its inputs as independent problems.

    The calibration modes solve one `(rows, columns)` matrix per batch
    index of the kernel (see `KernelAxes`). `kernel_to_view` lays the
    kernel out as `(batch, rows, columns)`, the kernel transposed by
    `kernel_permutation` and reshaped, and `inputs_to_view` lays the
    inputs out as `(batch, samples, rows)`, or `(samples, rows)` when
    `batch` is 1, so a row index means the same contracted position on
    both sides.

    Args:
        kernel_shape: The kernel's own shape.
        kernel_axes: The kernel's `KernelAxes`.
        input_batch_axes: Input axes matching `kernel_axes.batch`, in the
            same order.
        input_contracted_axes: Input axes matching
            `kernel_axes.contracted`, in the same order.
    """

    def __init__(
        self,
        kernel_shape,
        kernel_axes,
        *,
        input_batch_axes,
        input_contracted_axes,
    ):
        self.kernel_shape = tuple(int(d) for d in kernel_shape)
        self.kernel_permutation, _, _ = kernel_axes.matrix(
            self.kernel_shape, batch_in="rows"
        )
        if sorted(self.kernel_permutation) != list(
            range(len(self.kernel_shape))
        ):
            raise ValueError(
                "The batch, contracted and free axes must partition the "
                f"kernel axes. Received: kernel_shape={self.kernel_shape}, "
                f"kernel_axes={kernel_axes}"
            )
        self.batch, self.rows, self.columns = kernel_axes.sizes(
            self.kernel_shape
        )
        self.input_batch_axes = tuple(input_batch_axes)
        self.input_contracted_axes = tuple(input_contracted_axes)

    def kernel_to_view(self, kernel):
        """Lays the kernel out as `(batch, rows, columns)`."""
        if self.kernel_permutation != tuple(range(len(self.kernel_shape))):
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
    per verb for each family it lists in `geometry_families`, so
    introducing a family is a declaration plus those handlers, with no
    dispatch chain to edit anywhere.
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


# Projection hooks that no mode reads, and the hook that describes the
# same fact. A geometry that defines one of them is refused when its class
# is created: the override would have no effect.
_REPLACED_PROJECTION_HOOKS = {
    "kernel_matrix": "kernel_axes",
    "kernel_reduced_axes": "kernel_axes",
    "kernel_scale_shape": "kernel_scale_axes",
    "kernel_scale_axis": "kernel_scale_axes",
    "kernel_scale_for_storage": "kernel_scale_axes",
    "kernel_scale_for_dequant": "kernel_scale_axes",
}


class ProjectionGeometry(QuantizationGeometry):
    """Geometry of a 2D kernel `(input_dim, units)` contracted by matmul.

    A kernel of another layout overrides `kernel_axes`, plus `contract`,
    `contract_grad` and `add_lora_delta` for its own contraction
    (`contract_grad` only for int8, and int4 with an activation
    quantizer). The defaults of `kernel_scale_axes` and
    `contraction_view` derive from `kernel_axes`; their docstrings say
    when a layer overrides them too.
    """

    family = "projection"
    build_attributes = ("kernel_shape", "bias", "activation")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        replaced = sorted(set(vars(cls)) & set(_REPLACED_PROJECTION_HOOKS))
        if replaced:
            raise TypeError(
                f"`{cls.__name__}` defines geometry hooks that no "
                "quantization mode reads. Replace "
                + ", ".join(
                    f"`{name}` with `{_REPLACED_PROJECTION_HOOKS[name]}`"
                    for name in replaced
                )
                + "."
            )

    @property
    def weight_shape(self):
        """Shape of the float kernel, as the layer recorded it in `build()`.

        Quantized storage may be packed or flattened, so this is the
        logical shape every strategy reads rather than a variable's shape.
        """
        return tuple(self.layer.kernel_shape)

    @property
    def kernel_axes(self):
        """The roles of the kernel's axes, as a `KernelAxes`."""
        return KernelAxes(contracted=(0,), free=(1,))

    def contraction_view(self):
        """The `ContractionView` GPTQ and AWQ calibrate on.

        The default reads every contracted position from the last axis of
        the inputs. A kernel with batch axes, or inputs laid out another
        way, overrides it.
        """
        return ContractionView(
            self.weight_shape,
            self.kernel_axes,
            input_batch_axes=(),
            input_contracted_axes=(-1,),
        )

    def contract(self, inputs, kernel):
        """Contracts `inputs` against a kernel in the contraction shape."""
        return ops.matmul(inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        """Gradient of `contract` with respect to its inputs.

        The custom gradients of int8, and of int4 with an activation
        quantizer, read it. The other modes differentiate `contract`.
        """
        return ops.matmul(upstream, ops.transpose(float_kernel))

    @property
    def inputs_quantization_axis(self):
        """Input axes an activation quantizer reduces over."""
        return -1

    def align_inputs_scale(self, scale):
        """Aligns an activation scale with the contraction's outputs."""
        return scale

    @property
    def kernel_scale_axes(self):
        """Layout of a per-channel scale stored with the kernel (int8).

        For each axis of the stored scale, the kernel axis it runs along,
        or `None` for an axis of size one. int8 divides the contraction's
        outputs by the stored scale, so the scale follows the outputs'
        trailing axes. The default is the free and batch axes in the
        kernel's order: a layer whose outputs end with them in another
        order overrides it.
        """
        axes = self.kernel_axes
        return tuple(sorted(axes.free + axes.batch))

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

    Every hook derives from the labels of the equation, which
    `EinsumDense` records in `build()` (`einsum_axes`).
    """

    build_attributes = ("kernel_shape", "einsum_axes", "bias", "activation")

    @property
    def kernel_axes(self):
        return self.layer.einsum_axes.kernel_axes

    def contraction_view(self):
        axes = self.layer.einsum_axes
        kernel_axes = axes.kernel_axes
        input_contracted_axes = axes.input_axes(kernel_axes.contracted)
        # The equation analysis already refuses a kernel axis absent from
        # both the inputs and the output; what is left to check is that the
        # kernel has something to contract and something to output, and
        # that the inputs are not summed over an axis on their own.
        if (
            not kernel_axes.contracted
            or not kernel_axes.free
            or sorted(input_contracted_axes) != sorted(axes.input_reduced_axes)
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
            kernel_axes,
            input_batch_axes=axes.input_axes(kernel_axes.batch),
            input_contracted_axes=input_contracted_axes,
        )

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        # From https://stackoverflow.com/a/47609896
        return ops.einsum(
            self.layer.einsum_axes.gradient_equation, upstream, float_kernel
        )

    @property
    def inputs_quantization_axis(self):
        return self.layer.einsum_axes.input_reduced_axes

    def align_inputs_scale(self, scale):
        axes = self.layer.einsum_axes
        return lay_out_scale(scale, axes.inputs, axes.output)

    @property
    def kernel_scale_axes(self):
        # In the outputs' layout: the kernel's free and batch axes in the
        # order of the output, and an axis of size one for every output
        # axis the kernel does not have.
        axes = self.layer.einsum_axes
        return tuple(
            axes.kernel.index(label) if label in axes.kernel else None
            for label in axes.output
        )

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
