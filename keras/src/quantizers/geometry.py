"""Quantization geometry: the layer-side protocol behind the mode registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The mode descriptors in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

One geometry family exists so far: a projection, whose float kernel is
contracted against the inputs by a matmul (`Dense`). Further families are
added as their layers move onto the protocol.

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

The geometry is a thin adapter, so the mode implementations still read
state directly off the layer. Beyond what `Layer` already provides, a
quantizable layer must define:

- Projections: `_kernel` (the float kernel variable), `units`, `bias` and
  `activation` (either may be `None`), and, while LoRA is enabled,
  `lora_enabled`, `lora_kernel_a`, `lora_kernel_b`, `lora_alpha` and
  `lora_rank`. `EinsumProjectionGeometry` additionally relies on the
  equation analysis `EinsumDense` prepares in `_set_quantization_info()`.
- Lookups: `_embeddings`, `input_dim`, `output_dim`, and the
  `lora_embeddings_a` / `lora_embeddings_b` equivalents. A reversible
  lookup adds `tie_weights`, `logit_soft_cap`, and, when untied, the
  `reverse_embeddings` variables.

The rest comes from `Layer` itself: modes read `compute_dtype`,
`dtype_policy` and `path`, create their quantized variables through
`add_weight`, and re-enter through `Layer.quantized_build`, which routes
straight back to the mode. A layer never needs to know which mode is
running, and implements none of these itself.

Defining `_quantization_geometry()` on a subclass also makes that subclass
the owner of its quantization support: `Layer.quantize`'s type check
accepts instances of the exact class that defines the method. A `Dense`
subclass therefore opts in by defining it; without it the subclass is
skipped by `Model.quantize` and remains reachable through
`quantize(..., type_check=False)`.

Customizing what a mode does to a layer
---------------------------------------

Override a geometry hook rather than a mode method: the hooks on the
classes below are the only points at which mode implementations vary per
layer. `TernaryDense` is the in-tree example: its geometry supplies its
own straight-through ternarization values, and the ternary mode needs no
knowledge of the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one mode's math for itself alone, because that surface moved onto
the descriptors; replace the mode instead, by subclassing it, overriding
the one handler, and registering it under a new name. A new geometry family, on
the other hand, needs no dispatcher change at all: declare its `family`
and implement the mode's `_build_<family>`, `_call_<family>` and
`_quantize_<family>` methods.
"""

import numpy as np

from keras.src import ops


class QuantizationGeometry:
    """Base class for a layer's quantization geometry.

    A geometry names the *family* it belongs to. Mode descriptors
    implement one `_build_<family>`, `_call_<family>` and
    `_quantize_<family>` method per family they support, so introducing a
    family is a declaration plus those methods, with no dispatch chain to
    edit anywhere.
    """

    # Dispatch key for building and quantizing, and for the forward pass
    # unless `call_family` overrides it.
    family = None
    # Forward-pass dispatch key, when the forward pass needs a different
    # implementation from build/quantize (a reversible lookup does).
    call_family = None
    # Whether the layer also projects back through its weight.
    reversible = False

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

    @property
    def weight_shape(self):
        """Shape of the float weight that quantization replaces."""
        return self.layer._kernel.shape

    def prepare(self):
        """Computes any layout analysis the geometry needs (idempotent)."""

    def calibration_rows_columns(self, kernel_shape):
        """2D `(rows, columns)` view used by the calibration modes."""
        return kernel_shape[0], kernel_shape[1]

    def store_unpacked_columns(self, mode_name, columns):
        """Records the unpacked column count for the calibration call path."""
        del mode_name, columns  # The matmul case reads `layer.units` instead.

    def unpacked_columns(self, mode_name):
        """The unpacked column count recorded at calibration build time."""
        del mode_name
        return self.layer.units

    def contract(self, inputs, kernel):
        """Contracts `inputs` against a kernel in the contraction shape."""
        return ops.matmul(inputs, kernel)

    def reshape_kernel(self, kernel):
        """Restores a 2D dequantized kernel to the contraction shape."""
        return kernel

    def record_calibration_kernel_shape(self, kernel_shape):
        """Records the float kernel shape for the calibration write-back."""
        self.layer.kernel_shape = kernel_shape

    def ternary_values(self):
        """Returns `(ternary_kernel, scale)` for ternary quantization.

        The default applies the BitNet b1.58 rule to the float kernel:
        `threshold = 0.5 * mean(|W|)` and `scale = mean(|W|)`. A layer that
        owns its own ternarization rule (`TernaryDense` and its straight-
        through estimator) overrides this in its geometry.
        """
        kernel = self.layer._kernel
        kernel_np = ops.convert_to_numpy(kernel)
        abs_k = ops.convert_to_numpy(ops.abs(kernel))
        t = float(ops.convert_to_numpy(ops.mean(abs_k))) * 0.5
        kernel_ternary = np.sign(kernel_np) * (abs_k > t).astype(
            kernel_np.dtype
        )
        beta = float(np.mean(abs_k))
        return kernel_ternary, beta
