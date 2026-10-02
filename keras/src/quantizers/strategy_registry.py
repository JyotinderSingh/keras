"""Registry of quantization strategies, one per mode, and how they work.

A layer's dtype policy names its quantization mode: `"int4/128_from_float32"`
is mode `"int4"` with a block size of 128. The registry maps each mode to one
`QuantizationStrategy`. `Layer.quantize`, `Layer.quantized_build` and
`Layer.quantized_call` look up the strategy of the layer's mode and delegate
to it. The strategy reads the layer through its quantization geometry
(`Layer._quantization_geometry()`, `keras.src.quantizers.geometry`), and int8
and int4 then call their handler for the geometry's family
(`GeometryDispatchStrategy` in `keras.src.quantizers.modes.common`). A mode
that stores integer codes reads them back through a `QuantizedWeight`
(`keras.src.quantizers.quantized_weight`). The registry is internal API
(`keras.src.quantizers`), and its set of modes is closed:
`keras.src.quantizers.modes` registers the built-in modes, and Keras supports
no other modes. Layers stay open: a layer, built-in or custom, opts in to the
built-in modes through `_quantization_geometry()` and
`variable_serialization_spec`. The spec lists the modes the layer supports
and, for each mode, the variables it stores in checkpoint order.
`keras.src.quantizers.geometry` ("Making a layer quantizable") lists what such
a layer defines.

A strategy owns:

- the mode's config class and default-config resolution,
- the policy-string codec: routing a `"int4/128"`-style string to its
  dtype-policy class, and naming policies after quantization (each policy
  class parses its own grammar),
- the mode's math: the `build`/`call`/`quantize` methods create the mode's
  variables, run its forward pass, and compute its quantized values against
  the layer's quantization geometry (`keras.src.quantizers.geometry`; the
  calibrators of `keras.src.quantizers.calibrator` compute the values of
  GPTQ and AWQ); `encode` turns a float weight into the stored form, and
  `quantized_weight` reads the stored variables back through a
  `QuantizedWeight` (`keras.src.quantizers.quantized_weight`),
- per-layer hyperparameter resolution (block size, weight bits, group size),
- the model-level run (`model_run`: the calibration run of GPTQ and AWQ).

The built-in modes, the geometry families they handle and the built-in
layers whose spec lists them:

    mode     families            layers
    int8     projection, lookup  Dense, EinsumDense, Embedding,
                                 ReversibleEmbedding
    int4     projection, lookup  Dense, EinsumDense, Embedding,
                                 ReversibleEmbedding
    float8   projection          Dense, EinsumDense
    ternary  projection (2-D)    Dense, TernaryDense
    gptq     projection          Dense, EinsumDense
    awq      projection          Dense, EinsumDense

The variables each mode stores, by the names of the `Dense` and `Embedding`
specs. `K`, `N` and `B` are the sizes of a kernel's contracted, free and
batch axes (`KernelAxes`). A projection also stores its `bias` in every
mode.

    int8     projection: kernel (int8 codes in the kernel's shape) and
             kernel_scale (per output channel, in the outputs' layout).
             lookup: embeddings (int8 codes) and embeddings_scale (per
             row).
    int4     projection: kernel (the `(K, B * N)` matrix, two codes per
             byte) and kernel_scale (per column; grouped: per group of
             rows and column, with kernel_zero and g_idx).
             lookup: embeddings (two codes per byte) and embeddings_scale
             (per row; grouped: per row and group of columns, with
             embeddings_zero and g_idx).
    float8   projection: kernel (the float kernel stays), and a scale and
             an amax history each for the inputs, the kernel and the
             output gradient.
    ternary  projection: kernel (five codes per byte) and kernel_scale
             (one scalar).
    gptq     projection: quantized_kernel (the `(B * K, N)` matrix, packed
             by bit width), and kernel_scale, kernel_zero (per group of
             rows and column) and g_idx. No float kernel.
    awq      projection: as gptq, and awq_scales (one per row).

An untied `ReversibleEmbedding` also stores its reverse table:
`reverse_embeddings`, `reverse_embeddings_scale` and, for grouped int4,
`reverse_embeddings_zero`. int8, int4, gptq and awq add a LoRA update as a
separate term of the forward pass and merge it into the stored weight on
save; float8 and ternary refuse LoRA (`supports_lora`).

A layer is unbuilt, float, or quantized in one mode:

    unbuilt  --build()--------------->  float, or quantized when its
                                        policy names a mode
    float    --Layer.quantize(mode)-->  quantized (int8, int4, float8,
                                        ternary)
    float    --Model.quantize(mode)-->  quantized (every mode)

- `Layer.quantize` checks the layer before it changes it: built, not
  quantized, a compute dtype the mode accepts, a geometry, the type check,
  the spec, the geometry's `build_attributes`, the strategy's
  `check_quantizable` and LoRA support. Then `strategy.quantize` replaces
  the float weight with the mode's variables through `quantized_build`,
  which sets `_is_quantized`, and the policy becomes
  `<mode...>_from_<source>`. A quantization that raises leaves the layer
  as it was. `Operation._dispatch_call` runs `quantized_call` for a layer
  whose policy names a mode.
- A quantized policy set on a built float layer runs the same `quantize()`
  with the config that the strategy derives from the policy. On a
  quantized layer the setter accepts only a new source dtype: no
  transition leads to another mode or back to float.
- GPTQ and AWQ quantize only through `Model.quantize`, in one step;
  `Layer.quantize` and the policy setter refuse them. `model_run` creates
  a `CalibrationRun` before any layer changes. Each layer that the run
  covers stays float until its calibrator has observed the layer's
  inputs, and the calibrator's `write_back` then swaps the float kernel
  for the calibrated variables. No layer stays between float and
  quantized. A calibrated layer holds `quantization_config = None`: its
  policy name gives the bit width and the group size.
- A layer built under a quantized policy (a saved model that loads)
  creates the mode's variables in `quantized_build`, and no float weight
  when the strategy owns the weight storage (every mode but float8).

`keras.src.dtype_policies.dtype_policy` imports this module on first use: the
mode modules import the policy classes, so a module-level import there is a
cycle.
"""

from keras.src import ops

_MODE_TO_STRATEGY = {}  # mode -> QuantizationStrategy, in registration order.


class QuantizationStrategy:
    """Implementation of one quantization mode.

    Subclasses set `name`, `config_cls` and `geometry_families` and
    implement `build`, `call` and `quantize` against the layer's
    quantization geometry (`Layer._quantization_geometry()`), which
    describes the layer's quantizable structure without the layer knowing
    about any mode. A mode that stores integer codes also implements
    `quantized_weight`; a mode that supports a LoRA-merged save also
    implements `encode`, or overrides `merge_lora_delta`.
    """

    # The mode identifier, e.g. `"int8"`. Also the root of the policy-string
    # grammar (`"int8_from_float32"`, `"int4/128_from_float32"`).
    name = None

    # The `QuantizationConfig` subclass for this mode.
    config_cls = None

    # Whether `build` creates the layer's weight storage itself, replacing
    # the float weight. Modes that keep the float weight and only add
    # auxiliary variables (float8) set this to False, so the layer's `build`
    # still creates the float weight.
    owns_weight_storage = True

    # Whether a layer quantized with this mode can use LoRA. `enable_lora`
    # and `Layer.quantize` check it, so a mode that sets it to False refuses
    # LoRA in either order, before the layer changes.
    supports_lora = True

    # The geometry families (`QuantizationGeometry.family`) this mode's math
    # handles. A layer of another family is refused before it changes.
    geometry_families = ()

    # --- Config resolution ------------------------------------------------

    def default_config(self):
        """Returns the config used when `quantize(mode)` is called bare.

        A mode that needs an explicit config (the calibration modes need a
        dataset) raises `ValueError` instead.
        """
        return self.config_cls()

    def validate_config(self, config):
        """Validates a user-provided config object for this mode."""
        del config

    def config_from_policy(self, policy):
        """Builds the config equivalent to a quantized dtype policy.

        Used by the `Layer.dtype_policy` setter to forward the policy's full
        parameters into `quantize()`. A mode may instead raise to refuse
        policy-triggered quantization outright.
        """
        del policy
        return self.default_config()

    # --- Per-layer hyperparameter resolution ------------------------------

    # Mode-specific `resolve_*` helpers live on the concrete strategies and
    # read the layer's own policy (`Layer._own_dtype_policy`, the map entry
    # of a layer that holds a `DTypePolicyMap`).
    # `Int4Strategy.resolve_block_size` reads the config, then the policy,
    # then falls back to per-channel. The calibration modes read the
    # policy, then the config of the run (`resolve_weight_bits`,
    # `resolve_group_size`).

    # --- Policy-string codec ----------------------------------------------

    def policy_from_string(self, mode_str, source_name):
        """Builds the dtype policy for a `<mode>_from_<source>` string.

        The default is the generic `QuantizedDTypePolicy`; a mode with a
        dedicated policy class (`Int4DTypePolicy`, `GPTQDTypePolicy`, ...)
        overrides this to build it.
        """
        from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy

        return QuantizedDTypePolicy(mode_str, source_name)

    def policy_suffix(self, layer, config):
        """The mode fragment used to name the policy after quantization.

        E.g. `"int8"`, `"int4/128"`, `"gptq/4/128"`; `quantize()` appends
        `_from_<source>` to it.
        """
        del layer, config
        return self.name

    # --- Layer capability -------------------------------------------------

    def require_geometry(self, layer):
        """Returns `layer`'s quantization geometry, raising if it has none.

        The built-in strategies read the layer through its geometry, so a layer
        that does not define one, or whose geometry family the mode does
        not handle (`geometry_families`), is refused here with a clear
        error rather than failing deeper inside the mode.

        Args:
            layer: The layer being quantized.

        Returns:
            The layer's `QuantizationGeometry`.
        """
        geometry = layer._quantization_geometry()
        if geometry is None:
            raise NotImplementedError(
                f"Layer {layer.__class__.__name__} does not define a "
                f"quantization geometry, so mode '{self.name}' cannot be "
                "applied to it."
            )
        if geometry.family not in self.geometry_families:
            raise NotImplementedError(
                f"Quantization mode '{self.name}' does not support the "
                f"'{geometry.family}' quantization geometry of layer "
                f"{layer.__class__.__name__}."
            )
        return geometry

    def check_quantizable(self, layer):
        """Raises `NotImplementedError` if this mode cannot quantize `layer`.

        `Layer.quantize` calls it before the layer changes, so a refused
        layer stays as it was. The default requires a quantization
        geometry.

        Args:
            layer: The layer about to be quantized.
        """
        self.require_geometry(layer)

    # --- Mode math --------------------------------------------------------

    def build(self, layer, input_shape, config):
        """Creates the mode's variables on `layer`."""
        raise NotImplementedError(
            f"Quantization mode '{self.name}' does not implement `build`."
        )

    def call(self, layer, *args, **kwargs):
        """Runs the mode's forward pass on `layer`."""
        raise NotImplementedError(
            f"Quantization mode '{self.name}' does not implement `call`."
        )

    def quantize(self, layer, config):
        """Computes quantized values and swaps `layer`'s variables.

        A mode whose values arrive later from training (float8) only
        builds its variables here. Either way it builds them through
        `layer.quantized_build(shape, self.name, config)`, which also marks
        the layer quantized. The calibration modes raise: their values
        come from the run of `Model.quantize`.
        """
        raise NotImplementedError(
            f"Quantization mode '{self.name}' does not implement `quantize`."
        )

    def encode(self, layer, weight, config=None):
        """Quantizes a float `weight` into this mode's stored form.

        Returns `(codes, scale, zero_point)` exactly as the mode's variables
        hold them (packed and oriented for storage), ready to assign.
        The default `merge_lora_delta` uses it to re-quantize a merged
        weight.
        `zero_point` is `None` for a symmetric scheme.
        """
        raise NotImplementedError(
            f"Quantization mode '{self.name}' does not implement `encode`."
        )

    def merge_lora_delta(self, layer, delta):
        """Folds a LoRA update into `layer`'s stored weight, for saving.

        `delta` is the scaled LoRA update in the weight's shape. Returns
        `(codes, scale, zero_point)` as `encode` does. The default
        dequantizes the stored weight in the layer's variable dtype, adds
        the delta and encodes the sum with a fresh scale.
        """
        merged = ops.add(
            self.quantized_weight(layer).dequantize(layer.variable_dtype),
            delta,
        )
        return self.encode(layer, merged, layer.quantization_config)

    # --- Quantized weight view --------------------------------------------

    def quantized_weight(self, layer):
        """Returns the `QuantizedWeight` view of `layer`'s weight, or `None`.

        `None` means the mode holds no integer codes for the layer: it keeps
        the float weight (float8).
        """
        del layer
        return None

    def quantized_weights(self, layer):
        """All `QuantizedWeight` views of `layer`'s quantized weights.

        Most layers hold one; an untied `ReversibleEmbedding` also holds its
        reverse table. Empty when the mode holds no integer codes.
        """
        quantized_weight = self.quantized_weight(layer)
        return () if quantized_weight is None else (quantized_weight,)

    # --- Model-level orchestration ----------------------------------------

    def model_run(self, model, config):
        """The run that quantizes `model`'s layers together, or `None`.

        `Model.quantize` calls it before it changes any layer. With `None`,
        the walk quantizes each layer on its own (`Layer.quantize`). A mode
        whose values come from calibration data returns its run, which
        refuses here what it can refuse. The walk then asks the run whether
        it `covers` each layer and hands it the layers to quantize (`add`).
        `run()` quantizes them, and `quantized` lists the layers that
        changed, also when `run()` raises.
        """
        del model, config
        return None


def register_quantization_strategy(strategy):
    """Registers a `QuantizationStrategy`.

    Accepts an instance or a class (instantiated with no arguments), and
    returns its argument unchanged so it can also be used as a class
    decorator. Registration order is observable (validation errors render
    the registered names), so the built-in modes register explicitly in
    `keras.src.quantizers.modes` instead, where the order is written down
    rather than left to the import order.

    The strategy is validated at registration time, not at first use:
    the name must be a non-empty string, must not be registered already,
    must not contain the policy-grammar separators ("/" and "_from_"),
    must not shadow a standard dtype or mixed-precision policy name, and
    must not share a prefix with a built-in mode (built-in names are
    routed by `str.startswith` over policy strings; externally registered
    modes match only their exact grammar, so collisions between them are
    unambiguous).
    """
    from keras.src import backend
    from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES

    instance = strategy() if isinstance(strategy, type) else strategy
    name = instance.name
    if not isinstance(name, str) or not name:
        raise ValueError(
            "A quantization strategy must define a non-empty string `name`. "
            f"Received: name={name!r}"
        )
    if name in _MODE_TO_STRATEGY:
        raise ValueError(
            f"A quantization mode named '{name}' is already registered."
        )
    if "/" in name or "_from_" in name:
        raise ValueError(
            f"Cannot register quantization mode '{name}': its name must "
            "not contain '/' or '_from_', which are the policy-string "
            "grammar separators."
        )
    if name not in QUANTIZATION_MODES:
        try:
            backend.standardize_dtype(name)
            is_standard_dtype = True
        except ValueError:
            is_standard_dtype = False
        if is_standard_dtype or name.startswith("mixed_"):
            raise ValueError(
                f"Cannot register quantization mode '{name}': its name "
                "conflicts with a standard dtype or mixed-precision "
                "policy name."
            )
    for existing in _MODE_TO_STRATEGY:
        builtin_involved = (
            existing in QUANTIZATION_MODES or name in QUANTIZATION_MODES
        )
        if builtin_involved and (
            existing.startswith(name) or name.startswith(existing)
        ):
            raise ValueError(
                f"Cannot register quantization mode '{name}': its name "
                f"collides with registered mode '{existing}'. Built-in "
                "mode names are routed by prefix over policy strings, so "
                "no mode name may share a prefix with a built-in mode."
            )
    _MODE_TO_STRATEGY[name] = instance
    # Return the argument, not the strategy: as a class decorator this
    # must leave the class bound to its name, still subclassable.
    return strategy


def unregister_quantization_strategy(mode):
    """Removes the strategy registered for `mode` (intended for tests)."""
    _MODE_TO_STRATEGY.pop(mode, None)


def get_strategy(mode):
    """Returns the strategy registered for `mode`, or `None`."""
    return _MODE_TO_STRATEGY.get(mode)


def is_registered(mode):
    """Whether a strategy is registered for `mode`."""
    return mode in _MODE_TO_STRATEGY


def registered_modes():
    """All registered modes, as a tuple, in registration order."""
    return tuple(_MODE_TO_STRATEGY)
