"""Registry of quantization strategies, one per mode.

This module is the single dispatch point for quantization behavior. Each
quantization mode (`"int8"`, `"int4"`, `"float8"`, `"ternary"`,
`"gptq"`, `"awq"`) is implemented by one `QuantizationStrategy` that
owns:

- the mode's config class and default-config resolution,
- the policy-string codec: routing a `"int4/128"`-style string to its
  dtype-policy class, and naming policies after quantization (each policy
  class parses its own grammar),
- the mode's math: the `build`/`call`/`quantize` methods create the mode's
  variables, run its forward pass, and compute its quantized values against
  the layer's quantization geometry (`keras.src.quantizers.geometry`),
- per-layer hyperparameter resolution (block size, weight bits, group size),
- model-level orchestration hooks (calibration for structure-aware modes).

The registry is internal API (`keras.src.quantizers`), and its set of
modes is closed: `keras.src.quantizers.modes` registers the built-in
modes, and Keras supports no other modes. A policy string names a mode
only through the grammar `<mode>[/<params>]_from_<source>`, so a mode
name never captures a plain dtype policy name such as `"int8"`.

Layers stay open. A layer, built-in or custom, opts in to the built-in
modes through two declarations: `Layer._quantization_geometry()` returns
its quantizable structure, and `variable_serialization_spec` lists the
modes it supports. The strategies build the mode's variables, run its
forward pass and compute its quantized values through the geometry; the
layer implements none of them. `keras.src.quantizers.geometry` ("Making a
layer quantizable") lists what such a layer defines.

`keras.src.dtype_policies` imports this module on first use, because the
mode modules import the policy classes. `keras.src.layers.layer` imports
this module at module level, so this module must not import layers.
"""

_MODE_TO_STRATEGY = {}  # mode -> QuantizationStrategy, in registration order.


class QuantizationStrategy:
    """Implementation of one quantization mode.

    Subclasses set `name` and `config_cls` and implement the
    `build`/`call`/`quantize` adapters against the layer's quantization
    geometry (`Layer._quantization_geometry()`), which describes the layer's
    quantizable structure without the layer knowing about any mode.
    """

    # The mode identifier, e.g. `"int8"`. Also the root of the policy-string
    # grammar (`"int8_from_float32"`, `"int4/128_from_float32"`).
    name = None

    # The `QuantizationConfig` subclass for this mode, or `None` if the mode
    # constructs its default config another way.
    config_cls = None

    # Whether `quantize(mode)` without a config is an error (calibration
    # modes need datasets that only an explicit config can carry).
    requires_config = False

    # Whether `build` creates the layer's weight storage itself, replacing
    # the float weight. Modes that keep the float weight and only add
    # auxiliary variables (float8) set this to False, so the layer's `build`
    # still creates the float weight.
    owns_weight_storage = True

    # Whether `Model.quantize` must resolve a quantization layer structure
    # (pre-block layers + sequential blocks) before mutating any layer.
    requires_layer_structure = False

    # Storage byte multiplier used by `Model.quantization_summary` (packed
    # sub-byte formats store two values per byte).
    summary_byte_multiplier = 1

    # --- Config resolution ------------------------------------------------

    def default_config(self):
        """Returns the config used when `quantize(mode)` is called bare."""
        if self.requires_config:
            raise ValueError(self._missing_config_error())
        return self.config_cls()

    def _missing_config_error(self):
        return (
            f"For {self.name.upper()}, you must pass a config object in the "
            "`config` argument."
        )

    def validate_config(self, config):
        """Validates a user-provided config object for this mode."""
        if (
            self.requires_config
            and self.config_cls is not None
            and not isinstance(config, self.config_cls)
        ):
            raise ValueError(
                f"Mode '{self.name}' requires a valid `config` argument "
                f"of type `{self.config_cls.__name__}`. "
                f"Received: {type(config)}"
            )

    def config_from_policy(self, policy):
        """Builds the config equivalent to a quantized dtype policy.

        Used by the `Layer.dtype_policy` setter to forward the policy's full
        parameters into `quantize()`. Returns `None` when the mode's config
        cannot be derived from a bare policy (the subsequent `quantize` call
        then raises the mode's missing-config error). A mode may instead
        raise to refuse policy-triggered quantization outright.
        """
        del policy
        if self.requires_config:
            return None
        return self.default_config()

    # --- Per-layer hyperparameter resolution ------------------------------

    # Mode-specific `resolve_*` helpers live on the concrete strategies
    # (e.g. `Int4Strategy.resolve_block_size`). They all share the precedence:
    # explicit config > layer's quantized dtype policy > DTypePolicyMap
    # entry > mode-specific fallback.

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
        that does not define one (a layer still on its own per-mode
        methods, or a custom layer that a registered mode claims through
        `supports_layer`) is refused here with a clear error rather than
        failing deeper inside the mode.

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
        return geometry

    def supports_layer(self, layer):
        """Whether this mode claims support for `layer`.

        Layers primarily declare support by listing the mode in their
        `variable_serialization_spec`; this hook lets an externally
        registered mode claim layers it can quantize generically without
        the layer having to know about it.
        """
        del layer
        return False

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

        A mode whose values arrive later (from calibration, or from
        training) instead just builds its variables here.
        """
        raise NotImplementedError(
            f"Quantization mode '{self.name}' does not implement `quantize`."
        )

    # --- Model-level orchestration ----------------------------------------

    def finalize_model_quantization(self, model, config, structure, filters):
        """Hook run by `Model.quantize` after the per-layer walk.

        Structure-aware modes run their calibration pass here.
        """
        del model, config, structure, filters


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
    and must not contain the policy-grammar separators ("/" and "_from_").
    """
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
    has_config_source = (
        instance.config_cls is not None
        or instance.requires_config
        or type(instance).default_config
        is not QuantizationStrategy.default_config
    )
    if not has_config_source:
        raise ValueError(
            f"Quantization strategy '{name}' must define `config_cls`, set "
            "`requires_config = True`, or override `default_config()`."
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
