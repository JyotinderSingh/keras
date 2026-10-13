"""Shared building blocks for the built-in quantization modes.

`GeometryDispatchStrategy` is the base class of the modes that write one
handler per geometry family (int8, int4). The functions below are steps
that several modes share: the group index variable, the bias and
activation, the custom input gradient of a projection, which int8 and
int4 take against the dequantized kernel, and `contract_quantized`, the
one entry point through which a mode lets a route of `kernels.py` run
its contraction.
"""

from keras.src import ops
from keras.src.quantizers import kernels
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class GeometryDispatchStrategy(QuantizationStrategy):
    """A mode whose math is written once per geometry family.

    Each verb resolves the layer's geometry and calls the mode's handler
    for that verb and the geometry's family. For each family it
    supports, a mode implements:

    - `_build_<family>`, `_call_<family>` and `_quantize_<family>`: the
      variables, the forward pass and the conversion.
    - `_get_<family>_quantized_weight`: the `QuantizedWeight` view, or
      `None` when the mode holds no integer codes. The weight property,
      saving, `enable_lora` and `Model.quantization_summary` read it.
    - `_encode_<family>`, when the view is not `None`: the default
      `merge_lora_delta` re-quantizes the merged weight with it.
    - `_get_reverse_<family>_quantized_weight`, for a reversible family
      whose layer is untied and has a view: the reverse table's view.

    A mode implements every handler of each family it lists in
    `geometry_families`. `register_quantization_strategy` refuses a mode
    that lacks a build, call, quantize or view handler, the reverse view
    handler of a reversible family, or, when it supports LoRA and keeps
    the default `merge_lora_delta`, an encode handler. A subclass that
    overrides a verb itself needs no handler for it.
    """

    # The handler that each verb calls, by geometry family.
    _handler_names = {
        "build": "_build_{family}",
        "call": "_call_{family}",
        "quantize": "_quantize_{family}",
        "encode": "_encode_{family}",
        "quantized_weight": "_get_{family}_quantized_weight",
        "reverse_quantized_weight": "_get_reverse_{family}_quantized_weight",
    }

    # The families with a reversible geometry (`ReversibleLookupGeometry`),
    # whose untied layer holds a second table with its own view.
    _reversible_families = ("lookup",)

    def _get_missing_definitions(self):
        missing = super()._get_missing_definitions()
        verbs = ("build", "call", "quantize", "quantized_weight", "encode")
        for verb in verbs + ("reverse_quantized_weight",):
            if verb == "encode" and (
                not self.supports_lora
                or type(self).merge_lora_delta
                is not QuantizationStrategy.merge_lora_delta
            ):
                continue
            # A subclass that overrides the verb itself needs no handler.
            method = "quantized_weights" if verb.startswith("reverse") else verb
            if getattr(type(self), method) is not getattr(
                GeometryDispatchStrategy, method
            ):
                continue
            for family in self.geometry_families:
                if (
                    verb.startswith("reverse")
                    and family not in self._reversible_families
                ):
                    continue
                handler = self._handler_names[verb].format(family=family)
                if getattr(self, handler, None) is None:
                    missing.append(handler)
        return missing

    def build(self, layer, input_shape, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("build", geometry.family, layer)
        handler(layer, geometry, input_shape, config)

    def call(self, layer, *args, **kwargs):
        geometry = self.require_geometry(layer)
        handler = self._handler("call", geometry.family, layer)
        return handler(layer, geometry, *args, **kwargs)

    def quantize(self, layer, config):
        geometry = self.require_geometry(layer)
        handler = self._handler("quantize", geometry.family, layer)
        handler(layer, geometry, config)

    def encode(self, layer, weight, config=None):
        geometry = self.require_geometry(layer)
        handler = self._handler("encode", geometry.family, layer)
        return handler(layer, geometry, weight, config)

    def quantized_weight(self, layer):
        geometry = self.require_geometry(layer)
        handler = self._handler("quantized_weight", geometry.family, layer)
        return handler(layer, geometry)

    def quantized_weights(self, layer):
        views = super().quantized_weights(layer)
        geometry = self.require_geometry(layer)
        if views and geometry.reversible and not layer.tie_weights:
            # An untied reversible lookup holds a second table.
            handler = self._handler(
                "reverse_quantized_weight", geometry.family, layer
            )
            views += (handler(layer, geometry),)
        return views

    def _handler(self, verb, family, layer):
        """Returns this mode's implementation of `verb` for one family."""
        name = self._handler_names[verb].format(family=family)
        handler = getattr(self, name, None)
        if handler is None:
            raise NotImplementedError(
                f"Quantization mode '{self.name}' does not implement "
                f"`{name}` for the '{family}' quantization geometry of "
                f"layer {layer.__class__.__name__}."
            )
        return handler


def add_group_index(layer, length, group_size=None):
    """Adds `g_idx`, the group of each of `length` positions along an axis.

    With `group_size`, position `i` is in group `i // group_size`. Without
    it, the index starts as zeros and a calibration writes it.

    Stored as `float32` because TF has no GPU kernel for int32 resource
    variables (it would pin the variable to CPU and break `jit_compile` on
    GPU); consumers cast to int32 on-device. Not autocast: bfloat16 holds
    integers exactly only up to 256.
    """

    def contiguous_groups(shape, dtype):
        return ops.floor_divide(ops.arange(length, dtype=dtype), group_size)

    return layer.add_weight(
        name="g_idx",
        shape=(length,),
        initializer="zeros" if group_size is None else contiguous_groups,
        dtype="float32",
        trainable=False,
        autocast=False,
    )


def apply_bias_activation(layer, x):
    """Adds the layer's bias and applies its activation, when present."""
    if layer.bias is not None:
        x = ops.add(x, layer.bias)
    if layer.activation is not None:
        x = layer.activation(x)
    return x


def contract_with_float_kernel_gradient(layer, geometry, view, inputs, forward):
    """Runs `forward(inputs, quantized_weight)` with a custom input gradient.

    `view` is the layer's `QuantizedWeight`, and `forward` contracts the
    inputs against the weight it is given. The gradient with respect to
    the inputs is `geometry.contract_grad` against the kernel dequantized
    to the compute dtype, broadcast to the shape of the inputs. It passes
    straight through a rounding of the inputs, and the backward pass
    dequantizes the kernel again, so only the stored tensors are kept for
    it.
    """

    @ops.custom_gradient
    def contract(inputs, *tensors):
        quantized_weight = view.with_tensors(tensors)

        def grad_fn(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            float_kernel = quantized_weight.dequantize(layer.compute_dtype)
            inputs_grad = ops.broadcast_to(
                geometry.contract_grad(upstream, float_kernel),
                ops.shape(inputs),
            )
            return (inputs_grad,) + (None,) * len(tensors)

        return forward(inputs, quantized_weight), grad_fn

    # Read inside the autocast scope: on TensorFlow eager the gradient
    # runs after it, and the scale variable would then read float32.
    return contract(inputs, *view.read_tensors())


def contract_quantized(layer, geometry, view, inputs, reference):
    """Contracts `inputs` against the weight that `view` holds.

    `reference(inputs, quantized_weight)` is the mode's own contraction.
    When `kernels.select_route` selects a route, the route runs forward
    only, inside `contract_with_float_kernel_gradient`. Otherwise
    `reference(inputs, view)` runs as given, with the gradient the mode
    gives it. A route contracts float inputs: a mode with an activation
    quantizer runs its own path and does not call this.
    """
    selection = kernels.select_route(layer, geometry, view, inputs)
    if selection is None:
        return reference(inputs, view)
    return contract_with_float_kernel_gradient(
        layer,
        geometry,
        view,
        inputs,
        lambda inputs, quantized_weight: kernels.run_route(
            selection, layer, geometry, quantized_weight, inputs
        ),
    )
