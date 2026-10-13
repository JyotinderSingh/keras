"""Shared building blocks for the built-in quantization modes.

`GeometryDispatchStrategy` is the base class of the modes that write one
handler per geometry family (int8, int4). The functions below are steps
that several modes share: the group index variable, the bias and
activation, and the weight-only forward pass of a projection.
"""

from keras.src import ops
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
    `geometry_families`.
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


def add_group_index(layer, length, initializer="zeros"):
    """Adds `g_idx`, the group of each of `length` positions along an axis.

    Stored as `float32` because TF has no GPU kernel for int32 resource
    variables (it would pin the variable to CPU and break `jit_compile` on
    GPU); consumers cast to int32 on-device. Not autocast: bfloat16 holds
    integers exactly only up to 256.
    """
    return layer.add_weight(
        name="g_idx",
        shape=(length,),
        initializer=initializer,
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


def dequantize_and_contract(layer, geometry, quantized_weight, inputs):
    """The weight-only forward pass of a projection.

    Contracts the inputs against the weight dequantized to the compute
    dtype, then adds the LoRA update, the bias and the activation. The
    input gradient is autodiff's, through the dequantized weight.
    """
    weight = quantized_weight.dequantize(layer.compute_dtype)
    x = geometry.contract(inputs, weight)
    x = geometry.add_lora_delta(inputs, x)
    return apply_bias_activation(layer, x)
