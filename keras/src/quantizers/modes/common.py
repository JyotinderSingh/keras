"""Shared building blocks for the built-in quantization modes.

`GeometryDispatchStrategy` is the base class of the modes that write one
handler per geometry family (int8, int4). The functions below are steps
that several modes share: the group index variable and its write check,
the bias and activation, and the custom input gradient of a projection,
which int8 and int4 take against the dequantized kernel.
"""

import dataclasses

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.quantizers.strategy_registry import QuantizationStrategy
from keras.src.utils.module_utils import tensorflow as tf

if backend.backend() == "jax":
    import jax
elif backend.backend() == "torch":
    import torch
    from torch.distributed.tensor import DTensor


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

    With `group_size`, position `i` is in group `i // group_size`, and the
    mode fixes these groups: the variable holds `ContiguousGroups` as its
    write check (`_assign_check`), which refuses other groups. Without
    it, the index starts as zeros and a calibration writes it.

    Stored as `float32` because TF has no GPU kernel for int32 resource
    variables (it would pin the variable to CPU and break `jit_compile` on
    GPU); consumers cast to int32 on-device. Not autocast: bfloat16 holds
    integers exactly only up to 256.
    """

    def contiguous_groups(shape, dtype):
        return ops.floor_divide(ops.arange(length, dtype=dtype), group_size)

    variable = layer.add_weight(
        name="g_idx",
        shape=(length,),
        initializer="zeros" if group_size is None else contiguous_groups,
        dtype="float32",
        trainable=False,
        autocast=False,
    )
    if group_size is not None:
        variable._assign_check = ContiguousGroups(length, group_size)
    return variable


@dataclasses.dataclass(frozen=True)
class ContiguousGroups:
    """Refuses a group index other than `arange(length) // group_size`.

    The write check of a `g_idx` whose groups the mode fixes, stored on the
    variable as `_assign_check`. `Layer.set_weights` and
    `Model.set_state_tree` call it with the value of each write, before the
    write. The Keras-format and the legacy h5 loads call the check of each
    variable of a layer before its first write. It raises `ValueError` to
    refuse the value. A value that the host cannot read now (a tracer, a
    symbolic tensor, a JAX array on other processes) passes unchecked. The
    check reads a torch `DTensor` whole (`_host_value`). A value of another
    shape passes too, and the write raises its shape error. The check is a
    frozen dataclass, so it compares and hashes by value and copies with
    its variable.
    """

    length: int
    group_size: int

    def __call__(self, value):
        stored = _host_value(value)
        if stored is None or stored.shape != (self.length,):
            return
        groups = np.arange(self.length) // self.group_size
        if not np.array_equal(stored, groups):
            raise ValueError(
                "This `g_idx` must put position `k` in group "
                f"`k // {self.group_size}`, the groups that its quantization "
                "mode fixes. Received: g_idx starting "
                f"{stored[:8].tolist()} for {self.length} positions"
            )


def _host_value(value):
    """`value` as a NumPy array, or `None` if the host cannot read it now.

    The host cannot read a JAX tracer, a JAX array on other processes, a
    symbolic TensorFlow tensor, or a value under `torch.compile`. A torch
    `DTensor` holds only the shard of this process, so the check gathers
    the whole value (`full_tensor()`). Every process runs the same write,
    so each process joins the gather. A replicated `DTensor` needs no
    communication.
    """
    if backend.backend() == "jax":
        if isinstance(value, jax.core.Tracer) or not getattr(
            value, "is_fully_addressable", True
        ):
            return None
    elif backend.backend() == "torch":
        if torch.compiler.is_compiling():
            return None
        if isinstance(value, DTensor):
            value = value.full_tensor()
    elif backend.backend() == "tensorflow":
        if not tf.executing_eagerly():
            return None
    return np.asarray(ops.convert_to_numpy(value))


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
