"""Shared chassis for the calibration-based quantization modes.

GPTQ and AWQ allocate the same family of variables, run the same
dequantize-and-contract forward pass, and speak the same three-part policy
grammar; they differ only in the code bit-width (which fixes how the
kernel is packed), in one extra AWQ variable and its inverse scaling, and
in a handful of message fragments. Those differences are the hooks below.
"""

import math

from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.mode_registry import require_geometry
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.qtensor import Int2Quads
from keras.src.quantizers.qtensor import Int4Pairs
from keras.src.quantizers.qtensor import NoPack
from keras.src.quantizers.qtensor import QTensor
from keras.src.quantizers.qtensor import WeightScheme


class CalibrationMode(QuantizationMode):
    """A post-training mode whose values arrive from a calibration pass."""

    requires_config = True
    requires_layer_structure = True

    def quantize(self, layer, config):
        # The quantized values arrive later, so this only allocates the
        # mode's variables from the layer's current weight shape.
        geometry = require_geometry(layer, self.name)
        layer.quantized_build(geometry.weight_shape, self.name, config)

    # --- Config and policy-string surface ---------------------------------

    def _missing_config_error(self):
        return (
            f"For {self.name.upper()}, the `config` argument must be of "
            f"type `{self.config_cls.__name__}`."
        )

    def policy_suffix(self, layer, config):
        del layer
        return config.dtype_policy_string()

    def resolve_group_size(self, layer, config):
        """Determine the group size from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "group_size")

    def resolve_weight_bits(self, layer, config):
        """Determine the weight bits from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "weight_bits")

    def _resolve_from_config_or_policy(self, layer, config, attr):
        """Resolves a hyperparameter with config-over-policy precedence.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the value comes from the mode's dtype policy.
        """
        if isinstance(config, self.config_cls):
            return getattr(config, attr)
        policy = layer.dtype_policy
        if isinstance(policy, DTypePolicyMap):
            policy = policy[layer.path]
            if policy.quantization_mode != self.name:
                self._on_policy_map_mismatch(policy)
        if policy.quantization_mode == self.name:
            return getattr(policy, attr)
        raise ValueError(self._resolution_error(attr))

    def _on_policy_map_mismatch(self, policy):
        """Hook for modes that reject a mismatched `DTypePolicyMap` entry.

        Returning lets resolution fall through to `_resolution_error`.
        """

    def _resolution_error(self, attr):
        """The error raised when a hyperparameter cannot be resolved."""
        raise NotImplementedError

    # --- Variables --------------------------------------------------------

    def build(self, layer, input_shape, config):
        """Allocates the quantized kernel and quantization parameters.

        The variables hold uninitialized values until the calibration pass
        (run by `Model.quantize`) writes the quantized weights back.
        """
        geometry = require_geometry(layer, self.name)

        # Ensures the forward pass uses the original high-precision kernel
        # until calibration has been performed.
        setattr(layer, f"is_{self.name}_calibrated", False)
        geometry.record_calibration_kernel_shape(input_shape)

        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"{self.name.upper()} quantization only supports 2D or 3D "
                "kernels."
            )
        rows, columns = geometry.calibration_rows_columns(input_shape)

        bits = self.resolve_weight_bits(layer, config)
        kernel_columns = self._pack_layout(bits, columns).packed_length(columns)
        group_size = self.resolve_group_size(layer, config)
        n_groups = 1 if group_size == -1 else math.ceil(rows / group_size)

        geometry.prepare()

        # Stored in the kernel's own `[in, out]` orientation and packed
        # along the output axis, like the int4 layout, so the forward pass
        # unpacks and dequantizes without a transpose.
        layer.quantized_kernel = layer.add_weight(
            name="kernel",
            shape=(rows, kernel_columns),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=(n_groups, columns),
            initializer="ones",
            trainable=False,
        )
        layer.kernel_zero = layer.add_weight(
            name="zero_point",
            shape=(n_groups, columns),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        self._build_extra_variables(layer, rows)
        # `g_idx` is stored as `float32` because TF has no GPU kernel for
        # int32 resource variables (would pin the variable to CPU and break
        # jit_compile on GPU); consumers cast to int32 on-device.
        layer.g_idx = layer.add_weight(
            name="g_idx",
            shape=(rows,),
            initializer="zeros",
            dtype="float32",
            trainable=False,
        )

    def _build_extra_variables(self, layer, rows):
        """Creates any mode-specific variables, after the zero point."""

    def _input_scales(self, layer):
        """Per-input-row scales divided out of the dequantized kernel."""
        del layer
        return None

    @staticmethod
    def _pack_layout(bits, columns):
        """How `columns` codes of `bits` bits pack along the output axis."""
        if bits == 4:
            return Int4Pairs(axis=-1, orig_len=columns, dtype="uint8")
        if bits == 2:
            return Int2Quads(axis=-1, orig_len=columns, dtype="uint8")
        # 3-bit codes are not packed densely (3 does not divide 8) and
        # 8-bit codes need no packing: one code per byte.
        return NoPack()

    # --- Quantized weight view --------------------------------------------

    def qtensor(self, layer):
        if not getattr(layer, f"is_{self.name}_calibrated", False):
            # Before calibration the codes are uninitialized and the float
            # kernel is still the layer's weight.
            return None
        geometry = require_geometry(layer, self.name)
        config = layer.quantization_config
        bits = self.resolve_weight_bits(layer, config)
        group_size = self.resolve_group_size(layer, config)
        # The group parameters are stored as `[n_groups, out]`, so their
        # axes give the unpacked column count the packed codes stand for
        # and, with the group index, the row count.
        columns = int(layer.kernel_scale.shape[1])
        rows = int(layer.g_idx.shape[0])
        return QTensor(
            codes=layer.quantized_kernel,
            scale=layer.kernel_scale,
            zero_point=layer.kernel_zero,
            g_idx=layer.g_idx,
            layout=self._pack_layout(bits, columns),
            scheme=WeightScheme(
                bits=bits,
                code_range=(0, 2**bits - 1),
                zero_point_dtype="uint8",
                # `-1` means one group spanning every input row.
                group_size=rows if group_size == -1 else group_size,
                group_axis=0,
            ),
            input_scales=self._input_scales(layer),
            logical_shape=geometry.calibration_kernel_shape(),
            compute_dtype=layer.compute_dtype,
        )

    # --- Forward pass -----------------------------------------------------

    def call(self, layer, inputs, training=False):
        geometry = require_geometry(layer, self.name)
        qtensor = self.qtensor(layer)
        W = layer._kernel if qtensor is None else qtensor.dequantize()
        y = geometry.contract(inputs, W)
        return apply_bias_activation(layer, y)
