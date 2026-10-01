"""Shared chassis for the calibration-based quantization modes.

GPTQ and AWQ allocate the same family of variables, run the same
dequantize-and-contract forward pass, and speak the same three-part policy
grammar; they differ only in the code bit-width (which fixes how the
kernel is packed), in one extra AWQ variable and its inverse scaling, in
a handful of message fragments and, for the calibration run, in the
calibrator class. Those differences are the hooks below.

A LoRA update trains against the dequantized weight, as it does for int8
and int4: the forward pass adds it to the contraction, the calibrators
quantize the base kernel so the update stays a separate term, and a
merged save rounds the merged weight onto the calibrated grid
(`merge_lora_delta`).
"""

import math
import warnings

from keras.src import ops
from keras.src.quantizers.modes.common import add_group_index
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantized_weight import Int2Quads
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.strategy_registry import QuantizationStrategy

# Fraction of the codes a LoRA-merged save may clip to the calibrated range
# before it warns.
LORA_MERGE_CLIP_WARNING_FRACTION = 0.01


class CalibrationStrategy(QuantizationStrategy):
    """A post-training strategy whose values arrive from a calibration pass."""

    geometry_families = ("projection",)
    requires_layer_structure = True

    def quantize(self, layer, config):
        # The quantized values arrive later, so this only allocates the
        # mode's variables from the layer's current weight shape.
        geometry = self.require_geometry(layer)
        layer.quantized_build(geometry.weight_shape, self.name, config)
        # A live float layer keeps its float kernel as the weight until
        # `write_back` installs the calibrated codes.
        layer.calibration_pending = True

    # --- Config and policy-string surface ---------------------------------

    # The mode's dedicated dtype policy class.
    policy_cls = None

    def policy_from_string(self, mode_str, source_name):
        return self.policy_cls(mode_str, source_name)

    def config_from_policy(self, policy):
        name = self.name.upper()
        raise ValueError(
            f"Implicitly enabling {name} quantization by setting "
            f"`dtype_policy` to '{policy.name}' is not supported. "
            f"{name} requires a calibration dataset and a config object "
            f"(`{self.config_cls.__name__}`).\n\n"
            f"Please use the `.quantize('{self.name}', config=...)` method "
            "on the layer or model instead."
        )

    def default_config(self):
        # The calibration dataset comes only from an explicit config.
        raise ValueError(
            f"For {self.name.upper()}, the `config` argument must be of "
            f"type `{self.config_cls.__name__}`."
        )

    def validate_config(self, config):
        if not isinstance(config, self.config_cls):
            raise ValueError(
                f"Mode '{self.name}' requires a valid `config` argument "
                f"of type `{self.config_cls.__name__}`. "
                f"Received: {type(config)}"
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
        policy = layer._own_dtype_policy
        if policy.quantization_mode == self.name:
            return getattr(policy, attr)
        raise ValueError(
            f"For {self.name.upper()} quantization, the {attr} must be "
            "specified either through a `dtype_policy` of type "
            f"`{self.policy_cls.__name__}` or the `config` argument. "
            f"Received: dtype_policy={policy!r}"
        )

    # --- Calibration run --------------------------------------------------

    # The `Calibrator` class that solves this mode for one layer.
    calibrator_cls = None

    def calibrate(self, config, structure, filters=None):
        """Runs this mode's calibration over `structure` and writes back.

        Args:
            config: The mode's config, with its dataset and tokenizer.
            structure: Dict with keys `"pre_block_layers"` and
                `"sequential_blocks"`, as `Model.quantize` resolved it.
            filters: Optional filters that exclude layers from quantization.
        """
        # Imported here to avoid an import cycle: `calibration_run` imports
        # the layers, which import the strategies.
        from keras.src.quantizers.calibration_run import CalibrationRun
        from keras.src.quantizers.calibration_run import (
            calibration_no_grad_scope,
        )
        from keras.src.quantizers.calibration_run import get_dataloader

        if config.dataset is None or config.tokenizer is None:
            raise ValueError(
                f"{self.name.upper()} quantization requires a dataset and a "
                "tokenizer. Please provide them in the "
                f"`{self.config_cls.__name__}`."
            )
        dataloader = get_dataloader(
            config.tokenizer,
            config.sequence_length,
            config.dataset,
            num_samples=config.num_samples,
        )
        with calibration_no_grad_scope():
            CalibrationRun(self, config, structure, filters).run(dataloader)

    def finalize_model_quantization(self, model, config, structure, filters):
        del model
        self.calibrate(config, structure, filters)

    # --- Variables --------------------------------------------------------

    def build(self, layer, input_shape, config):
        """Allocates the quantized kernel and quantization parameters.

        The variables hold uninitialized values until the calibration pass
        (run by `Model.quantize`) writes the quantized weights back.
        """
        geometry = self.require_geometry(layer)
        # Allocation alone leaves nothing pending: a layer built under a
        # calibration policy loads its codes from a checkpoint. `quantize`
        # marks a live float layer pending after this returns.
        layer.calibration_pending = False

        view = geometry.contraction_view()
        rows = view.batch * view.rows
        columns = view.columns

        bits = self.resolve_weight_bits(layer, config)
        kernel_columns = self._pack_layout(bits, columns).packed_length(columns)
        group_size = self.resolve_group_size(layer, config)
        n_groups = view.batch * (
            1 if group_size == -1 else math.ceil(view.rows / group_size)
        )

        # Stored as the view's `(batch * rows, columns)` matrix in `[in,
        # out]` orientation (the view's axis order, which may permute the
        # kernel's), packed along the output axis like the int4 layout, so
        # the forward pass unpacks and dequantizes without a transpose of
        # its own. The problems of a batch axis stack along the rows, each
        # with its own groups.
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
        layer.g_idx = add_group_index(layer, rows)

    def _build_extra_variables(self, layer, rows):
        """Creates any mode-specific variables, after the zero point."""

    def _input_scales(self, layer):
        """Per-input-row scales divided out of the dequantized kernel."""
        del layer
        return None

    # --- Calibration state ------------------------------------------------

    def write_back(self, layer, codes, scale, zero_point, g_idx, **extra):
        """Installs the calibrated values and retires the float kernel.

        `codes` are the unpacked codes in the kernel's `[in, out]`
        orientation; they are packed here as `build` laid out the variable.
        """
        bits = self.resolve_weight_bits(layer, layer.quantization_config)
        codes = ops.cast(codes, layer.quantized_kernel.dtype)
        codes = self._pack_layout(bits, codes.shape[-1]).pack(codes)
        del layer._kernel
        layer.quantized_kernel.assign(codes)
        layer.kernel_scale.assign(scale)
        layer.kernel_zero.assign(zero_point)
        layer.g_idx.assign(g_idx)
        self._assign_extra_variables(layer, **extra)
        layer.calibration_pending = False

    def _assign_extra_variables(self, layer, **extra):
        """Assigns any mode-specific calibrated values."""
        del layer
        if extra:
            raise TypeError(
                f"Quantization mode '{self.name}' has no extra calibrated "
                f"variables. Received: {sorted(extra)}"
            )

    def check_saveable(self, layer):
        if layer.calibration_pending:
            raise ValueError(
                f"Cannot save layer '{layer.name}' because it is quantized "
                f"with mode '{self.name}' but has never been calibrated. Its "
                "quantized weights are uninitialized, so saving would "
                "produce a corrupted model. Run calibration first, e.g. via "
                "`model.quantize(...)` with a quantization layer structure "
                "that covers this layer, or exclude the layer from "
                "quantization with `filters`."
            )

    def unstored_variables(self, layer):
        # A layer pending calibration still holds its float kernel.
        return (layer._kernel,) if layer.calibration_pending else ()

    def variables_loaded(self, layer):
        # A stored calibration checkpoint is always calibrated: loading
        # completes the transition and retires the float kernel a live
        # `quantize()` left in place.
        if layer.calibration_pending:
            del layer._kernel
            layer.calibration_pending = False

    # --- LoRA merge -------------------------------------------------------

    def merge_lora_delta(self, layer, delta):
        """Rounds the merged weight onto the calibrated grid.

        The scale, zero point, group index and AWQ's `awq_scales` are the
        calibration's result and stay as they are. Only the codes change:
        the merged weight is rounded to the nearest code under those
        parameters, in `float32`, so a zero delta keeps every code. A
        weight the delta pushes past its group's range clips to it, and a
        merge that clips more than `LORA_MERGE_CLIP_WARNING_FRACTION` of
        the codes warns.
        """
        quantized_weight = self.quantized_weight(layer)
        merged = ops.add(
            quantized_weight.dequantize("float32"), ops.cast(delta, "float32")
        )
        codes = ops.round(quantized_weight.code_image(merged))
        low, high = quantized_weight.scheme.code_range
        outside = ops.logical_or(ops.less(codes, low), ops.greater(codes, high))
        clipped = float(
            ops.convert_to_numpy(ops.mean(ops.cast(outside, "float32")))
        )
        if clipped > LORA_MERGE_CLIP_WARNING_FRACTION:
            warnings.warn(
                f"Merging the LoRA update into layer '{layer.name}' clipped "
                f"{clipped:.1%} of its {self.name.upper()} codes to the "
                "range its calibrated scale and zero point cover. The saved "
                "weights keep the calibration but lose that part of the "
                "update; for a large update, apply the update to the float "
                "model and calibrate it again.",
                stacklevel=2,
            )
        return (
            quantized_weight.pack_image(codes),
            quantized_weight.scale,
            quantized_weight.zero_point,
        )

    @staticmethod
    def _pack_layout(bits, columns):
        """How `columns` codes of `bits` bits pack along the output axis."""
        if bits == 4:
            return Int4Pairs(axis=-1, orig_len=columns)
        if bits == 2:
            return Int2Quads(axis=-1, orig_len=columns)
        # 3-bit codes are not packed densely (3 does not divide 8) and
        # 8-bit codes need no packing: one code per byte.
        return NoPack()

    # --- Quantized weight view --------------------------------------------

    def quantized_weight(self, layer):
        if layer.calibration_pending:
            # The codes are uninitialized; the float kernel is still the
            # layer's weight.
            return None
        geometry = self.require_geometry(layer)
        config = layer.quantization_config
        bits = self.resolve_weight_bits(layer, config)
        group_size = self.resolve_group_size(layer, config)
        view = geometry.contraction_view()
        return QuantizedWeight(
            codes=layer.quantized_kernel,
            scale=layer.kernel_scale,
            zero_point=layer.kernel_zero,
            g_idx=layer.g_idx,
            layout=self._pack_layout(bits, view.columns),
            scheme=WeightScheme(
                code_range=(0, 2**bits - 1),
                scale_form="multiplier",
                has_zero_point=True,
                # `-1` means one group spanning every input row of a
                # problem.
                group_size=view.rows if group_size == -1 else group_size,
            ),
            shape=geometry.weight_shape,
            axis=0,
            permutation=view.kernel_permutation,
            input_scales=self._input_scales(layer),
        )

    # --- Forward pass -----------------------------------------------------

    def call(self, layer, inputs, training=False):
        geometry = self.require_geometry(layer)
        quantized_weight = self.quantized_weight(layer)
        W = (
            layer._kernel
            if quantized_weight is None
            else quantized_weight.dequantize(layer.compute_dtype)
        )
        y = geometry.contract(inputs, W)
        y = geometry.add_lora_delta(inputs, y)
        return apply_bias_activation(layer, y)
