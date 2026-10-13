"""Handlers for the lookup family (`Embedding`, `ReversibleEmbedding`).

One implementation serves every mode that stores an embeddings table as
integer codes. A mode says what the codes mean (`_get_lookup_scheme`),
how they pack (`_get_lookup_layout`) and how a float table becomes codes
(`_encode_lookup`). The variables, both forward passes, the views and the
conversion follow from those.
"""

import math

from keras.src import backend
from keras.src import ops
from keras.src.quantizers.modes.common import add_group_index
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantizers import dequantize_with_sz_map


class LookupHandlers:
    """The lookup handlers of a `GeometryDispatchStrategy`.

    The table is stored `(input_dim, packed(output_dim))`. The scale runs
    per row, or per row and group of columns for a grouped scheme (with a
    zero point and a group index). An untied reversible layer stores its
    reverse table as that layout transposed.
    """

    # --- What a mode supplies ---------------------------------------------

    def _get_lookup_scheme(self, layer, config):
        """Returns the `WeightScheme` of the layer's tables."""
        raise NotImplementedError

    def _get_lookup_layout(self, axis, length):
        """Returns the `PackLayout` of `length` codes along `axis`."""
        return NoPack()

    def _default_reverse_inputs_quantizer(self):
        """Returns the reverse projection's default activation quantizer."""
        return None

    def _encode_lookup(self, layer, geometry, weight, config):
        """Returns `(codes, scale, zero_point)` of a float forward table."""
        raise NotImplementedError

    # --- Variables --------------------------------------------------------

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        input_dim, output_dim = embeddings_shape
        scheme = self._get_lookup_scheme(layer, config)
        packed_output_dim = self._get_lookup_layout(
            -1, output_dim
        ).packed_length(output_dim)
        # The scale reduces `output_dim` (typically much smaller than
        # `input_dim`): one entry per row, or per row and group.
        scale_shape = (input_dim,)
        if scheme.group_size is not None:
            scale_shape += (math.ceil(output_dim / scheme.group_size),)

        def add_table(name, codes_shape, scale_shape):
            codes = layer.add_weight(
                name=name,
                shape=codes_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            scale = layer.add_weight(
                name=f"{name}_scale",
                shape=scale_shape,
                initializer="ones",
                trainable=False,
            )
            zero_point = None
            if scheme.has_zero_point:
                zero_point = layer.add_weight(
                    name=f"{name}_zero",
                    shape=scale_shape,
                    initializer="zeros",
                    dtype="int8",
                    trainable=False,
                )
            return codes, scale, zero_point

        layer._embeddings, layer.embeddings_scale, layer.embeddings_zero = (
            add_table("embeddings", (input_dim, packed_output_dim), scale_shape)
        )
        layer.g_idx = None
        if scheme.group_size is not None:
            layer.g_idx = add_group_index(
                layer,
                output_dim,
                lambda shape, dtype: ops.floor_divide(
                    ops.arange(output_dim, dtype=dtype), scheme.group_size
                ),
            )
        if geometry.reversible:
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(
                    config, self._default_reverse_inputs_quantizer()
                )
            )
            if not layer.tie_weights:
                # The forward layout transposed: packed along `output_dim`
                # (axis 0), parameters per column or per group and column.
                (
                    layer.reverse_embeddings,
                    layer.reverse_embeddings_scale,
                    layer.reverse_embeddings_zero,
                ) = add_table(
                    "reverse_embeddings",
                    (packed_output_dim, input_dim),
                    tuple(reversed(scale_shape)),
                )

    # --- Views ------------------------------------------------------------

    def _get_lookup_quantized_weight(self, layer, geometry):
        return QuantizedWeight(
            codes=layer._embeddings,
            scale=layer.embeddings_scale,
            zero_point=layer.embeddings_zero,
            g_idx=layer.g_idx,
            layout=self._get_lookup_layout(-1, layer.output_dim),
            scheme=self._get_lookup_scheme(layer, layer.quantization_config),
            shape=(layer.input_dim, layer.output_dim),
            # A grouped scale runs per row and group of columns, a
            # per-channel scale per row.
            axis=None if layer.g_idx is None else -1,
            scale_axes=(0,) if layer.g_idx is None else None,
        )

    def _get_reverse_lookup_quantized_weight(self, layer, geometry):
        if layer.tie_weights:
            # A tied layer's reverse table is its forward table transposed.
            codes = ops.transpose(layer._embeddings)
            scale = layer.embeddings_scale
            zero_point = layer.embeddings_zero
            if zero_point is not None:
                # Grouped parameters are 2-D, `(input_dim, n_groups)`.
                scale = ops.transpose(scale)
                zero_point = ops.transpose(zero_point)
        else:
            codes = layer.reverse_embeddings
            scale = layer.reverse_embeddings_scale
            zero_point = layer.reverse_embeddings_zero
        return QuantizedWeight(
            codes=codes,
            scale=scale,
            zero_point=zero_point,
            g_idx=layer.g_idx,
            layout=self._get_lookup_layout(0, layer.output_dim),
            scheme=self._get_lookup_scheme(layer, layer.quantization_config),
            shape=(layer.output_dim, layer.input_dim),
            axis=None if layer.g_idx is None else 0,
            scale_axes=(1,) if layer.g_idx is None else None,
        )

    # --- Forward passes ---------------------------------------------------

    def _call_lookup(self, layer, geometry, inputs, reverse=False):
        if reverse:
            return self._reverse_lookup(layer, geometry, inputs)
        if backend.standardize_dtype(inputs.dtype) not in ("int32", "int64"):
            inputs = ops.cast(inputs, "int32")
        table = self._get_lookup_quantized_weight(layer, geometry)
        # Gather the stored rows first, then unpack only those. The table
        # is frozen, so no custom gradient is needed.
        codes = table.layout.unpack(ops.take(table.codes, inputs, axis=0))
        codes = ops.cast(codes, layer.compute_dtype)
        scale = ops.take(table.scale, inputs, axis=0)
        if table.g_idx is None:
            outputs = ops.divide(codes, ops.expand_dims(scale, axis=-1))
        else:
            # `scale` and `zero_point` are `[batch..., n_groups]`; `g_idx`
            # expands them along `output_dim`.
            zero_point = ops.take(table.zero_point, inputs, axis=0)
            outputs = dequantize_with_sz_map(
                codes, scale, zero_point, table.g_idx, group_axis=-1
            )
        if layer.lora_enabled:
            lora_outputs = ops.take(layer.lora_embeddings_a, inputs, axis=0)
            lora_outputs = ops.matmul(lora_outputs, layer.lora_embeddings_b)
            outputs = ops.add(
                outputs, (layer.lora_alpha / layer.lora_rank) * lora_outputs
            )
            outputs = ops.cast(outputs, dtype=layer.compute_dtype)
        return outputs

    def _reverse_lookup(self, layer, geometry, inputs):
        # As the float layer: `reverse_dtype` when set, else `compute_dtype`.
        dtype = layer.reverse_dtype or layer.compute_dtype
        inputs = ops.cast(inputs, dtype)
        table = self._get_reverse_lookup_quantized_weight(layer, geometry)
        codes = table.unpack()
        if layer.inputs_quantizer:
            inputs_q, inputs_scale = layer.inputs_quantizer(inputs)
        else:
            inputs_q, inputs_scale = inputs, ops.ones((1,), dtype=dtype)
        if table.g_idx is None:
            # Symmetric: matmul on the codes, then fold both scales into
            # the logits.
            logits = ops.cast(ops.matmul(inputs_q, codes), dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, table.scale))
        else:
            # A zero point cannot be pulled out of the matmul, so the table
            # is dequantized first. Not `table.dequantize(dtype)`: that
            # rounds the table to `dtype`, and here it keeps the precision
            # of the scale.
            float_table = dequantize_with_sz_map(
                ops.cast(codes, dtype),
                table.scale,
                table.zero_point,
                table.g_idx,
                group_axis=0,
            )
            logits = ops.divide(ops.matmul(inputs_q, float_table), inputs_scale)
        # The scales are float32 variables; the projection reports its own
        # dtype, as the float layer does.
        logits = ops.cast(logits, dtype)
        if layer.tie_weights and layer.lora_enabled:
            # Only a tied layer projects back through the adapted table. The
            # delta is taken from the float inputs.
            lora_logits = ops.matmul(
                inputs, ops.transpose(layer.lora_embeddings_b)
            )
            lora_logits = ops.matmul(
                lora_logits, ops.transpose(layer.lora_embeddings_a)
            )
            logits = ops.add(
                logits,
                ops.cast(
                    (layer.lora_alpha / layer.lora_rank) * lora_logits,
                    logits.dtype,
                ),
            )
        if layer.logit_soft_cap is not None:
            soft_cap = layer.logit_soft_cap
            logits = ops.multiply(
                ops.tanh(ops.divide(logits, soft_cap)), soft_cap
            )
        return logits

    # --- Conversion -------------------------------------------------------

    def _quantize_lookup(self, layer, geometry, config):
        encoded = [
            self._encode_lookup(layer, geometry, layer._embeddings, config)
        ]
        del layer._embeddings
        if geometry.reversible and not layer.tie_weights:
            # The reverse table is the forward layout transposed, so its
            # transpose is encoded by the forward rule (including a
            # user-supplied `weight_quantizer`) and transposed back.
            codes, scale, zero_point = self._encode_lookup(
                layer, geometry, ops.transpose(layer.reverse_embeddings), config
            )
            encoded.append(
                (
                    ops.transpose(codes),
                    ops.transpose(scale),
                    None if zero_point is None else ops.transpose(zero_point),
                )
            )
            del layer.reverse_embeddings
        layer.quantized_build(geometry.weight_shape, self.name, config)
        # The views of an untied layer hold the variables themselves.
        for table, (codes, scale, zero_point) in zip(
            self.quantized_weights(layer), encoded
        ):
            table.codes.assign(codes)
            table.scale.assign(scale)
            if zero_point is not None:
                table.zero_point.assign(zero_point)
