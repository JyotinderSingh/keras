"""Handlers for the lookup family (`Embedding`, `ReversibleEmbedding`).

One implementation serves int8 and int4, the modes that store an
embeddings table as integer codes. A mode says what the codes mean
(`_get_lookup_scheme`), how they pack (`_get_lookup_layout`) and which
activation quantizer the reverse projection uses by default
(`_get_default_reverse_inputs_quantizer`). The encoding, the variables,
the views and the conversion follow from those, and both forward passes
read the table through the views.
"""

import math

from keras.src import backend
from keras.src import ops
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import kernels
from keras.src.quantizers.modes.common import add_group_index
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantized_weight import NoPack
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)


class LookupHandlers:
    """The lookup handlers of a `GeometryDispatchStrategy`.

    The table is stored `(input_dim, packed(output_dim))`, in one of two
    schemes: an ungrouped divisor scale per row, without a zero point, or
    a grouped multiplier scale per row and group of columns, with a zero
    point and a group index. The default encoding writes these two, and
    the build refuses any other scheme. An untied reversible layer stores
    its reverse table as that layout transposed.

    Both forward passes read the table through its `QuantizedWeight`
    views, and the scheme chooses the arithmetic. The lookup gathers the
    stored rows and dequantizes only those (`dequantize_rows`), in the
    precision of the scale. The reverse projection contracts with the
    codes and applies the scales to the logits (`apply_scale`), or, when
    the scheme has a zero point, contracts with the table dequantized in
    the precision of its scale. A route of `kernels.py` can replace the
    int4 reverse contraction with no activation quantizer; it runs inside
    a custom gradient that repeats the reference's.
    """

    # --- What a mode supplies ---------------------------------------------

    def _get_lookup_scheme(self, layer, config):
        """Returns the `WeightScheme` of the layer's tables."""
        raise NotImplementedError

    def _get_lookup_layout(self, axis, length):
        """Returns the `PackLayout` of `length` codes along `axis`."""
        return NoPack()

    def _get_default_reverse_inputs_quantizer(self):
        """Returns the reverse projection's default activation quantizer."""
        return None

    # --- Variables --------------------------------------------------------

    def _build_lookup(self, layer, geometry, embeddings_shape, config):
        input_dim, output_dim = embeddings_shape
        scheme = self._get_lookup_scheme(layer, config)
        # A grouped `WeightScheme` is always a multiplier with a zero point.
        if scheme.group_size is None and (
            scheme.scale_form != "divisor" or scheme.has_zero_point
        ):
            raise NotImplementedError(
                f"Mode '{self.name}' gives the embeddings table a scheme "
                "that `LookupHandlers` does not implement. It implements an "
                "ungrouped divisor scale without a zero point, or a grouped "
                f"multiplier scale with a zero point. Received: {scheme}"
            )
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
            layer.g_idx = add_group_index(layer, output_dim, scheme.group_size)
        if geometry.reversible:
            layer.inputs_quantizer = (
                QuantizationConfig.activation_quantizer_or_default(
                    config, self._get_default_reverse_inputs_quantizer()
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
            # The groups run along `output_dim`: column `d` is in group
            # `d // G`, so the view takes no `g_idx`.
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
            # Its scale is read through its `value`, so its dtype is the
            # one it reads in.
            codes = ops.transpose(layer._embeddings)
            scale = layer.embeddings_scale.value
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
            # The groups run along `output_dim`: row `d` is in group
            # `d // G`, so the view takes no `g_idx`.
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
        if standardize_dtype(inputs.dtype) not in ("int32", "int64"):
            inputs = ops.cast(inputs, "int32")
        table = self._get_lookup_quantized_weight(layer, geometry)
        # The tensors are read first: inside a model under a mixed policy
        # an `Embedding` does not autocast, so its scale reads in its own
        # dtype (float32 by default), while the variable reports
        # `floatx()`. The view gathers the stored rows and dequantizes only
        # those, in the precision of the scale as read. The table is
        # frozen, so no custom gradient is needed.
        table = table.with_tensors(table.read_tensors())
        outputs = table.dequantize_rows(
            inputs, backend.result_type(layer.compute_dtype, table.scale.dtype)
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
        route = None
        if not layer.inputs_quantizer:
            # A route reads the table as stored: a tied table is the
            # forward table.
            orientation = "tied" if layer.tie_weights else "untied"
            if layer.tie_weights:
                stored = self._get_lookup_quantized_weight(layer, geometry)
            else:
                stored = self._get_reverse_lookup_quantized_weight(
                    layer, geometry
                )
            route = kernels.select_lookup_route(
                layer, stored, orientation, inputs
            )
        if route is not None:
            logits = _contract_routed(
                route, layer, stored, orientation, inputs, dtype
            )
        else:
            table = self._get_reverse_lookup_quantized_weight(layer, geometry)
            inputs_q, inputs_scale = inputs, None
            if layer.inputs_quantizer:
                inputs_q, inputs_scale = layer.inputs_quantizer(inputs)
            if table.scheme.has_zero_point:
                table = table.with_tensors(table.read_tensors())
                # A zero point cannot be pulled out of the matmul, so the table
                # is dequantized first, in the precision its scale reads in:
                # below it, only the inputs and the logits are cast to `dtype`.
                float_table = table.dequantize(
                    backend.result_type(dtype, table.scale.dtype)
                )
                logits = ops.matmul(inputs_q, float_table)
                if inputs_scale is not None:
                    logits = ops.divide(logits, inputs_scale)
            else:
                # Matmul on the codes, then the view applies the scales to the
                # logits.
                logits = ops.cast(ops.matmul(inputs_q, table.unpack()), dtype)
                logits = table.apply_scale(logits, inputs_scale)
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

    def _encode_lookup(self, layer, geometry, weight, config):
        """Returns `(codes, scale, zero_point)` of a float forward table.

        An ungrouped scheme takes one abs-max scale per row, or the
        config's `weight_quantizer` when it sets one. A grouped scheme
        takes a scale and a zero point per row and group of columns.
        """
        scheme = self._get_lookup_scheme(layer, config)
        if scheme.group_size is None:
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=-1, value_range=scheme.code_range, output_dtype="int8"
                ),
            )
            codes, scale = weight_quantizer(weight, to_numpy=True)
            scale, zero_point = ops.squeeze(scale, axis=-1), None
        else:
            # The groups run along `output_dim`, the first axis of the
            # transposed table.
            codes, scale, zero_point = (
                ops.transpose(part)
                for part in abs_max_quantize_grouped_with_zero_point(
                    ops.transpose(weight),
                    block_size=scheme.group_size,
                    value_range=scheme.code_range,
                    dtype="int8",
                    to_numpy=True,
                )
            )
        layout = self._get_lookup_layout(-1, weight.shape[-1])
        return layout.pack(codes), scale, zero_point

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
            if zero_point is not None:
                # Only the grouped parameters are 2-D.
                scale = ops.transpose(scale)
                zero_point = ops.transpose(zero_point)
            encoded.append((ops.transpose(codes), scale, zero_point))
            del layer.reverse_embeddings
        layer.quantized_build(geometry.weight_shape, self.name, config)
        tables = self.quantized_weights(layer)
        if not tables:
            raise NotImplementedError(
                f"Mode '{self.name}' has no `QuantizedWeight` view of the "
                "embeddings table, so `LookupHandlers` cannot write its codes."
            )
        # The views of an untied layer hold the variables themselves.
        for table, (codes, scale, zero_point) in zip(
            tables, encoded, strict=True
        ):
            table.codes.assign(codes)
            table.scale.assign(scale)
            if zero_point is not None:
                table.zero_point.assign(zero_point)


def _contract_routed(route, layer, table, orientation, inputs, dtype):
    """Runs a reverse lookup `route` with the gradient of the reference.

    `table` is the stored table. The route runs forward only. The
    backward repeats the arithmetic of autodiff through the reference
    contraction of `_reverse_lookup`, on the `(D, V)` view of the stored
    tensors, so the input gradient is the reference's. Its matmul can sum
    in another order than autodiff's: with oneDNN on x86, TensorFlow runs
    autodiff's `MatMul`, which transposes the table, as another gemm
    call. Only the stored tensors are kept for it, and they get no
    gradient: the table is frozen.
    """

    @ops.custom_gradient
    def contract(inputs, *tensors):
        stored = table.with_tensors(tensors)

        def grad_fn(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            view = kernels.reverse_lookup_view(stored, orientation)
            if view.zero_point is not None:
                precision = backend.result_type(dtype, view.scale.dtype)
                grad = ops.matmul(
                    ops.cast(upstream, precision),
                    ops.transpose(view.dequantize(precision)),
                )
            else:
                grad = ops.cast(ops.divide(upstream, view.scale), dtype)
                grad = ops.matmul(
                    grad, ops.transpose(ops.cast(view.unpack(), dtype))
                )
            return (ops.cast(grad, dtype),) + (None,) * len(tensors)

        logits = kernels.run_lookup_route(
            route, layer, stored, orientation, inputs
        )
        return ops.cast(logits, dtype), grad_fn

    # Read inside the autocast scope: on TensorFlow eager the gradient
    # runs after it, and the scale variable would then read in another
    # dtype.
    return contract(inputs, *table.read_tensors())
