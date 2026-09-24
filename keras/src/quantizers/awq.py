"""AWQ (Activation-aware Weight Quantization) algorithm implementation.

AWQ protects salient weights by finding optimal per-channel scales based on
activation magnitudes, then applies those scales before quantization.

Reference: https://arxiv.org/abs/2306.00978, llm-awq (mit-han-lab) and
AutoAWQ (casper-hansen), whose duo-scaling grid and clipping search this
module follows.

Both searches score a candidate by the layer's mean squared output error
over the calibration set, the paper's objective. The error is computed
exactly from the input second moment `mean(x x^T)` the calibrator
accumulates, not from stored activations: for a weight error `E` it is
`mean(sum(E @ G * E, axis=1))`, and a group's partial output uses the
group's diagonal block of `G`. The references run the layer on the stored
activations, all of them for the scale search and a strided sample of 512
rows for the clipping search.

Two departures from the references remain. They search one scale per set
of layers that share an input (query, key and value together) and score
it on the output of the enclosing module, folding `1 / s` into the
preceding operation; Keras searches each layer on its own output and
keeps `1 / s` in the layer as `awq_scales`, so the layers need not agree.
And `CalibrationRun` feeds each block the quantized outputs of the block
before it, GPTQ's rule, where the references calibrate every block on
the float model's activations; on SmolLM2-135M the two gave the same
held-out perplexity over three calibration seeds.
"""

import functools

from keras.src import ops
from keras.src import quantizers
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.calibrator import Calibrator
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_zero_point


def _get_weight_scale(weights, group_size):
    """Per-in-channel weight magnitude used in the AWQ scale formula.

    Mirrors llm-awq's ``get_weight_scale`` (and AutoAWQ's weight term): the
    weights are normalized by their per-group maximum so each group lives on a
    0-1 scale, then averaged over the output channels to obtain a single
    statistic per input channel.

    Args:
        weights: Weight matrix ``[out_features, in_features]``.
        group_size: Quantization group size (``-1`` for per-channel).

    Returns:
        Per-in-channel weight statistic ``[in_features]``.
    """
    weights = ops.cast(weights, "float32")
    out_features, in_features = ops.shape(weights)
    w_abs = ops.abs(weights)
    if group_size and group_size > 0 and in_features % group_size == 0:
        n_groups = in_features // group_size
        w_grouped = ops.reshape(w_abs, (out_features, n_groups, group_size))
        group_max = ops.max(w_grouped, axis=2, keepdims=True)
        w_norm = ops.divide(w_grouped, ops.add(group_max, 1e-6))
        w_norm = ops.reshape(w_norm, (out_features, in_features))
    else:
        group_max = ops.max(w_abs, axis=1, keepdims=True)
        w_norm = ops.divide(w_abs, ops.add(group_max, 1e-6))
    return ops.mean(w_norm, axis=0)


# The scale/zero rule of AWQ's 4-bit asymmetric codes; `group_size` is
# passed per call because the clipping search narrows it to one group.
compute_awq_scale_zero = functools.partial(
    compute_quantization_parameters,
    bits=4,
    symmetric=False,
    per_channel=True,
    compute_dtype="float32",
)


def _fake_quantize_weights(
    weights_scaled,
    in_features,
    group_size,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Quantize then dequantize a weight matrix (4-bit asymmetric).

    Shared by the scale search and the clipping search so both evaluate the
    exact same quantizer that is used to produce the final packed weights.

    Args:
        weights_scaled: Weight matrix ``[out_features, in_features]``.
        in_features: Number of input features (columns).
        group_size: Quantization group size (``-1`` for per-channel).
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        The dequantized weight matrix, same shape as ``weights_scaled``.
    """
    if group_size == -1:
        scale, zero, maxq = compute_scale_zero(weights_scaled, group_size=-1)
        quantized = quantize_with_zero_point(weights_scaled, scale, zero, maxq)
        return dequantize_with_zero_point(quantized, scale, zero)

    scale, zero, maxq = compute_scale_zero(
        weights_scaled, group_size=group_size
    )
    g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")
    quantized = quantize_with_sz_map(weights_scaled, scale, zero, g_idx, maxq)
    return dequantize_with_sz_map(quantized, scale, zero, g_idx)


def _output_error(weight_error, second_moment):
    """Mean squared output error a weight error causes on the calibration set.

    `weight_error` is `[out_features, in_features]`, the float weights minus
    their reconstruction; `second_moment` is the `[in_features, in_features]`
    mean of `x x^T` over the calibration rows. The result equals
    `mean(square(x @ weight_error^T))` over rows and output features, taken
    without the rows.
    """
    projected = ops.matmul(weight_error, ops.cast(second_moment, "float32"))
    return ops.mean(ops.sum(ops.multiply(projected, weight_error), axis=1))


def awq_search_optimal_scales(
    weights,
    activation_magnitudes,
    second_moment,
    *,
    num_grid_points=20,
    group_size=-1,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Search for optimal AWQ scales using grid search.

    The AWQ algorithm finds scaling factors that protect salient weights.
    For each channel, we search for an optimal ratio in `[0, 1)` that
    minimizes the layer's output error after quantization.

    The key insight: we MULTIPLY weights by scales before quantization to
    expand salient weights. This ensures quantization noise is small relative
    to the expanded weight magnitude. During inference, we divide by scales
    to restore the original magnitude.

    Scale formula (AutoAWQ's duo scaling; llm-awq uses the activation term
    alone):
        scales = (x_stat**ratio / (w_stat**(1 - ratio) + 1e-4)).clamp(1e-4)
    where `x_stat` is the per-channel activation magnitude and `w_stat` is
    the per-in-channel weight magnitude from `_get_weight_scale`. Scales are
    then normalized by `sqrt(max * min)`. `ratio` takes `num_grid_points`
    values `i / num_grid_points`, so `1.0` is not a candidate, as in the
    references.

    Loss: the layer's mean squared output error over the calibration set,
    computed from `second_moment` (`_output_error`). The reference runs
    the layer on the stored activations; the value is the same.

    Args:
        weights: Weight tensor [out_features, in_features] (transposed kernel).
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        second_moment: Mean of `x x^T` over the calibration rows
            [in_features, in_features].
        num_grid_points: Number of grid search points. Defaults to 20.
        group_size: Group size for quantization (-1 for per-channel).
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        best_scales: Optimal per-channel scales [in_features].
    """
    in_features = ops.shape(weights)[1]
    x_stat = ops.cast(activation_magnitudes, "float32")
    # A channel that never fires gets scale 1 rather than the clamp floor,
    # which would drag the normalization of every other channel with it.
    x_stat = ops.where(ops.less(x_stat, 1e-8), ops.ones_like(x_stat), x_stat)
    w_stat = _get_weight_scale(weights, group_size)

    best_loss = None
    best_scales = ops.ones((in_features,), dtype="float32")
    for i in range(num_grid_points):
        ratio = i / num_grid_points
        scales = ops.divide(
            ops.power(x_stat, ratio),
            ops.add(ops.power(w_stat, 1.0 - ratio), 1e-4),
        )
        scales = ops.maximum(scales, 1e-4)
        scale_mean = ops.sqrt(ops.multiply(ops.max(scales), ops.min(scales)))
        scales = ops.divide(scales, scale_mean)
        scales = ops.where(ops.isfinite(scales), scales, ops.ones_like(scales))

        weights_scaled = ops.multiply(weights, scales)
        dequantized = _fake_quantize_weights(
            weights_scaled, in_features, group_size, compute_scale_zero
        )
        reconstructed = ops.divide(dequantized, scales)
        loss = _output_error(
            ops.subtract(weights, reconstructed), second_moment
        )
        # Strict improvement keeps the first minimum, as in the reference.
        if best_loss is None or ops.less(loss, best_loss):
            best_loss = loss
            best_scales = scales
    return best_scales


def awq_search_best_clip(
    weights_scaled,
    second_moment,
    awq_scales,
    *,
    group_size=-1,
    num_grid_points=20,
    max_shrink=0.5,
    output_channel_batch_size=64,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Per-group clipping search (llm-awq `auto_clip_layer`).

    For every output channel and group, the group's absolute maximum is
    shrunk by `1 - i / num_grid_points` for `i` in
    `range(int(max_shrink * num_grid_points))`, the scaled weights are
    clipped to the bound and fake-quantized, and the bound with the smallest
    partial-output error against the unclipped scaled weights is kept. The
    error is exact over the calibration set: it uses the group's diagonal
    block of the second moment of the scaled inputs `x / awq_scales`, where
    the reference estimates it on a strided sample of 512 rows. The search
    runs after the scale search, on the scaled weights, as in the reference.

    Args:
        weights_scaled: Scaled weights [out_features, in_features].
        second_moment: Mean of `x x^T` over the calibration rows
            [in_features, in_features], for the unscaled inputs.
        awq_scales: The per-input-channel scales [in_features].
        group_size: Quantization group size (-1 for per-channel).
        num_grid_points: Grid resolution; the step is `1 / num_grid_points`.
            Defaults to 20.
        max_shrink: Largest shrink searched, exclusive: with the defaults
            the bounds run from `1.0` down to `0.55` of the group max.
            Defaults to 0.5.
        output_channel_batch_size: Output channels searched per step, to
            bound the intermediate tensors. It does not change the result.
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        `(clip_bound, effective_group_size, n_groups)`: the bound is
        `[out_features, n_groups, 1]`.
    """
    out_features, in_features = ops.shape(weights_scaled)
    awq_scales = ops.cast(awq_scales, "float32")
    if group_size and group_size > 0 and in_features % group_size == 0:
        effective_group_size = group_size
    else:
        effective_group_size = in_features
    n_groups = in_features // effective_group_size

    # The second moment of the scaled inputs `x / s` is `G / (s s^T)`; a
    # group's partial output only involves its diagonal block.
    inverse_scales = ops.reciprocal(awq_scales)
    scaled_moment = ops.multiply(
        ops.cast(second_moment, "float32"),
        ops.outer(inverse_scales, inverse_scales),
    )
    blocks = ops.reshape(
        scaled_moment,
        (n_groups, effective_group_size, n_groups, effective_group_size),
    )
    # [group, group] -> [n_groups, group, group]
    blocks = ops.transpose(ops.diagonal(blocks, axis1=0, axis2=2), (2, 0, 1))

    w_grouped = ops.reshape(
        weights_scaled, (out_features, n_groups, effective_group_size)
    )
    group_max = ops.max(ops.abs(w_grouped), axis=-1, keepdims=True)
    num_shrinks = int(max_shrink * num_grid_points)
    quantizer_group_size = effective_group_size if n_groups > 1 else -1
    clip_bound_parts = []
    num_batches = (
        out_features + output_channel_batch_size - 1
    ) // output_channel_batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * output_channel_batch_size
        batch_end = min(batch_start + output_channel_batch_size, out_features)
        batch_size = batch_end - batch_start
        batch_weights = w_grouped[batch_start:batch_end]
        batch_group_max = group_max[batch_start:batch_end]
        clip_bound = batch_group_max
        best_error = None
        for shrink_idx in range(num_shrinks):
            bound = ops.multiply(
                batch_group_max, 1.0 - shrink_idx / num_grid_points
            )
            weights_clipped = ops.clip(
                batch_weights, ops.negative(bound), bound
            )
            weights_dequantized = _fake_quantize_weights(
                ops.reshape(weights_clipped, (batch_size, in_features)),
                in_features,
                quantizer_group_size,
                compute_scale_zero,
            )
            weight_error = ops.subtract(
                ops.reshape(
                    weights_dequantized,
                    (batch_size, n_groups, effective_group_size),
                ),
                batch_weights,
            )
            # Mean squared partial-output error per (channel, group).
            error = ops.einsum(
                "ong,ngh,onh->on", weight_error, blocks, weight_error
            )
            error = ops.expand_dims(error, axis=-1)
            if best_error is None:
                best_error = error
                clip_bound = bound
            else:
                better = ops.less(error, best_error)
                best_error = ops.where(better, error, best_error)
                clip_bound = ops.where(better, bound, clip_bound)
        clip_bound_parts.append(clip_bound)
    clip_bound = ops.concatenate(clip_bound_parts, axis=0)
    return clip_bound, effective_group_size, n_groups


def awq_quantize_matrix(
    weights_transpose,
    activation_magnitudes,
    second_moment,
    *,
    num_grid_points=20,
    group_size=-1,
    apply_clip=False,
    clip_num_grid_points=20,
    clip_max_shrink=0.5,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Quantizes one weight matrix with AWQ.

    Runs the scale search, scales the weights, optionally runs the clipping
    search on the scaled weights, and quantizes them group-wise.

    Args:
        weights_transpose: Weights [out_features, in_features].
        activation_magnitudes: Per-channel mean `|x|` [in_features].
        second_moment: Mean of `x x^T` over the calibration rows
            [in_features, in_features].
        num_grid_points: Grid points of the scale search.
        group_size: Quantization group size (-1 for per-channel).
        apply_clip: Whether to run the clipping search.
        clip_num_grid_points: Grid resolution of the clipping search.
        clip_max_shrink: Largest shrink of the clipping search.
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        `(quantized, scale, zero, awq_scales, g_idx)`: the codes
        [out_features, in_features], the multiplier scale and zero point
        [out_features, n_groups], the input scales [in_features] and the
        group index [in_features].
    """
    out_features, in_features = ops.shape(weights_transpose)
    awq_scales = awq_search_optimal_scales(
        weights_transpose,
        activation_magnitudes,
        second_moment,
        num_grid_points=num_grid_points,
        group_size=group_size,
        compute_scale_zero=compute_scale_zero,
    )
    weights_scaled = ops.multiply(weights_transpose, awq_scales)

    if apply_clip:
        clip_bound, effective_group_size, n_groups = awq_search_best_clip(
            weights_scaled,
            second_moment,
            awq_scales,
            group_size=group_size,
            num_grid_points=clip_num_grid_points,
            max_shrink=clip_max_shrink,
            compute_scale_zero=compute_scale_zero,
        )
        w_grouped = ops.reshape(
            weights_scaled, (out_features, n_groups, effective_group_size)
        )
        w_grouped = ops.clip(w_grouped, ops.negative(clip_bound), clip_bound)
        weights_scaled = ops.reshape(w_grouped, (out_features, in_features))

    if group_size == -1:
        scale, zero, maxq = compute_scale_zero(weights_scaled, group_size=-1)
        quantized = quantize_with_zero_point(weights_scaled, scale, zero, maxq)
        g_idx = ops.zeros((in_features,), dtype="int32")
    else:
        scale, zero, maxq = compute_scale_zero(
            weights_scaled, group_size=group_size
        )
        g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")
        quantized = quantize_with_sz_map(
            weights_scaled, scale, zero, g_idx, maxq
        )
    return quantized, scale, zero, awq_scales, g_idx


class AWQCalibrator(Calibrator):
    """AWQ calibrator for a single layer.

    This class accumulates activation statistics during calibration and
    performs AWQ quantization on layer weights.

    The AWQ algorithm works by:
    1. Collecting per-channel mean activation magnitudes and the input
       second moment, one of each per problem of the calibration view
    2. Using activation magnitudes to determine weight saliency
    3. Finding optimal per-channel scales via grid search on the layer's
       output error
    4. (Optional) Searching per-group weight clipping bounds
    5. Applying scales before quantization to protect salient weights

    Args:
        layer: A layer with a projection geometry (`Dense`, `EinsumDense`)
            that supports the `awq` mode.
        config: `AWQConfig` instance with quantization parameters.
    """

    mode = "awq"

    def __init__(self, layer, config=None):
        config = config or AWQConfig(dataset=None, tokenizer=None)
        super().__init__(layer, config)
        self.compute_scale_zero = compute_awq_scale_zero

        # Running per-channel mean of |x| (the reference's activation
        # statistic) and the running mean of `x x^T`, from which both
        # searches compute the output error exactly.
        shape = (self.rows,)
        moment_shape = (self.rows, self.rows)
        if self.batch > 1:
            shape = (self.batch,) + shape
            moment_shape = (self.batch,) + moment_shape
        self.activation_magnitudes = ops.zeros(shape, dtype="float32")
        self.second_moment = ops.zeros(moment_shape, dtype="float32")

    def observe(self, inputs):
        """Updates the activation statistics with a new batch.

        Tracks the running per-channel mean of `|x|` and the running mean
        of `x x^T` over all calibration rows, each with a batch-count-weighted
        update, so the result does not depend on the batching.

        Args:
            inputs: A 2D or higher-dimensional tensor of input activations
                from a calibration batch, in the layer's input layout.

        Raises:
            ValueError: If the feature dimension of `inputs` does not match
                the per-channel statistics `self.activation_magnitudes`.
        """
        self._check_inputs(inputs)
        input_features = self.view.input_features(inputs)
        if input_features != self.rows:
            raise ValueError(
                f"Activation statistics ({self.rows}) do not match input "
                f"features ({input_features})."
            )
        # Lay out as [batch_samples, in_features], with a leading problem
        # axis for a batched kernel.
        x = ops.cast(self.view.inputs_to_view(inputs), "float32")
        num_new_samples = int(ops.shape(x)[-2])
        total_samples = self.num_samples + num_new_samples
        weight = num_new_samples / total_samples

        batch_mean = ops.mean(ops.abs(x), axis=-2)
        self.activation_magnitudes = ops.add(
            self.activation_magnitudes,
            ops.multiply(
                ops.subtract(batch_mean, self.activation_magnitudes), weight
            ),
        )
        gram = ops.matmul(ops.swapaxes(x, -1, -2), x)
        gram = ops.divide(ops.add(gram, ops.swapaxes(gram, -1, -2)), 2.0)
        batch_moment = ops.divide(gram, float(num_new_samples))
        self.second_moment = ops.add(
            self.second_moment,
            ops.multiply(
                ops.subtract(batch_moment, self.second_moment), weight
            ),
        )
        self.num_samples = total_samples

    def quantize(self):
        """Perform AWQ quantization on the layer.

        This method:
        1. Runs the AWQ grid search to find optimal scales
        2. Quantizes the layer weights
        3. Updates the layer's quantized variables
        """
        kernel = self._kernel_view()
        magnitudes = self.activation_magnitudes
        moments = self.second_moment
        if self.batch == 1:
            magnitudes = ops.expand_dims(magnitudes, 0)
            moments = ops.expand_dims(moments, 0)
        apply_clip = self.config.apply_clip and not self._skips_clip()

        codes = []
        scales = []
        zeros = []
        input_scales = []
        group_indices = []
        for index in range(self.batch):
            quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
                ops.transpose(kernel[index]),
                magnitudes[index],
                moments[index],
                num_grid_points=self.config.num_grid_points,
                group_size=self.config.group_size,
                apply_clip=apply_clip,
                compute_scale_zero=self.compute_scale_zero,
            )
            # Cast to uint8 for storage. The algorithm works on `[out, in]`;
            # the layer stores the view's `[in, out]` orientation with the
            # group parameters as `[n_groups, out]`, so the forward pass
            # never transposes.
            codes.append(ops.transpose(ops.cast(quantized, "uint8")))
            scales.append(ops.transpose(scale))
            zeros.append(ops.transpose(zero))
            input_scales.append(awq_scales)
            # Each problem's groups follow the previous problem's.
            group_indices.append(
                ops.add(g_idx, index * ops.shape(scales[-1])[0])
            )
        quantized = ops.concatenate(codes, axis=0)
        scale = ops.concatenate(scales, axis=0)
        zero = ops.concatenate(zeros, axis=0)
        awq_scales = ops.concatenate(input_scales, axis=0)
        g_idx = ops.concatenate(group_indices, axis=0)

        # Pack to 4-bit along the output axis.
        quantized, _, _ = quantizers.pack_int4(
            quantized, axis=-1, dtype="uint8"
        )

        strategy_registry.get_strategy("awq").write_back(
            self.original_layer,
            quantized,
            scale,
            zero,
            g_idx,
            awq_scales=awq_scales,
        )

    def _skips_clip(self):
        """Whether the layer is excluded from the clipping search by name.

        The references skip the query and key projections: the clipping
        objective is one layer's output error, and the attention scores
        depend on the product of the two.
        """
        name = self.original_layer.name
        return any(
            pattern in name for pattern in self.config.clip_skip_patterns
        )

    def release(self):
        """Drops the statistics."""
        del self.activation_magnitudes
        del self.second_moment
