"""AWQ (Activation-aware Weight Quantization) algorithm implementation.

AWQ protects salient weights by finding optimal per-channel scales based on
activation magnitudes, then applies those scales before quantization.

Reference: https://arxiv.org/abs/2306.00978, llm-awq (mit-han-lab) and
AutoAWQ (casper-hansen), whose duo-scaling grid and clipping search this
module follows.

Both searches score a candidate by the layer's mean squared output error
over the calibration set, the paper's objective. The error is computed
exactly from the Hessian `H = 2 mean(x x^T)` of the layer's inputs, which
the calibrator accumulates as GPTQ does, not from stored activations: for
a weight error `E` it is `mean(sum(E @ H * E, axis=1)) / 2`, and a group's
partial output uses the group's diagonal block of `H`. The references run
the layer on stored activations, all of them for the scale search and a
strided sample of 512 rows for the clipping search. For a layer with `K`
inputs the Hessian holds `K^2` floats, as GPTQ's does, more than a 512-row
activation sample (`512 * K`) once `K > 512`; peak AWQ calibration memory
matches GPTQ's.

Departures from the references:

- They search one scale per set of layers that share an input (query, key
  and value together), score it on the output of the enclosing module and
  fold `1 / s` into the preceding operation. Keras searches each layer on
  its own output and keeps `1 / s` in the layer as `awq_scales`, so the
  layers need not agree.
- `CalibrationRun` feeds each block the quantized outputs of the block
  before it, GPTQ's rule, where the references calibrate every block on
  the float model's activations. On SmolLM2-135M the two gave the same
  held-out perplexity over three calibration seeds.
- A channel that never fires has its activation statistic taken as 1, so
  its scale follows the weight term only. In the references its scale
  falls to the `1e-4` floor.
- Each group's range is stretched to include zero (the rule of
  `compute_quantization_parameters`), where llm-awq quantizes the raw
  group range. The two agree on every group that spans zero.
- A width that the group size does not divide ends each row in a short
  group, which the searches pad with zeros as the quantizer does. The
  references refuse such a width.
"""

import functools

from keras.src import ops
from keras.src.quantizers.calibrator import Calibrator
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_sz_map

# The clipping search (llm-awq `auto_clip_layer`) shrinks each group's
# absolute maximum in steps of `1 / _CLIP_GRID_POINTS` and tries
# `int(_CLIP_MAX_SHRINK * _CLIP_GRID_POINTS)` bounds: from the maximum down
# to 0.55 of it. It searches `_CLIP_CHANNEL_BATCH` output channels at a
# time, to bound the intermediate tensors.
_CLIP_GRID_POINTS = 20
_CLIP_MAX_SHRINK = 0.5
_CLIP_CHANNEL_BATCH = 64


def _group_view(weights, group_size):
    """Lays `[out, in]` weights out as `[out, n_groups, group]`.

    `group_size=-1` is one group per row. A last group shorter than the
    group size is padded with zeros, as `compute_quantization_parameters`
    pads it: a zero changes neither a group's absolute maximum nor its
    range, which always includes zero.
    """
    out_features, in_features = ops.shape(weights)
    group = in_features if group_size == -1 else min(group_size, in_features)
    padding = -in_features % group
    if padding:
        weights = ops.pad(weights, [[0, 0], [0, padding]])
    return ops.reshape(weights, (out_features, -1, group))


def _get_weight_scale(weights, group_size):
    """Per-in-channel weight magnitude used in the AWQ scale formula.

    Mirrors llm-awq's `get_weight_scale` (and AutoAWQ's weight term): the
    weights are normalized by their per-group maximum so each group lives on a
    0-1 scale, then averaged over the output channels to obtain a single
    statistic per input channel.

    Args:
        weights: Weight matrix `[out_features, in_features]`.
        group_size: Quantization group size (`-1` for per-channel).

    Returns:
        Per-in-channel weight statistic `[in_features]`.
    """
    out_features, in_features = ops.shape(weights)
    w_grouped = _group_view(ops.abs(ops.cast(weights, "float32")), group_size)
    group_max = ops.max(w_grouped, axis=2, keepdims=True)
    w_norm = ops.divide(w_grouped, ops.add(group_max, 1e-6))
    w_norm = ops.reshape(w_norm, (out_features, -1))[:, :in_features]
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


def _quantize(weights, group_size):
    """Quantizes `[out, in]` weights to AWQ's 4-bit asymmetric codes.

    `group_size=-1` is one group per row. Returns `(codes, scale, zero,
    g_idx)`, with the scale and zero point `[out, n_groups]`.
    """
    in_features = ops.shape(weights)[1]
    scale, zero, maxq = compute_awq_scale_zero(weights, group_size=group_size)
    group = in_features if group_size == -1 else group_size
    g_idx = ops.cast(ops.arange(0, in_features) // group, "int32")
    codes = quantize_with_sz_map(weights, scale, zero, g_idx, maxq)
    return codes, scale, zero, g_idx


def _fake_quantize_weights(weights, group_size):
    """Quantizes and dequantizes `weights` with the final quantizer."""
    codes, scale, zero, g_idx = _quantize(weights, group_size)
    return dequantize_with_sz_map(codes, scale, zero, g_idx)


def _output_error(weight_error, hessian):
    """Twice the mean squared output error a weight error causes.

    `weight_error` is `[out_features, in_features]`, the float weights minus
    their reconstruction; `hessian` is `2 mean(x x^T)` over the calibration
    rows `x`. The result is `2 mean(square(x @ weight_error^T))` over rows
    and output features, computed without the rows. The searches only
    compare errors, so the factor 2 does not change their choice.
    """
    projected = ops.matmul(weight_error, ops.cast(hessian, "float32"))
    return ops.mean(ops.sum(ops.multiply(projected, weight_error), axis=1))


def awq_search_optimal_scales(
    weights,
    activation_magnitudes,
    hessian,
    *,
    num_grid_points=20,
    group_size=-1,
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

    Args:
        weights: Weight tensor [out_features, in_features] (transposed kernel).
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        hessian: `2 mean(x x^T)` over the calibration rows
            [in_features, in_features].
        num_grid_points: Number of grid search points. Defaults to 20.
        group_size: Group size for quantization (-1 for per-channel).

    Returns:
        best_scales: Optimal per-channel scales [in_features].
    """
    in_features = ops.shape(weights)[1]
    x_stat = ops.cast(activation_magnitudes, "float32")
    # A channel that never fires has its statistic taken as 1, so its scale
    # follows the weight term only (see the module docstring).
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
        dequantized = _fake_quantize_weights(weights_scaled, group_size)
        reconstructed = ops.divide(dequantized, scales)
        loss = _output_error(ops.subtract(weights, reconstructed), hessian)
        # Strict improvement keeps the first minimum, as in the reference.
        if best_loss is None or ops.less(loss, best_loss):
            best_loss = loss
            best_scales = scales
    return best_scales


def awq_search_best_clip(weights_scaled, hessian, awq_scales, *, group_size):
    """Per-group clipping search (llm-awq `auto_clip_layer`).

    For every output channel and group, the scaled weights are clipped to
    each bound of the grid (see `_CLIP_GRID_POINTS`) and fake-quantized, and
    the bound with the smallest partial-output error against the unclipped
    scaled weights is kept. The error is exact over the calibration set: it
    uses the group's diagonal block of the Hessian of the scaled inputs
    `x / awq_scales`, where the reference estimates it on a strided sample
    of 512 rows. The search runs after the scale search, on the scaled
    weights, as in the reference.

    Args:
        weights_scaled: Scaled weights [out_features, in_features].
        hessian: `2 mean(x x^T)` over the calibration rows
            [in_features, in_features], for the unscaled inputs.
        awq_scales: The per-input-channel scales [in_features].
        group_size: Quantization group size (-1 for per-channel).

    Returns:
        The bound of every group, `[out_features, n_groups, 1]`.
    """
    out_features, in_features = ops.shape(weights_scaled)
    awq_scales = ops.cast(awq_scales, "float32")
    w_grouped = _group_view(weights_scaled, group_size)
    _, n_groups, effective_group_size = ops.shape(w_grouped)
    padded_features = n_groups * effective_group_size

    # The Hessian of the scaled inputs `x / s` is `H / (s s^T)`; a group's
    # partial output only involves its diagonal block. The padding of a
    # short last group has no inputs.
    inverse_scales = ops.reciprocal(awq_scales)
    scaled_hessian = ops.multiply(
        ops.cast(hessian, "float32"),
        ops.outer(inverse_scales, inverse_scales),
    )
    padding = padded_features - in_features
    if padding:
        scaled_hessian = ops.pad(scaled_hessian, [[0, padding], [0, padding]])
    blocks = ops.reshape(
        scaled_hessian,
        (n_groups, effective_group_size, n_groups, effective_group_size),
    )
    # [group, group] -> [n_groups, group, group]
    blocks = ops.transpose(ops.diagonal(blocks, axis1=0, axis2=2), (2, 0, 1))

    group_max = ops.max(ops.abs(w_grouped), axis=-1, keepdims=True)
    num_shrinks = int(_CLIP_MAX_SHRINK * _CLIP_GRID_POINTS)
    quantizer_group_size = effective_group_size if n_groups > 1 else -1
    clip_bound_parts = []
    for batch_start in range(0, out_features, _CLIP_CHANNEL_BATCH):
        batch_end = min(batch_start + _CLIP_CHANNEL_BATCH, out_features)
        batch_size = batch_end - batch_start
        batch_weights = w_grouped[batch_start:batch_end]
        batch_group_max = group_max[batch_start:batch_end]
        clip_bound = batch_group_max
        best_error = None
        for shrink_idx in range(num_shrinks):
            bound = ops.multiply(
                batch_group_max, 1.0 - shrink_idx / _CLIP_GRID_POINTS
            )
            weights_clipped = ops.clip(
                batch_weights, ops.negative(bound), bound
            )
            weights_dequantized = _fake_quantize_weights(
                ops.reshape(weights_clipped, (batch_size, padded_features)),
                quantizer_group_size,
            )
            weight_error = ops.subtract(
                ops.reshape(
                    weights_dequantized,
                    (batch_size, n_groups, effective_group_size),
                ),
                batch_weights,
            )
            # Partial-output error per (channel, group).
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
    return ops.concatenate(clip_bound_parts, axis=0)


def awq_quantize_matrix(
    weights_transpose,
    activation_magnitudes,
    hessian,
    *,
    num_grid_points=20,
    group_size=-1,
    apply_clip=False,
):
    """Quantizes one weight matrix with AWQ.

    Runs the scale search, scales the weights, optionally runs the clipping
    search on the scaled weights, and quantizes them group-wise.

    Args:
        weights_transpose: Weights [out_features, in_features].
        activation_magnitudes: Per-channel mean `|x|` [in_features].
        hessian: `2 mean(x x^T)` over the calibration rows
            [in_features, in_features].
        num_grid_points: Grid points of the scale search.
        group_size: Quantization group size (-1 for per-channel).
        apply_clip: Whether to run the clipping search.

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
        hessian,
        num_grid_points=num_grid_points,
        group_size=group_size,
    )
    weights_scaled = ops.multiply(weights_transpose, awq_scales)

    if apply_clip:
        clip_bound = awq_search_best_clip(
            weights_scaled, hessian, awq_scales, group_size=group_size
        )
        w_grouped = _group_view(weights_scaled, group_size)
        w_grouped = ops.clip(w_grouped, ops.negative(clip_bound), clip_bound)
        weights_scaled = ops.reshape(w_grouped, (out_features, -1))
        weights_scaled = weights_scaled[:, :in_features]

    quantized, scale, zero, g_idx = _quantize(weights_scaled, group_size)
    return quantized, scale, zero, awq_scales, g_idx


class AWQCalibrator(Calibrator):
    """AWQ calibrator for one layer: the activation statistics of its inputs.

    It accumulates the per-channel mean of `|x|` beside the Hessian
    `2 mean(x x^T)` of every calibrator, one of each per problem of the
    contraction view.

    Args:
        strategy: The `awq` mode's `CalibrationStrategy`.
        layer: A float layer with a projection geometry (`Dense`,
            `EinsumDense`) that supports the `awq` mode.
        config: `AWQConfig` instance with quantization parameters.
    """

    def __init__(self, strategy, layer, config):
        super().__init__(strategy, layer, config)
        self.activation_magnitudes = ops.zeros(
            self._per_problem((self.rows,)), dtype="float32"
        )

    def _observe(self, x):
        # mean <- mean + (batch_mean - mean) * n / (count + n)
        num_new_samples = int(ops.shape(x)[-2])
        total_samples = self.num_samples + num_new_samples
        batch_mean = ops.mean(ops.abs(x), axis=-2)
        delta = ops.subtract(batch_mean, self.activation_magnitudes)
        self.activation_magnitudes = ops.add(
            self.activation_magnitudes,
            ops.multiply(delta, num_new_samples / total_samples),
        )

    def _solve(self, weights, index):
        # The references skip clipping the query and key projections: the
        # attention scores depend on the product of the two, and one
        # layer's output error is a poor guide for them.
        apply_clip = self.config.apply_clip and not any(
            pattern in self.layer.name
            for pattern in self.config.clip_skip_patterns
        )
        codes, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights,
            self._problem(self.activation_magnitudes, index),
            self._problem(self.hessian, index),
            num_grid_points=self.config.num_grid_points,
            group_size=self.config.group_size,
            apply_clip=apply_clip,
        )
        return codes, scale, zero, g_idx, awq_scales
