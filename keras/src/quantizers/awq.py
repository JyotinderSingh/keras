"""AWQ (Activation-aware Weight Quantization) algorithm implementation.

AWQ protects salient weights by finding optimal per-channel scales based on
activation magnitudes, then applies those scales before quantization.

Reference: https://arxiv.org/abs/2306.00978
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

# Maximum number of activation rows stashed per problem of a layer for the
# AutoAWQ-style clipping search (a kernel with a batch axis stashes this
# many rows per expert). Bounds calibration memory; a few hundred rows is
# enough to estimate per-group reconstruction error (matches AutoAWQ's
# ``n_sample_token`` default of 512).
MAX_CLIP_SAMPLE_ROWS = 512


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


def awq_search_optimal_scales(
    weights,
    activation_magnitudes,
    *,
    num_grid_points=20,
    group_size=-1,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Search for optimal AWQ scales using grid search.

    The AWQ algorithm finds scaling factors that protect salient weights.
    For each channel, we search for an optimal ratio in [0, 1] that minimizes
    the activation-weighted quantization error.

    The key insight: we MULTIPLY weights by scales before quantization to
    expand salient weights. This ensures quantization noise is small relative
    to the expanded weight magnitude. During inference, we divide by scales
    to restore the original magnitude.

    Scale formula (reference llm-awq / AutoAWQ):
        scales = (x_stat**ratio / w_stat**(1 - ratio)).clamp(min=1e-4)
    where ``x_stat`` is the per-channel activation magnitude and ``w_stat`` is
    the per-in-channel weight magnitude from :func:`_get_weight_scale`. Scales
    are then normalized by ``sqrt(max * min)``.
    Loss function: Activation-weighted MSE (approximates output error)

    Args:
        weights: Weight tensor [out_features, in_features] (transposed kernel).
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points. Defaults to 20.
        group_size: Group size for quantization (-1 for per-channel).
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        best_scales: Optimal per-channel scales [in_features].
    """
    in_features = ops.shape(weights)[1]

    # Per-channel activation statistic (reference AWQ uses mean(|x|)).
    x_stat = ops.cast(activation_magnitudes, "float32")
    # Avoid zero or very small values.
    x_stat = ops.where(ops.less(x_stat, 1e-8), ops.ones_like(x_stat), x_stat)

    # Per-in-channel weight statistic (llm-awq get_weight_scale). This is the
    # term the previous implementation dropped.
    w_stat = _get_weight_scale(weights, group_size)
    w_stat = ops.where(ops.less(w_stat, 1e-8), ops.ones_like(w_stat), w_stat)

    best_loss = None
    best_scales = ops.ones((in_features,), dtype="float32")

    # Grid search over ratio values from 0 to 1
    for i in range(num_grid_points + 1):
        ratio = i / num_grid_points

        # Reference scale formula: balance activation and weight magnitudes.
        scales = ops.divide(
            ops.power(x_stat, ratio), ops.power(w_stat, 1.0 - ratio)
        )
        scales = ops.maximum(scales, 1e-4)

        # Normalize scales to avoid extreme values
        scale_mean = ops.sqrt(ops.multiply(ops.max(scales), ops.min(scales)))
        scale_mean = ops.maximum(scale_mean, 1e-8)
        scales = ops.divide(scales, scale_mean)

        # Apply scales to weights by MULTIPLYING (expand salient weights)
        # weights_scaled: [out_features, in_features]
        weights_scaled = ops.multiply(weights, scales)

        dequantized = _fake_quantize_weights(
            weights_scaled, in_features, group_size, compute_scale_zero
        )

        # Scale back down by DIVIDING to restore original magnitude
        reconstructed = ops.divide(dequantized, scales)

        # Compute activation-weighted MSE loss
        # This approximates the output error: ||W*X - W_hat*X||^2
        # by weighting each channel's error by x_stat^2
        weight_error = ops.square(ops.subtract(weights, reconstructed))
        # Weight by activation magnitudes squared (broadcast over out_features)
        weighted_error = ops.multiply(weight_error, ops.square(x_stat))
        loss = ops.mean(weighted_error)

        # Track best
        if best_loss is None:
            best_loss = loss
            best_scales = scales
        else:
            is_better = ops.less(loss, best_loss)
            if is_better:
                best_loss = loss
                best_scales = scales

    return best_scales


def awq_search_best_clip(
    weights_scaled,
    activation_sample,
    awq_scales,
    *,
    group_size=-1,
    num_grid_points=20,
    max_shrink=0.5,
    output_channel_batch_size=64,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Search per-group weight clipping bounds (AutoAWQ ``best_clip``).

    After the AWQ scales have been applied, quantization error can be further
    reduced by clipping the weight magnitudes. For each output channel and
    quantization group this grid-searches a shrink factor on the per-group max
    and keeps the value that minimizes the reconstruction error of the layer
    output against a stashed activation sample.

    The reconstruction uses the *scaled* input (``x / awq_scales``) so that it
    matches the effective inference computation ``(x / s) @ (W * s)^T``.

    Args:
        weights_scaled: Scaled weight matrix ``[out_features, in_features]``
            (``W * awq_scales``).
        activation_sample: Raw activation sample ``[rows, in_features]``.
        awq_scales: Per-in-channel AWQ scales ``[in_features]``.
        group_size: Quantization group size (``-1`` for per-channel).
        num_grid_points: Number of shrink factors to try. Defaults to 20.
        max_shrink: Maximum fractional shrink of the per-group max (the search
            spans ``[1, 1 - max_shrink]``). Defaults to 0.5.
        output_channel_batch_size: Output-channel batch size, to bound peak
            memory.
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        clip_bound: Per-group clipping bound ``[out_features, n_groups, 1]``.
        effective_group_size: Group size used for reshaping.
        n_groups: Number of groups.
    """
    out_features, in_features = ops.shape(weights_scaled)
    awq_scales = ops.cast(awq_scales, "float32")
    x = ops.cast(activation_sample, "float32")
    if ops.ndim(x) > 2:
        x = ops.reshape(x, (-1, in_features))
    # Effective input to the scaled weights (see docstring).
    x_scaled = ops.divide(x, awq_scales)

    if group_size and group_size > 0 and in_features % group_size == 0:
        effective_group_size = group_size
    else:
        # Per-channel, or a group size that does not evenly divide the input:
        # fall back to a single group per output channel for the clip search.
        effective_group_size = in_features
    n_groups = in_features // effective_group_size
    # [rows, n_groups, effective_group_size]
    x_grouped = ops.reshape(x_scaled, (-1, n_groups, effective_group_size))
    w_grouped = ops.reshape(
        weights_scaled, (out_features, n_groups, effective_group_size)
    )
    group_max = ops.max(
        ops.abs(w_grouped), axis=-1, keepdims=True
    )  # [out_features, n_groups, 1]

    num_shrinks = max(1, int(num_grid_points))
    step = max_shrink / num_grid_points
    quantizer_group_size = effective_group_size if n_groups > 1 else -1

    clip_bound_parts = []
    num_batches = (
        out_features + output_channel_batch_size - 1
    ) // output_channel_batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * output_channel_batch_size
        batch_end = min(batch_start + output_channel_batch_size, out_features)
        batch_size = batch_end - batch_start
        # [batch_size, n_groups, effective_group_size]
        batch_weights = w_grouped[batch_start:batch_end]
        batch_group_max = group_max[batch_start:batch_end]
        # Reference output for this output-channel batch:
        # [batch_size, rows, n_groups].
        reference_output = ops.einsum("rng,ong->orn", x_grouped, batch_weights)

        clip_bound = batch_group_max
        best_error = None
        for shrink_idx in range(num_shrinks):
            bound = ops.multiply(batch_group_max, 1.0 - shrink_idx * step)
            weights_clipped = ops.clip(
                batch_weights, ops.negative(bound), bound
            )
            weights_dequantized = _fake_quantize_weights(
                ops.reshape(weights_clipped, (batch_size, in_features)),
                in_features,
                quantizer_group_size,
                compute_scale_zero,
            )
            weights_dequantized = ops.reshape(
                weights_dequantized,
                (batch_size, n_groups, effective_group_size),
            )
            clipped_output = ops.einsum(
                "rng,ong->orn", x_grouped, weights_dequantized
            )
            error = ops.mean(
                ops.square(ops.subtract(clipped_output, reference_output)),
                axis=1,
            )  # [batch_size, n_groups]
            error = ops.expand_dims(error, axis=-1)  # [batch_size, n_groups, 1]
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
    *,
    num_grid_points=20,
    group_size=-1,
    apply_clip=False,
    activation_sample=None,
    clip_num_grid_points=20,
    clip_max_shrink=0.5,
    compute_scale_zero=compute_awq_scale_zero,
):
    """Quantize a weight matrix using AWQ.

    This function performs the complete AWQ quantization process:
    1. Find optimal per-channel scales via grid search
    2. Apply scales to weights
    3. (Optional) Search and apply per-group clipping bounds
    4. Compute quantization parameters
    5. Quantize weights

    Args:
        weights_transpose: Weight matrix [out_features, in_features].
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points.
        group_size: Group size for quantization.
        apply_clip: Whether to run the AutoAWQ-style clipping search. Requires
            ``activation_sample`` to be provided.
        activation_sample: Optional raw activation sample [rows, in_features]
            used only for the clipping search.
        clip_num_grid_points: Number of shrink factors for the clipping
            search.
        clip_max_shrink: Maximum fractional shrink for the clipping search.
        compute_scale_zero: Function to compute scale and zero for
            quantization.

    Returns:
        quantized: Quantized weights [out_features, in_features].
        scale: Quantization scales [out_features, n_groups].
        zero: Zero points [out_features, n_groups].
        awq_scales: AWQ per-channel scales [in_features].
        g_idx: Group indices [in_features].
    """
    out_features, in_features = ops.shape(weights_transpose)

    # Step 1: Find optimal AWQ scales via grid search
    awq_scales = awq_search_optimal_scales(
        weights_transpose,
        activation_magnitudes,
        num_grid_points=num_grid_points,
        group_size=group_size,
        compute_scale_zero=compute_scale_zero,
    )

    # Step 2: Apply AWQ scales by MULTIPLYING (expand salient weights)
    # weights_scaled: [out_features, in_features]
    weights_scaled = ops.multiply(weights_transpose, awq_scales)

    # Step 3: (Optional) Search and apply per-group clipping bounds.
    if apply_clip and activation_sample is not None:
        clip_bound, effective_group_size, n_groups = awq_search_best_clip(
            weights_scaled,
            activation_sample,
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
        # Per-channel quantization (no grouping)
        scale, zero, maxq = compute_scale_zero(weights_scaled, group_size=-1)

        # Quantize
        quantized = quantize_with_zero_point(weights_scaled, scale, zero, maxq)

        # Build group indices (all 0s for per-channel). Integer group
        # metadata, kept as int32.
        g_idx = ops.zeros((in_features,), dtype="int32")
    else:
        # Grouped quantization - use proper per-row grouping
        scale, zero, maxq = compute_scale_zero(
            weights_scaled, group_size=group_size
        )

        # Compute group indices: maps each input feature to its group
        g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")

        # Quantize using group index mapping
        quantized = quantize_with_sz_map(
            weights_scaled, scale, zero, g_idx, maxq
        )

    return quantized, scale, zero, awq_scales, g_idx


class AWQCalibrator(Calibrator):
    """AWQ calibrator for a single layer.

    This class accumulates activation statistics during calibration and
    performs AWQ quantization on layer weights.

    The AWQ algorithm works by:
    1. Collecting per-channel mean activation magnitudes
    2. Using activation magnitudes to determine weight saliency
    3. Finding optimal per-channel scales via grid search
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

        # Initialize activation magnitude accumulator (running per-channel
        # MEAN of |x|, as in the reference AWQ implementations), one row
        # of statistics per problem of the calibration view.
        shape = (self.rows,)
        if self.batch > 1:
            shape = (self.batch,) + shape
        self.activation_magnitudes = ops.zeros(shape, dtype="float32")

        # Bounded stash of raw activation rows for the clipping search.
        self._clip_samples = []
        self._clip_sample_rows = 0

    def observe(self, inputs):
        """Updates the per-channel activation magnitudes with a new batch.

        This tracks the running per-channel MEAN of the absolute activation
        value across all calibration batches (matching llm-awq / AutoAWQ),
        accumulated with a numerically stable batch-count-weighted update. It
        also stashes a bounded sample of raw activation rows that the clipping
        search reuses.

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

        # Running per-channel mean of |x| via a stable weighted update:
        #   mean <- mean + (batch_mean - mean) * n / (count + n)
        batch_mean = ops.mean(ops.abs(x), axis=-2)
        delta = ops.subtract(batch_mean, self.activation_magnitudes)
        self.activation_magnitudes = ops.add(
            self.activation_magnitudes,
            ops.multiply(delta, num_new_samples / total_samples),
        )
        self.num_samples = total_samples

        # Stash a bounded sample of raw activations for the clipping search.
        if (
            self.config.apply_clip
            and self._clip_sample_rows < MAX_CLIP_SAMPLE_ROWS
        ):
            take = min(
                num_new_samples, MAX_CLIP_SAMPLE_ROWS - self._clip_sample_rows
            )
            self._clip_samples.append(x[..., :take, :])
            self._clip_sample_rows += take

    def quantize(self):
        """Perform AWQ quantization on the layer.

        This method:
        1. Runs the AWQ grid search to find optimal scales
        2. Quantizes the layer weights
        3. Updates the layer's quantized variables
        """
        kernel = self._kernel_view()
        magnitudes = self.activation_magnitudes

        # Assemble the stashed activation sample for the clipping search.
        apply_clip = self.config.apply_clip
        activation_sample = None
        if apply_clip and self._clip_samples:
            activation_sample = ops.concatenate(self._clip_samples, axis=-2)
        if self.batch == 1:
            magnitudes = ops.expand_dims(magnitudes, 0)
            if activation_sample is not None:
                activation_sample = ops.expand_dims(activation_sample, 0)

        codes = []
        scales = []
        zeros = []
        input_scales = []
        group_indices = []
        for index in range(self.batch):
            # Perform AWQ quantization
            quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
                ops.transpose(kernel[index]),
                magnitudes[index],
                num_grid_points=self.config.num_grid_points,
                group_size=self.config.group_size,
                apply_clip=apply_clip and activation_sample is not None,
                activation_sample=(
                    None
                    if activation_sample is None
                    else activation_sample[index]
                ),
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

    def release(self):
        """Drops the statistics and the stashed rows."""
        del self.activation_magnitudes
        self._clip_samples = []
        self._clip_sample_rows = 0
