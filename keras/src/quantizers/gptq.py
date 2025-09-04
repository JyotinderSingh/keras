import types
from functools import partial

from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.ops import linalg
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantizers import GPTQQuantizer
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_zero_point


def _stable_permutation(metric):
    """Return a stable permutation that sorts `metric` in descending order.
    Uses an index-based jitter to break ties deterministically."""
    n = ops.shape(metric)[0]
    idx = ops.arange(0, n, dtype="int32")

    # tiny jitter = (idx / n) * 1e-12 so it never flips a real strict ordering
    jitter = ops.divide(ops.cast(idx, "float32"), ops.cast(n, "float32"))
    metric_jittered = ops.add(metric, ops.multiply(jitter, 1e-12))

    # argsort by negative to get descending
    return ops.argsort(ops.negative(metric_jittered))


def gptq_quantize_matrix(
    weights_transpose,
    inv_hessian,
    *,
    blocksize=128,
    per_channel=False,
    group_size=-1,
    activation_order=False,
    order_metric=None,
    compute_scale_zero=compute_quantization_parameters,
):
    """
    Returns:
      q_int:     int32, [out_features, in_features]
      scale_map: float32, broadcast-compatible with q_int
      zero_map:  float32, broadcast-compatible with q_int
    """
    out_features = ops.shape(weights_transpose)[0]
    in_features = ops.shape(weights_transpose)[1]

    # ---- optional activation-order permutation ----
    if activation_order:
        if order_metric is None:
            order_metric = ops.reciprocal(
                ops.add(ops.diagonal(inv_hessian), 1e-12)
            )
        else:
            order_metric = ops.cast(order_metric, "float32")
            order_metric = ops.where(
                ops.isfinite(order_metric),
                order_metric,
                ops.zeros_like(order_metric),
            )
        perm = _stable_permutation(order_metric)  # descending
        inv_perm = ops.argsort(perm)

        weights_transpose = ops.take(weights_transpose, perm, axis=1)
        inv_hessian = ops.take(
            ops.take(inv_hessian, perm, axis=0), perm, axis=1
        )
    else:
        perm = inv_perm = None

    # ---- outputs ----
    q_int_buffer = ops.zeros(ops.shape(weights_transpose), dtype="int32")

    if not per_channel:
        # Minimal broadcast maps for per-tensor: [1, in]
        scale_map = ops.zeros((1, in_features), dtype="float32")
        zero_map = ops.zeros((1, in_features), dtype="int32")
    else:
        # For per-channel(,grouped), params vary by (row, possibly column),
        # so we materialize full maps for correctness.
        scale_map = ops.zeros(ops.shape(weights_transpose), dtype="float32")
        zero_map = ops.zeros(ops.shape(weights_transpose), dtype="int32")

    # ---- path helpers ----
    # A) per-tensor (per_channel == False): 1 scalar per column, broadcast over rows.
    def _params_per_tensor(col2d):  # col2d: [out,1]
        # Use weight=False so compute_quantization_parameters returns a single scalar
        # for this "tensor" (the column). That scalar will broadcast over [out,1].
        s, z, maxq = compute_scale_zero(col2d)
        # Ensure shapes [1,1] explicitly
        s = ops.reshape(s, (1, 1))
        z = ops.reshape(z, (1, 1))
        return s, z, maxq

    # B) per-channel, no grouping: compute per-column per-row params (full map).
    def _params_per_channel(col2d):  # [out,1]
        s, z, maxq = compute_scale_zero(col2d)  # returns [out,1]
        return s, z, maxq

    # C) per-channel, grouped: reuse per-group params across columns in the group.
    cached = {"group_start": -1, "scale": None, "zero": None, "maxq": None}

    def _params_grouped(global_idx, weights_buf):
        group_start = (global_idx // group_size) * group_size
        if group_start != cached["group_start"]:
            group_end = ops.minimum(group_start + group_size, in_features)
            group_slice = weights_buf[:, group_start:group_end]  # [out, group_len]
            s, z, m = compute_scale_zero(group_slice)  # expect [out,1]
            cached.update(group_start=group_start, scale=s, zero=z, maxq=m)
        return cached["scale"], cached["zero"], cached["maxq"], group_start

    # ---- GPTQ sweep ----
    weights_buffer = weights_transpose
    for block_start in range(0, in_features, blocksize):
        block_end = min(block_start + blocksize, in_features)
        block_size = block_end - block_start

        block_weights = weights_buffer[:, block_start:block_end]  # [out, b]
        block_q_int = ops.zeros_like(block_weights, dtype="int32")  # [out, b]
        block_error = ops.zeros_like(block_weights, dtype="float32")  # [out, b]
        block_inv_h = inv_hessian[
            block_start:block_end, block_start:block_end
        ]  # [b, b]

        # For per-tensor, we also keep a row-vector of params for this block
        if not per_channel:
            block_scale_row = ops.zeros(
                (1, block_size), dtype="float32"
            )  # [1, b]
            block_zero_row = ops.zeros(
                (1, block_size), dtype="float32"
            )  # [1, b]

        for block_idx in range(block_size):
            j = block_start + block_idx
            col = block_weights[:, block_idx]  # [out]
            col2d = ops.expand_dims(col, 1)  # [out,1]

            # ---- pick params by path ----
            if not per_channel:
                s, z, maxq = _params_per_tensor(col2d)  # [1,1]
                # store into row-vectors at (0, block_idx)
                block_scale_row = ops.slice_update(
                    block_scale_row, (0, block_idx), s
                )  # s is [1,1]
                block_zero_row = ops.slice_update(
                    block_zero_row, (0, block_idx), z
                )
            elif group_size == -1:
                s, z, maxq = _params_per_channel(col2d)  # [out,1]
                # write the whole column into maps
                scale_map = ops.slice_update(scale_map, (0, j), s)
                zero_map = ops.slice_update(zero_map, (0, j), z)
            else:
                s, z, maxq, grp_start = _params_grouped(
                    j, weights_buffer
                )  # [out,1]
                scale_map = ops.slice_update(scale_map, (0, j), s)
                zero_map = ops.slice_update(zero_map, (0, j), z)

            # ---- quantize to int ----
            q_int_col = ops.cast(
                quantize_with_zero_point(col2d, s, z, maxq), dtype="int32"
            )  # [out,1] -> int32
            block_q_int = ops.slice_update(
                block_q_int, (0, block_idx), q_int_col
            )

            # ---- GPTQ error feedback uses dequantized col ----
            dq_col = dequantize_with_zero_point(q_int_col, s, z)[:, 0]  # [out]
            diag = block_inv_h[block_idx, block_idx]
            err = ops.divide(ops.subtract(col, dq_col), diag)  # [out]
            block_error = ops.slice_update(
                block_error, (0, block_idx), ops.expand_dims(err, 1)
            )

            if block_idx < block_size - 1:
                update = ops.matmul(
                    ops.expand_dims(err, 1),
                    ops.expand_dims(block_inv_h[block_idx, block_idx + 1 :], 0),
                )  # [out, remaining]
                tail = block_weights[:, block_idx + 1 :]
                block_weights = ops.slice_update(
                    block_weights,
                    (0, block_idx + 1),
                    ops.subtract(tail, update),
                )

        # stitch q_int
        left_q = q_int_buffer[:, :block_start]
        right_q = q_int_buffer[:, block_end:]
        q_int_buffer = ops.concatenate([left_q, block_q_int, right_q], axis=1)

        # stitch per-tensor row maps
        if not per_channel:
            left_s = scale_map[:, :block_start]
            right_s = scale_map[:, block_end:]
            scale_map = ops.concatenate(
                [left_s, block_scale_row, right_s], axis=1
            )  # [1, in]

            left_z = zero_map[:, :block_start]
            right_z = zero_map[:, block_end:]
            zero_map = ops.concatenate(
                [left_z, block_zero_row, right_z], axis=1
            )  # [1, in]

        # propagate block errors to future columns
        if block_end < in_features:
            total_update = ops.matmul(
                block_error, inv_hessian[block_start:block_end, block_end:]
            )  # [out, future]
            weights_buffer = ops.concatenate(
                [
                    weights_buffer[:, :block_end],
                    ops.subtract(weights_buffer[:, block_end:], total_update),
                ],
                axis=1,
            )

    # undo permutation
    if activation_order:
        q_int_buffer = ops.take(q_int_buffer, inv_perm, axis=1)
        scale_map = ops.take(scale_map, inv_perm, axis=1)
        zero_map = ops.take(zero_map, inv_perm, axis=1)

    return q_int_buffer, scale_map, zero_map


class GPTQ:
    def __init__(self, layer, config=GPTQConfig(tokenizer=None, dataset=None)):
        self.original_layer = layer
        self.num_samples = 0
        self.config = config
        self.quantizer = GPTQQuantizer(config)

        # Explicitly handle each supported layer type
        if isinstance(layer, Dense) or (
            isinstance(layer, EinsumDense) and layer.kernel.ndim == 2
        ):
            # For a standard Dense layer, the dimensions are straightforward.
            self.kernel_shape = layer.kernel.shape
            # rows: [input_features]
            self.rows = self.kernel_shape[0]
            # columns: [output_features]
            self.columns = self.kernel_shape[1]
            self.layer = layer

        # Handle 3D EinsumDense layers (typically from attention blocks).
        elif isinstance(layer, EinsumDense) and layer.kernel.ndim == 3:
            # For EinsumDense, we determine the effective 2D dimensions.
            self.kernel_shape = layer.kernel.shape
            shape = list(self.kernel_shape)
            try:
                d_model_dim_index = shape.index(max(shape))
            except ValueError:
                raise TypeError(
                    f"Could not determine hidden dimension from shape {shape}"
                )

            if d_model_dim_index == 0:  # QKV projection case
                in_features, heads, head_dim = shape
                self.rows, self.columns = (
                    in_features,
                    ops.multiply(heads, head_dim),
                )
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                self.rows, self.columns = (
                    ops.multiply(heads, head_dim),
                    out_features,
                )

            # Create a temporary object that holds a reshaped
            # 2D version of the kernel.
            self.layer = types.SimpleNamespace(
                kernel=ops.reshape(layer.kernel, (self.rows, self.columns)),
            )

        else:
            # Raise an error if the layer is not supported.
            raise TypeError(f"Unsupported layer type for GPTQ: {type(layer)}")
        self.hessian = ops.zeros((self.rows, self.rows), dtype="float32")

    def update_hessian_with_batch(self, input_batch):
        """
        Updates the running average of the Hessian matrix with a new batch.

        This method computes the Hessian matrix for a given batch of input
        activations and updates the accumulated Hessian (`self.hessian`) using a
        numerically stable running average. This allows the Hessian to be
        computed over a large dataset without loading all samples into memory
        at once.

        The input tensor is first reshaped into a 2D matrix [num_samples,
        num_features] before the Hessian is calculated.

        Args:
            input_batch: A 2D or higher-dimensional tensor of input activations
                from a calibration batch.

        Raises:
            ValueError: If the feature dimension of the input tensor
                `input_batch` does not match the dimensions of the
                pre-initialized Hessian matrix `self.hessian`.
        """
        if input_batch is None:
            raise ValueError("Input tensor cannot be None.")
        if len(input_batch.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(input_batch.shape)})."
            )
        if ops.size(input_batch) == 0:
            raise ValueError("Input tensor cannot be empty.")

        if len(input_batch.shape) > 2:
            # [batch, features]
            input_batch = ops.reshape(input_batch, (-1, input_batch.shape[-1]))
        x = ops.cast(input_batch, "float32")

        num_new_samples = ops.shape(x)[0]
        num_prev_samples = self.num_samples
        total_samples = ops.add(num_prev_samples, num_new_samples)

        if ops.shape(self.hessian)[0] != ops.shape(x)[-1]:
            raise ValueError(
                f"Hessian dimensions ({ops.shape(self.hessian)[0]}) do not "
                f"match input features ({ops.shape(x)[-1]})."
            )

        # gram_matrix: [features, features]
        gram_matrix = ops.matmul(ops.transpose(x), x)
        # Ensures numerical stability and symmetry in case of large floating
        # point activations.
        gram_matrix = ops.divide(
            ops.add(gram_matrix, ops.transpose(gram_matrix)), 2.0
        )

        # Decay previous mean and add current per-sample contribution
        # (factor 2/N)
        if self.num_samples > 0:
            self.hessian = ops.multiply(
                self.hessian, ops.divide(num_prev_samples, total_samples)
            )
        self.hessian = ops.add(
            self.hessian,
            ops.multiply(ops.divide(2.0, total_samples), gram_matrix),
        )

        self.num_samples = self.num_samples + ops.shape(x)[0] or 0

    def quantize_and_correct_layer(
        self,
        blocksize=128,
    ):
        """
        Performs GPTQ quantization and correction on the layer's weights.

        This method implements the core logic of the "Optimal Brain Quant"
        (OBQ) method, as applied by GPTQ, to quantize the weights of a single
        layer. It iteratively quantizes blocks of weights and corrects for the
        quantization error by updating the remaining weights.

        The algorithm follows these main steps:
        1.  Initialization: It optionally reorders the weight columns based
            on activation magnitudes (`activation_order=True`) to protect more
            salient
            weights.
        2.  Hessian Modification: The Hessian matrix, pre-computed from
            calibration data, is dampened to ensure its invertibility and
            stability.
        3.  Iterative Quantization: The function iterates through the
            weight columns in blocks (`blocksize`). In each iteration, it:
            a. Quantizes one column.
            b. Calculates the quantization error.
            c. Updates the remaining weights in the *current* block by
                distributing the error, using the inverse Hessian.
        4.  Block-wise Correction: After a block is quantized, the total
            error from that block is propagated to the *next* block of weights
            to be processed.
        5.  Finalization: The quantized weights are reordered back if
            `activation_order` was used, and the layer's weights are updated.

        This implementation is based on the official GPTQ paper and repository.
        For more details, see:
        - Paper: https://arxiv.org/abs/2210.17323
        - Original Code: https://github.com/IST-DASLab/gptq

        Args:
            blocksize: (int, optional) The size of the weight block to process
             at a time. Defaults to 128.
        """

        weights_matrix = ops.transpose(self.layer.kernel)
        orig_wts = ops.copy(weights_matrix)

        # Dampen the Hessian for Stability
        hessian_diagonal = ops.diagonal(self.hessian)
        dead_diagonal = ops.equal(hessian_diagonal, 0.0)
        hessian_diagonal = ops.where(dead_diagonal, 1.0, hessian_diagonal)
        hessian_matrix = ops.add(
            self.hessian,
            ops.diag(
                ops.where(dead_diagonal, 1.0, ops.zeros_like(hessian_diagonal))
            ),
        )

        # Add dampening factor to the Hessian diagonal
        damping_factor = ops.multiply(
            self.config.hessian_damping, ops.mean(hessian_diagonal)
        )
        hessian_diagonal = ops.add(hessian_diagonal, damping_factor)
        hessian_matrix = ops.add(
            ops.subtract(
                hessian_matrix, ops.diag(ops.diagonal(hessian_matrix))
            ),
            ops.diag(hessian_diagonal),
        )

        # Compute the inverse Hessian, which is used for error correction
        inverse_hessian = linalg.inv(hessian_matrix)

        quantized_weights, scale_map, zero_map = gptq_quantize_matrix(
            weights_matrix,
            inv_hessian=inverse_hessian,
            blocksize=blocksize,
            group_size=self.config.group_size,
            activation_order=self.config.activation_order,
            per_channel=self.config.per_channel,
            order_metric=ops.diagonal(hessian_matrix),
            compute_scale_zero=partial(
                compute_quantization_parameters,
                bits=self.config.weight_bits,
                symmetric=self.config.symmetric,
                per_channel=self.config.per_channel,
                group_size=self.config.group_size,
                weight=True
            ),
        )

        flattened_view_err = ops.mean(
                # ops.abs(
                    ops.subtract(
                        dequantize_with_zero_point(
                            quantized_weights, scale_map, zero_map
                        ),
                        orig_wts,
                    )
                # )
            ).numpy().item() # type: ignore

        # passes
        assert (
            flattened_view_err < 1e-1
        ), f"Flattened quantization error after GPTQ is too high: {flattened_view_err}"

        reshaped_view_err = (
            ops.mean(
                # ops.abs(
                    ops.subtract(
                        ops.reshape(
                            ops.transpose(
                                dequantize_with_zero_point(
                                    quantized_weights, scale_map, zero_map
                                )
                            ),
                            self.kernel_shape,
                        ),
                        self.original_layer.kernel,
                    )
                # )
            )
            .numpy() # type: ignore
            .item() # type: ignore
        )

        # passes
        assert (
            reshaped_view_err < 1e-1
        ), f"Reshaped quantization error after GPTQ is too high: {reshaped_view_err}"

        # passes
        assert (
            flattened_view_err - reshaped_view_err < 1e-5
        ), f"Quantization errors do not match: {flattened_view_err=} vs {reshaped_view_err=}"

        # Set the new quantized weights in the original layer (in flat shape)
        self.original_layer._is_quantized = True
        self.original_layer.quantized_kernel.assign(quantized_weights)
        # del self.original_layer._kernel
        self.original_layer.kernel_scale.assign(scale_map)
        self.original_layer.kernel_zero.assign(zero_map)

        # Sanity check to ensure the above assertions still work.

        flattened_view_err = (
            ops.mean(
                # ops.abs(
                    ops.subtract(
                        dequantize_with_zero_point(
                            self.original_layer.quantized_kernel,
                            self.original_layer.kernel_scale,
                            self.original_layer.kernel_zero,
                        ),
                        orig_wts,
                    )
                # )
            )
            .numpy() # type: ignore
            .item() # type: ignore
        )  # type: ignore

        # passes
        assert (
            flattened_view_err < 1e-1
        ), f"Flattened quantization error after GPTQ is too high: {flattened_view_err}"

        reshaped_view_err = (
            ops.mean(
                # ops.abs(
                    ops.subtract(
                        ops.reshape(
                            ops.transpose(
                                dequantize_with_zero_point(
                                    self.original_layer.quantized_kernel,
                                    self.original_layer.kernel_scale,
                                    self.original_layer.kernel_zero,
                                )
                            ),
                            self.kernel_shape,
                        ),
                        self.original_layer.kernel,
                    )
                # )
            )
            .numpy()  # type: ignore
            .item()  # type: ignore
        )

        # passes
        assert (
            reshaped_view_err < 1e-1
        ), f"Reshaped quantization error after GPTQ is too high: {reshaped_view_err}"

        # passes
        assert (
            flattened_view_err - reshaped_view_err < 1e-5
        ), f"Quantization errors do not match: {flattened_view_err=} vs {reshaped_view_err=}"

    def free(self):
        self.hessian = None
