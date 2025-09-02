import types

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
    group_size=-1,
    activation_order=False,
    order_metric=None,
    compute_scale_zero=compute_quantization_parameters,
    original_kernel_shape=None,
):
    """
    Implements GPTQ error-correction and returns INT8 weights with row-wise scales/zeros.

    Returns:
        q_int8:  int8 weights, shape [out_features, in_features]
        scale2d: float32 scales, shape [out_features, in_features] (row-wise broadcast)
        zero2d:  float32 zeros,  shape [out_features, in_features] (row-wise broadcast)
    """
    in_features = ops.shape(weights_transpose)[1]

    # Optional activation-order permutation on feature axis (axis=1)
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

        perm = _stable_permutation(order_metric)
        inv_perm = ops.argsort(perm)

        weights_transpose = ops.take(weights_transpose, perm, axis=1)
        inv_hessian = ops.take(
            ops.take(inv_hessian, perm, axis=0), perm, axis=1
        )
    else:
        perm = inv_perm = None

    # === Buffers in [out_features, in_features] layout ===
    weights_buffer = weights_transpose
    # ### NEW: int8 kernel buffer
    q_int8_buffer = ops.zeros_like(weights_buffer, dtype="int8")  # [C, K]
    # ### NEW: 2-D scale/zero buffers (float32), row-wise but broadcast across columns
    scale2d = ops.zeros_like(weights_buffer, dtype="float32")  # [C, K]
    zero2d = ops.zeros_like(weights_buffer, dtype="float32")  # [C, K]

    # If per-channel only, compute once and broadcast everywhere
    cached_scale_row = None  # ### NEW: row vector [C]
    cached_zero_row = None  # ### NEW: row vector [C]
    cached_maxq = None
    if group_size == -1:
        # ### NEW: one set for entire matrix, row-wise (per output channel)
        s, z, maxq = compute_scale_zero(
            weights_buffer, weight=True
        )  # s,z shapes typically [1, C]
        s_row = ops.cast(s, "float32")[0, :]  # [C]
        z_row = ops.cast(z, "float32")[0, :]  # [C]
        # broadcast across all columns
        scale2d = ops.expand_dims(s_row, 1) + ops.zeros_like(
            weights_buffer, dtype="float32"
        )  # [C, K]
        zero2d = ops.expand_dims(z_row, 1) + ops.zeros_like(
            weights_buffer, dtype="float32"
        )  # [C, K]
        cached_scale_row = s_row
        cached_zero_row = z_row
        cached_maxq = maxq

    for block_start in range(0, in_features, blocksize):
        block_end = min(block_start + blocksize, in_features)
        block_size = block_end - block_start

        block_weights = ops.cast(
            weights_buffer[:, block_start:block_end], "float32"
        )  # [C, B]
        block_error = ops.zeros_like(block_weights, dtype="float32")  # [C, B]
        block_invH = inv_hessian[
            block_start:block_end, block_start:block_end
        ]  # [B, B]

        # Group caches
        cached_group_start = -1

        for block_idx in range(block_size):
            global_idx = block_start + block_idx
            w_col = block_weights[:, block_idx]  # [C]

            # ### CHANGED: compute/fill scale/zero once per group (if grouped)
            if group_size != -1:
                group_start = (global_idx // group_size) * group_size
                if group_start != cached_group_start:
                    group_end = min(group_start + group_size, in_features)
                    group_slice = weights_buffer[
                        :, group_start:group_end
                    ]  # [C, G]
                    s, z, maxq = compute_scale_zero(
                        group_slice, weight=True
                    )  # s,z ~ [1, C]
                    s_row = ops.cast(s, "float32")[0, :]  # [C]
                    z_row = ops.cast(z, "float32")[0, :]  # [C]
                    # broadcast row-wise into the group's columns
                    tgt = weights_buffer[:, group_start:group_end]  # [C, G]
                    sblk = ops.expand_dims(s_row, 1) + ops.zeros_like(
                        tgt, dtype="float32"
                    )  # [C, G]
                    zblk = ops.expand_dims(z_row, 1) + ops.zeros_like(
                        tgt, dtype="float32"
                    )  # [C, G]
                    scale2d = ops.slice_update(
                        scale2d, (0, group_start), sblk
                    )  # fill once per group
                    zero2d = ops.slice_update(zero2d, (0, group_start), zblk)
                    cached_scale_row = s_row
                    cached_zero_row = z_row
                    cached_maxq = maxq
                    cached_group_start = group_start

                # params for this column come from the current group's row vectors
                s_use = cached_scale_row  # [C]
                z_use = cached_zero_row  # [C]
                maxq = cached_maxq
            else:
                # per-channel: we already filled scale2d/zero2d; reuse the global row vectors
                s_use = cached_scale_row  # [C]
                z_use = cached_zero_row  # [C]
                maxq = cached_maxq

            # ### NEW: quantize to INT8, store INT8; dequantize only for error feedback
            q_col_int = quantize_with_zero_point(
                ops.expand_dims(w_col, 1), s_use, z_use, maxq
            )[:, 0]  # [C]
            q_int8_buffer = ops.slice_update(
                q_int8_buffer,
                (0, global_idx),
                ops.expand_dims(ops.cast(q_col_int, "int8"), 1),
            )

            # dequantized column for error:
            dq_col = dequantize_with_zero_point(
                ops.expand_dims(q_col_int, 1), s_use, z_use
            )[:, 0]  # [C]

            # error feedback within the block
            invH_ii = block_invH[block_idx, block_idx]
            err = ops.divide(ops.subtract(w_col, dq_col), invH_ii)  # [C]
            block_error = ops.slice_update(
                block_error, (0, block_idx), ops.expand_dims(err, 1)
            )

            if block_idx < block_size - 1:
                update = ops.matmul(
                    ops.expand_dims(err, 1),
                    ops.expand_dims(block_invH[block_idx, block_idx + 1 :], 0),
                )  # [C, B-1]
                tail = ops.cast(block_weights[:, block_idx + 1 :], "float32")
                block_weights = ops.slice_update(
                    block_weights,
                    (0, block_idx + 1),
                    ops.subtract(tail, update),
                )

        # Propagate block errors to future columns
        if block_end < in_features:
            total_update = ops.matmul(
                block_error, inv_hessian[block_start:block_end, block_end:]
            )  # [C, K-rest]
            weights_buffer = ops.concatenate(
                [
                    weights_buffer[:, :block_end],
                    ops.subtract(weights_buffer[:, block_end:], total_update),
                ],
                axis=1,
            )

    # Undo activation-order permutation for ALL outputs
    if activation_order:
        q_int8_buffer = ops.take(q_int8_buffer, inv_perm, axis=1)  # ### NEW
        scale2d = ops.take(scale2d, inv_perm, axis=1)  # ### NEW
        zero2d = ops.take(zero2d, inv_perm, axis=1)  # ### NEW

    # ---- NEW: transpose back to [in_features, out_features] ----
    qK_C = ops.transpose(q_int8_buffer)  # [K, C], int8
    sK_C = ops.transpose(scale2d)  # [K, C], float32
    zK_C = ops.transpose(zero2d)  # [K, C], float32

    # ---- NEW: reshape to the layer's original kernel shape ----
    if original_kernel_shape is None:
        # Fallback: keep 2-D KxC shape
        q_kernel = qK_C
        scale_kernel = sK_C
        zero_kernel = zK_C
    else:
        q_kernel = ops.reshape(qK_C, original_kernel_shape)
        scale_kernel = ops.reshape(sK_C, original_kernel_shape)
        zero_kernel = ops.reshape(zK_C, original_kernel_shape)

    return q_kernel, scale_kernel, zero_kernel


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
                bias=layer.bias,
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

        qC_K, sC_K, zC_K = gptq_quantize_matrix(
            weights_matrix,
            inv_hessian=inverse_hessian,
            blocksize=blocksize,
            group_size=self.config.group_size,
            activation_order=self.config.activation_order,
            order_metric=ops.diagonal(hessian_matrix),
            compute_scale_zero=self.quantizer.find_params,
            original_kernel_shape=self.kernel_shape,
        )

        # quantized_weights = ops.transpose(qC_K)

        # if isinstance(self.original_layer, EinsumDense):
        #     quantized_weights = ops.reshape(
        #         quantized_weights, self.kernel_shape
        #     )

        # Set the new quantized weights in the original layer
        self.original_layer.qptq_initialized = True
        self.original_layer.quantized_kernel.assign(qC_K)
        del self.original_layer._kernel
        self.original_layer.kernel_scale.assign(sC_K)
        self.original_layer.kernel_zero.assign(zC_K)

    def free(self):
        self.hessian = None
