import types

from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.gptq_quant import GPTQQuantizer


class GPTQ:
    def __init__(self, layer, config):
        self.original_layer = layer
        self.num_samples = 0
        self.quantizer = GPTQQuantizer()
        self.config = config

        # Explicitly handle each supported layer type
        if isinstance(layer, Dense) or (
            isinstance(layer, EinsumDense) and layer.kernel.ndim == 2
        ):
            # For a standard Dense layer, the dimensions are straightforward.
            # kernel_shape: [input_features, output_features]
            self.kernel_shape = layer.kernel.shape
            # rows: number of input features
            self.rows = self.kernel_shape[0]
            # columns: number of output features
            self.columns = self.kernel_shape[1]
            self.layer = layer

        # Handle 3D EinsumDense layers (typically from attention blocks).
        elif isinstance(layer, EinsumDense) and layer.kernel.ndim == 3:
            # For EinsumDense, we determine the effective 2D dimensions.
            # kernel_shape: e.g., [in_features, heads, head_dim] or [heads, head_dim, out_features]
            self.kernel_shape = layer.kernel.shape
            try:
                # d_model_dim_index: index of the largest dimension, assumed to be the model's hidden size
                d_model_dim_index = ops.argmax(self.kernel_shape)
            except ValueError:
                raise TypeError(
                    f"Could not determine hidden dimension from shape {self.kernel_shape}"
                )

            if d_model_dim_index == 0:  # QKV projection case
                # in_features: e.g., 768
                # heads: e.g., 12
                # head_dim: e.g., 64
                in_features, heads, head_dim = self.kernel_shape
                # rows: number of input features
                self.rows, self.columns = (
                    in_features,
                    ops.multiply(heads, head_dim),
                )
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                # heads: e.g., 12
                # head_dim: e.g., 64
                # out_features: e.g., 768
                heads, head_dim, out_features = self.kernel_shape
                # rows: effective number of input features (heads * head_dim)
                # columns: number of output features
                self.rows, self.columns = (
                    ops.multiply(heads, head_dim),
                    out_features,
                )

            # Work on a 2D view (rows=in_features, cols=out_features) via transpose later
            self.layer = types.SimpleNamespace(
                # kernel: [rows, columns]
                kernel=ops.reshape(layer.kernel, (self.rows, self.columns)),
                bias=layer.bias,
            )

        else:
            # Raise an error if the layer is not supported.
            raise TypeError(f"Unsupported layer type for GPTQ: {type(layer)}")
        # hessian: [rows, rows] or [input_features, input_features]
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
                shape: [batch_size, ..., features]

        Raises:
            ValueError: If the feature dimension of the input tensor
                `input_batch` does not match the dimensions of the
                pre-initialized Hessian matrix `self.hessian`.
        """
        if input_batch is None:
            raise ValueError("Input tensor 'input_batch' cannot be None.")

        if len(input_batch.shape) < 2:
            raise ValueError(
                f"Input tensor 'input_batch' must have a rank of at least 2 "
                f"(e.g., [batch, features]), but got rank "
                f"{len(input_batch.shape)}."
            )
        if ops.size(input_batch) == 0:
            raise ValueError("Input tensor 'input_batch' cannot be empty.")

        if len(input_batch.shape) > 2:
            # input_batch: [total_samples, features]
            input_batch = ops.reshape(input_batch, (-1, input_batch.shape[-1]))
        input_batch = ops.cast(input_batch, "float32")

        if self.hessian.shape[0] != input_batch.shape[-1]:
            raise ValueError(
                f"Hessian dimensions ({self.hessian.shape[0]}) do not "
                f"match input features ({input_batch.shape[-1]})."
            )

        # current_hessian: 2 * X^T X
        # [features, features] or [rows, rows]
        current_hessian = ops.multiply(
            2, ops.matmul(ops.transpose(input_batch), input_batch)
        )

        if self.num_samples == 0:
            self.hessian = current_hessian
        else:
            # total_samples: scalar
            total_samples = ops.add(self.num_samples, input_batch.shape[0])
            # old_hessian_weight: scalar
            old_hessian_weight = ops.divide(self.num_samples, total_samples)
            # current_hessian_weight: scalar
            current_hessian_weight = ops.divide(
                input_batch.shape[0], total_samples
            )

            # Update the accumulated Hessian
            # old_term: [rows, rows]
            old_term = ops.multiply(self.hessian, old_hessian_weight)
            # current_term: [rows, rows]
            current_term = ops.multiply(current_hessian, current_hessian_weight)
            # self.hessian: [rows, rows]
            self.hessian = ops.add(old_term, current_term)

        # self.num_samples: scalar
        self.num_samples = ops.add(self.num_samples, input_batch.shape[0])

    def quantize_and_correct_block(
        self,
        blocksize=128,
        hessian_damping=0.01,
        group_size=-1,
        activation_order=False,
    ):
        """
        Performs GPTQ quantization and correction on the layer's weights.

        This method implements the core logic of the "Optimal Brain Quant"
        (OBQ) method, as applied by GPTQ, to quantize the weights of a single
        layer. It iteratively quantizes blocks of weights and corrects for the
        quantization error by updating the remaining weights.

        The algorithm follows these main steps:
        1.  **Initialization**: It optionally reorders the weight columns based
            on activation magnitudes (`activation_order=True`) to protect more
            salient weights.
        2.  **Hessian Modification**: The Hessian matrix, pre-computed from
            calibration data, is dampened to ensure its invertibility and
            stability.
        3.  **Iterative Quantization**: The function iterates through the
            weight columns in blocks (`blocksize`). In each iteration, it:
            a. Quantizes one column.
            b. Calculates the quantization error.
            c. Updates the remaining weights in the *current* block by
                distributing the error, using the inverse Hessian.
        4.  **Block-wise Correction**: After a block is quantized, the total
            error from that block is propagated to the *next* block of weights
            to be processed.
        5.  **Finalization**: The quantized weights are reordered back if
            `activation_order` was used, and the layer's weights are updated.

        This implementation is based on the official GPTQ paper and repository.
        For more details, see:
        - Paper: https://arxiv.org/abs/2210.17323
        - Original Code: https://github.com/IST-DASLab/gptq

        Args:
            blocksize: (int, optional) The size of the weight block to process
             at a time. Defaults to 128.
            hessian_damping: (float, optional) The percentage of dampening to
                add the
                Hessian's diagonal. A value of 0.01 is recommended.
                Defaults to 0.01.
            group_size: (int, optional) The number of weights that share the
                same quantization parameters (scale and zero-point).
                A value of -1 indicates per-channel quantization.
            activation_order: (bool, optional) If True, reorders weight columns
                based
                on their activation's second-order information.
        """

        self.original_layer.quantize("gptq", config=self.config)

        # weights_matrix: [columns, rows] = [out_features, in_features]
        weights_matrix = ops.transpose(ops.cast(self.layer.kernel, "float32"))
        # hessian_matrix: [rows, rows] = [in_features, in_features]
        hessian_matrix = ops.cast(self.hessian, "float32")

        if activation_order:
            # Sort indices by negative Hessian diagonal (descending importance)
            # permutation: [rows]
            permutation = ops.argsort(
                ops.negative(ops.diagonal(hessian_matrix))
            )

            # Apply permutation to weights and Hessian
            # weights_matrix: [columns, rows] (columns permuted)
            weights_matrix = ops.take(weights_matrix, permutation, axis=1)
            # hessian_matrix: [rows, rows] (rows and columns permuted)
            hessian_matrix = ops.take(hessian_matrix, permutation, axis=0)
            hessian_matrix = ops.take(hessian_matrix, permutation, axis=1)

            # Store inverse permutation for later restoration
            # inverse_permutation: [rows]
            inverse_permutation = ops.argsort(permutation)

        # Dampen the Hessian for Stability
        # hessian_diagonal: [rows]
        hessian_diagonal = ops.diagonal(hessian_matrix)

        # Detect zero entries on the diagonal
        # dead_diagonal: [rows] (boolean)
        dead_diagonal = ops.equal(hessian_diagonal, 0.0)

        # Replace zeros in the diagonal with ones
        hessian_diagonal = ops.where(dead_diagonal, 1.0, hessian_diagonal)

        # Update Hessian diagonal in-place by adding stabilizing values
        # stabilizer: [rows]
        stabilizer = ops.where(
            dead_diagonal, 1.0, ops.zeros_like(hessian_diagonal)
        )
        # hessian_matrix: [rows, rows]
        hessian_matrix = ops.add(hessian_matrix, ops.diag(stabilizer))

        # Scale damping by the average diagonal value
        # damping_factor: scalar
        damping_factor = ops.multiply(
            hessian_damping, ops.mean(hessian_diagonal)
        )

        # Add damping to the diagonal
        # hessian_diagonal: [rows]
        hessian_diagonal = ops.add(hessian_diagonal, damping_factor)

        # Replace the old diagonal in the Hessian with the damped version
        # hessian_off_diag: [rows, rows]
        hessian_off_diag = ops.subtract(
            hessian_matrix, ops.diag(ops.diagonal(hessian_matrix))
        )
        # hessian_matrix: [rows, rows]
        hessian_matrix = ops.add(hessian_off_diag, ops.diag(hessian_diagonal))

        # Compute the inverse Hessian, which is used for error correction
        # inverse_hessian: [rows, rows]
        inverse_hessian = ops.linalg.inv(hessian_matrix)

        # quantized_weights_int: [columns, rows]
        quantized_weights_int = ops.zeros_like(weights_matrix, dtype="int8")
        # We'll store row-wise scales/zeros and broadcast to columns (groups)
        # shapes below mirror weights_matrix to keep assignment simple
        # scales: [columns, rows]
        scales = ops.zeros_like(weights_matrix, dtype="float32")
        # zeros: [columns, rows]
        zeros = ops.zeros_like(weights_matrix, dtype="float32")

        # If per-channel only, compute once and broadcast everywhere (like ref)
        precomputed_scale = None
        precomputed_zero = None
        if group_size == -1:
            # One set of row-wise params over full weights_matrix
            self.quantizer.find_params(weights_matrix, weight=True)  # weights_matrix is [columns, rows]
            
            # quantizer.scale/zero are row-wise (length = columns = out_features)
            precomputed_scale = ops.cast(self.quantizer.scale, "float32")[0, :]  # [columns]
            precomputed_zero = ops.cast(self.quantizer.zero, "float32")[0, :]    # [columns]
            
            # Broadcast across all columns (rows = input_features)
            scales = ops.add(ops.expand_dims(precomputed_scale, 1), ops.zeros_like(weights_matrix))
            zeros = ops.add(ops.expand_dims(precomputed_zero, 1), ops.zeros_like(weights_matrix))

        # Blockwise OBQ pass on columns of K (rows = in_features)
        for block_start in range(0, self.rows, blocksize):
            block_end = min(ops.add(block_start, blocksize), self.rows)
            # block_size: scalar
            block_size = ops.subtract(block_end, block_start)

            # Extract current weight block and its inverse-Hessian submatrix
            # block_weights: [columns, block_size]
            block_weights = weights_matrix[:, block_start:block_end]
            # block_errors: [columns, block_size]
            block_errors = ops.zeros_like(block_weights)
            # block_inverse_hessian: [block_size, block_size]
            block_inverse_hessian = inverse_hessian[
                block_start:block_end, block_start:block_end
            ]

            # Process one column at a time within the block
            for col_idx in range(block_size):
                # Absolute/relative indices
                # abs_col: scalar
                abs_col = ops.add(block_start, col_idx)

                # Group-wise params: recompute once per group boundary and fill the group's columns
                if group_size != -1 and ops.equal(ops.mod(abs_col, group_size), 0):
                    g_start = abs_col
                    g_end = ops.min(ops.add(g_start, group_size), self.rows)
                    # [columns, group_size]
                    group_slice = weights_matrix[:, g_start:g_end]
                    self.quantizer.find_params(group_slice, weight=True)
                    # g_scale: [columns]
                    g_scale = ops.cast(self.quantizer.scale, "float32")[0, :]
                    # g_zero: [columns]
                    g_zero = ops.cast(self.quantizer.zero, "float32")[0, :]
                    # Broadcast row-wise vector across the group's columns
                    # target: [columns, group_size]
                    target = weights_matrix[:, g_start:g_end]
                    # scale_block: [columns, group_size]
                    scale_block = ops.expand_dims(g_scale, 1) + ops.zeros_like(target)
                    # zero_block: [columns, group_size]
                    zero_block = ops.expand_dims(g_zero, 1) + ops.zeros_like(target)
                    scales = ops.slice_update(scales, (0, g_start), scale_block)
                    zeros = ops.slice_update(zeros, (0, g_start), zero_block)

                # Current column and the corresponding (i,i) of inv(H)
                # weight_column: [columns]
                weight_column = block_weights[:, col_idx]
                # invH_ii: scalar
                invH_ii = block_inverse_hessian[col_idx, col_idx]
                quantized_column = self.quantizer.quantize(
                    ops.expand_dims(weight_column, 1)
                )[:, 0]

                # Write integer weights
                quantized_weights_int = ops.slice_update(
                    quantized_weights_int,
                    (0, abs_col),
                    # shape: [columns, 1]
                    ops.expand_dims(
                        ops.cast(quantized_column, "int8"),
                        axis=1,
                    ),
                )

                # Error & in-block rank-1 update
                # Dequantize back to float32 for error correction.
                # dequantized_column: [columns]
                dequantized_column = self.quantizer.dequantize(
                    ops.expand_dims(quantized_column, 1),
                )[:, 0]

                # quantization_error: [columns]
                quantization_error = ops.divide(
                    ops.subtract(weight_column, dequantized_column),
                    invH_ii,
                )
                block_errors = ops.slice_update(
                    block_errors,
                    (0, col_idx),
                    # shape: [columns, 1]
                    ops.expand_dims(quantization_error, axis=1),
                )

                has_future = ops.less(col_idx, ops.subtract(block_size, 1))
                if has_future:
                    # next_start: scalar
                    next_start = ops.add(col_idx, 1)

                    # rank-1 update: q_error * invH[i, i+1:]
                    # invH_row_future: [block_size - 1 - col_idx]
                    invH_row_future = block_inverse_hessian[
                        col_idx, next_start:
                    ]

                    # error_update: [columns, block_size - 1 - col_idx]
                    error_update = ops.matmul(
                        # shape: [columns, 1]
                        ops.expand_dims(quantization_error, 1),
                        # shape: [1, block_size - 1 - col_idx]
                        ops.expand_dims(invH_row_future, 0),
                    )

                    # Efficiently update the remaining part of the
                    # block_weights tensor.
                    # slice_to_update: [columns, block_size - 1 - col_idx]
                    slice_to_update = block_weights[:, next_start:]
                    # updated_slice: [columns, block_size - 1 - col_idx]
                    updated_slice = ops.subtract(slice_to_update, error_update)
                    block_weights = ops.slice_update(
                        block_weights, (0, next_start), updated_slice
                    )

            # Propagate accumulated block errors to the remaining columns
            # (to the right of the block)
            if block_end < self.rows:
                # right_invH: [block_size, rows - block_end]
                right_invH = inverse_hessian[block_start:block_end, block_end:]
                # total_error_update: [columns, rows - block_end]
                total_error_update = ops.matmul(block_errors, right_invH)

                # weights_matrix[:, block_end:] -= total_error_update
                # left: [columns, block_end]
                left = weights_matrix[:, :block_end]
                # right: [columns, rows - block_end]
                right = ops.subtract(
                    weights_matrix[:, block_end:], total_error_update
                )
                # weights_matrix: [columns, rows]
                weights_matrix = ops.concatenate([left, right], axis=1)

        # Undo activation order if used
        if activation_order:
            # Reorder back to the original arrangement
            # quantized_weights_int: [columns, rows]
            quantized_weights_int = ops.take(
                quantized_weights_int, inverse_permutation, axis=1
            )
            # scales: [columns, rows]
            scales = ops.take(scales, inverse_permutation, axis=1)
            # zeros: [columns, rows]
            zeros = ops.take(zeros, inverse_permutation, axis=1)

        # Transpose back to original layout [in_features, out_features]
        # quantized_kernel: [rows, columns]
        quantized_kernel = ops.transpose(quantized_weights_int)
        # scale: [rows, columns]
        scale = ops.transpose(scales)
        # zero_point: [rows, columns]
        zero_point = ops.transpose(zeros)

        # NOTE:
        # We deliberately avoid any EinsumDense "reduction" of scale/zero.
        # We keep kernel_scale/zero_point broadcastable to the flattened 2-D view.
        # For EinsumDense, we only reshape the *kernel* back to its original N-D shape.
        if isinstance(self.original_layer, EinsumDense) and self.kernel_shape != quantized_kernel.shape:
            quantized_kernel = ops.reshape(quantized_kernel, self.kernel_shape)
            # Keep scale/zero_point as 2-D (rows x columns) to match matmul layout.

        # Assign final tensors (kernel is int8; scale/zero are float, row-wise broadcast across K)
        self.original_layer.kernel_scale.assign(scale)
        self.original_layer.zero_point.assign(zero_point)
        self.original_layer._kernel.assign(quantized_kernel)

    def free(self):
        self.hessian = None
