"""CUDA kernels for optimized quantized operations.

This module provides optimized CUDA kernels for quantized matrix multiplication,
primarily targeting GPTQ and AWQ quantized models.

The main component is the Marlin kernel, which achieves near-ideal ~4x speedups
for FP16xINT4 matrix multiplication on NVIDIA GPUs with compute capability 8.0+
(Ampere, Ada Lovelace, Hopper architectures).

Usage:
    >>> from keras.src.quantizers.cuda_kernels import dispatch
    >>> if dispatch.should_use_cuda_kernels():
    ...     from keras.src.quantizers.cuda_kernels import marlin_ops
    ...     output = marlin_ops.gptq_matmul(inputs, weights, scales, zeros, g_idx)
    ... else:
    ...     # Fall back to standard implementation
    ...     pass

Submodules:
    dispatch: Kernel availability detection and dispatch logic
    marlin: Low-level Marlin CUDA kernel interface
    marlin_ops: High-level operations for Keras layers

Credits:
    The Marlin kernel is derived from IST-DASLab/marlin (Apache-2.0 license).
    See marlin/CREDITS.md for full attribution and citation information.
"""

from keras.src.quantizers.cuda_kernels.dispatch import is_marlin_available
from keras.src.quantizers.cuda_kernels.dispatch import reset_cache
from keras.src.quantizers.cuda_kernels.dispatch import should_use_cuda_kernels
