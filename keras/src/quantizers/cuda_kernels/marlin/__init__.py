"""Marlin FP16xINT4 CUDA kernel for quantized matrix multiplication.

This kernel is derived from IST-DASLab/marlin (Apache-2.0 license).
See CREDITS.md for attribution and citation information.

The Marlin kernel achieves near-ideal ~4x speedups for FP16xINT4 matrix
multiplication on NVIDIA GPUs with compute capability 8.0+ (Ampere and newer).

Example usage:
    >>> from keras.src.quantizers.cuda_kernels import marlin
    >>> if marlin.is_available():
    ...     output = torch.empty((M, N), dtype=torch.float16, device="cuda")
    ...     workspace = torch.zeros(N // 128 * max_par, dtype=torch.int32,
    ...                             device="cuda")
    ...     marlin.mul(A, B, output, scales, workspace)
"""

import os

# Lazy-loaded CUDA extension
_marlin_cuda = None
_load_attempted = False
_load_error = None


def _load_marlin():
    """Lazily load the Marlin CUDA extension.

    The extension is compiled on first import using PyTorch's JIT compiler.
    Compilation requires CUDA toolkit and may take a few minutes on first run.

    Returns:
        The compiled CUDA extension module, or None if loading fails.

    Raises:
        ImportError: If PyTorch is not available.
    """
    global _marlin_cuda, _load_attempted, _load_error

    if _load_attempted:
        if _load_error:
            raise _load_error
        return _marlin_cuda

    _load_attempted = True

    try:
        import torch
        from torch.utils.cpp_extension import load

        cuda_dir = os.path.dirname(__file__)
        _marlin_cuda = load(
            name="marlin_cuda",
            sources=[
                os.path.join(cuda_dir, "marlin_cuda.cpp"),
                os.path.join(cuda_dir, "marlin_cuda_kernel.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _marlin_cuda
    except Exception as e:
        _load_error = ImportError(
            f"Failed to load Marlin CUDA kernel: {e}. "
            "Ensure CUDA toolkit is installed and PyTorch has CUDA support."
        )
        raise _load_error


def is_available():
    """Check if Marlin CUDA kernel is available.

    Returns:
        True if the kernel can be loaded successfully, False otherwise.
    """
    global _load_attempted, _load_error

    if _load_attempted:
        return _load_error is None

    try:
        _load_marlin()
        return True
    except (ImportError, OSError):
        return False


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 matrix multiplication.

    Performs C = A @ dequantize(B) where B contains packed 4-bit weights.

    Args:
        A: Input activations tensor [M, K] in FP16.
        B: Packed 4-bit weights tensor [K/16, N*16/8] in INT32.
            The weights must be in Marlin's specific packed format.
        C: Output tensor [M, N] in FP16. Must be pre-allocated.
        s: Quantization scales tensor [K/groupsize, N] in FP16.
            Use shape [1, N] for per-column quantization (groupsize=-1).
        workspace: Barrier workspace tensor of at least [N/128 * max_par]
            elements in INT32. Used for synchronization between threadblocks.
        thread_k: Thread tile size for K dimension. -1 for auto-selection.
        thread_n: Thread tile size for N dimension. -1 for auto-selection.
        sms: Number of SMs to use. -1 for all available.
        max_par: Maximum parallel batches for large matrices. Default 16.

    Returns:
        None. Output is written to C in-place.

    Raises:
        RuntimeError: If problem dimensions are incompatible with kernel.
        ImportError: If Marlin CUDA kernel cannot be loaded.

    Note:
        The kernel requires NVIDIA GPU with compute capability 8.0+.
        Input A must be contiguous in row-major format.
        Weight packing format is specific to Marlin - use the provided
        repacking utilities to convert from standard GPTQ format.
    """
    marlin = _load_marlin()
    marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)
