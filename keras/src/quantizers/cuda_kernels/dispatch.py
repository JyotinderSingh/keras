"""Kernel dispatch logic for quantized operations.

This module provides utilities to detect whether optimized CUDA kernels
are available and should be used for quantized operations.

The dispatch logic checks:
1. Backend is PyTorch (CUDA kernels require PyTorch)
2. CUDA device is available
3. GPU compute capability is 8.0+ (Ampere or newer)
4. Marlin kernel is compiled and loadable

Example usage:
    >>> from keras.src.quantizers.cuda_kernels import dispatch
    >>> if dispatch.should_use_cuda_kernels():
    ...     # Use optimized CUDA kernel path
    ...     from keras.src.quantizers.cuda_kernels import marlin_ops
    ...     output = marlin_ops.gptq_matmul(...)
    ... else:
    ...     # Fall back to standard implementation
    ...     output = standard_matmul(...)
"""

from keras.src import backend

# Cache for availability checks to avoid repeated detection
_marlin_available = None
_cuda_kernels_available = None

# Override flag for testing
_force_cuda = None


def is_marlin_available():
    """Check if Marlin CUDA kernel is compiled and loadable.

    Returns:
        True if Marlin kernel can be loaded, False otherwise.
    """
    global _marlin_available

    if _marlin_available is None:
        try:
            from keras.src.quantizers.cuda_kernels import marlin

            _marlin_available = marlin.is_available()
        except ImportError:
            _marlin_available = False

    return _marlin_available


def _check_compute_capability():
    """Check if GPU has sufficient compute capability for Marlin.

    Marlin requires compute capability 8.0+ (Ampere architecture or newer).

    Returns:
        True if compute capability is sufficient, False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Get compute capability of default device
        major, minor = torch.cuda.get_device_capability()
        return major >= 8
    except (ImportError, RuntimeError):
        return False


def should_use_cuda_kernels():
    """Determine if CUDA kernels should be used for quantized operations.

    This function performs all necessary checks to determine if the optimized
    CUDA kernels can and should be used:

    1. Checks if Keras backend is PyTorch (required for CUDA kernels)
    2. Checks if CUDA device is available
    3. Checks if GPU compute capability is 8.0+ (Ampere or newer)
    4. Checks if Marlin kernel is compiled and loadable

    Returns:
        True if all conditions are met, False otherwise.
    """
    global _cuda_kernels_available, _force_cuda

    # Allow override for testing
    if _force_cuda is not None:
        return _force_cuda

    if _cuda_kernels_available is not None:
        return _cuda_kernels_available

    # Check 1: Must be using PyTorch backend
    if backend.backend() != "torch":
        _cuda_kernels_available = False
        return False

    # Check 2: CUDA must be available
    try:
        import torch

        if not torch.cuda.is_available():
            _cuda_kernels_available = False
            return False
    except ImportError:
        _cuda_kernels_available = False
        return False

    # Check 3: Compute capability must be 8.0+
    if not _check_compute_capability():
        _cuda_kernels_available = False
        return False

    # Check 4: Marlin kernel must be available
    if not is_marlin_available():
        _cuda_kernels_available = False
        return False

    _cuda_kernels_available = True
    return True


def reset_cache():
    """Reset cached availability checks.

    This is useful for testing or after environment changes.
    """
    global _marlin_available, _cuda_kernels_available, _force_cuda
    _marlin_available = None
    _cuda_kernels_available = None
    _force_cuda = None
