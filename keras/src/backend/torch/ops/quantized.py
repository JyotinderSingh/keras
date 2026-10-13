"""The hooks of the torch backend for the quantized kernels.

`keras.src.quantizers.kernels` reads the device class of a call, whether
a tensor is sharded and whether Dynamo traces the call from this module
(`keras.src.backend.common.quantized_kernels`). The module has no
`KERNELS`, so the backend has no kernel.
"""

import functools
import platform

import torch
from torch.distributed.tensor import DTensor

from keras.src.backend.torch.ops.core import get_device


@functools.lru_cache(maxsize=None)
def device_capability(device):
    """The compute capability of a CUDA device, as `(major, minor)`."""
    return torch.cuda.get_device_capability(device)


def device_class():
    """The class of the device of `get_device()`: `cuda-sm<major><minor>`,
    `cpu-<machine>`, or the device type (`"mps"`, and `"cuda"` on ROCm).

    torch ops move every operand to that device.
    """
    return _device_class(get_device())


@functools.lru_cache(maxsize=None)
def _device_class(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.version.cuda:
        return "cuda-sm%d%d" % device_capability(device)
    if device.type == "cpu":
        return f"cpu-{platform.machine()}"
    return device.type


def is_sharded(tensor):
    """Whether `tensor` is a DTensor."""
    return isinstance(tensor, DTensor)


def is_compiling():
    """Whether Dynamo traces the call (`torch.compile`)."""
    return torch.compiler.is_compiling()
