"""Test helpers of the Triton kernels of the torch backend."""

import dataclasses

from keras.src.backend.torch.ops import quantized
from keras.src.quantizers import kernels


def register_interpret_routes():
    """Adds `"triton_interpret"` to the matmul routes under the Triton
    interpreter, and returns `{"triton_interpret": "triton"}`. Elsewhere,
    it adds nothing and returns `{}`.

    The routes of `KERNELS` pass no CPU device: no measurement runs the
    Triton interpreter. The interpret route is the route `"triton"` with
    the device check "a CPU device", so `force_route` and a measured row
    run the kernel on CPU through real layers. It has no `kernel`, so
    `"require"` does not run it. `kernels_test_utils` calls this at
    import.
    """
    if not quantized.INTERPRETED:
        return {}
    (kernel,) = [k for k in quantized.KERNELS if k.name == "triton"]
    kernel = dataclasses.replace(
        kernel,
        supports_device=lambda device_class: device_class.startswith("cpu-"),
    )
    matmul, _ = kernels._kernel_routes(kernel)
    kernels._ROUTES["triton_interpret"] = dataclasses.replace(
        matmul, kernel=None
    )
    return {"triton_interpret": "triton"}
