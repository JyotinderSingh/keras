"""The contract between a backend's quantized kernels and Keras.

A backend with kernels for 4-bit weight-only contractions exports, from
its `ops/quantized.py`:

- `KERNELS`: a tuple of `Kernel` records, in the order that the
  `"require"` policy tries them.
- `device_class()` (optional): the class of the device that a call runs
  on, for example `"cuda-sm89"`, `"tpu-TPU v6 lite"`, `"mps"` or
  `"cpu-arm64"`. Without it, Keras reads the default device.
- `is_sharded(tensor)` (optional): whether a tensor is sharded across
  devices. Without it, no tensor is.
- `is_compiling()` (optional): whether a compiler that guards on Python
  state traces the call. Without it, none does.

`keras.src.quantizers.kernels` turns each record into a route. The
module imports no framework and nothing from `keras.src.quantizers`.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Kernel:
    """A backend kernel for a 4-bit weight-only contraction.

    A kernel runs forward only: its caller runs it inside a custom
    gradient whose backward does not call the kernel. Each function
    selects its own launch configuration from the device, the dtype of
    `x` and the number of rows.

    Attributes:
        name: The route name, for example `"triton"`.
        matmul: `matmul(x, codes, scale, zero, *, group_size, signed,
            per_channel, input_scales=None)` returns the `(M, N)`
            contraction of the `(M, K)` activations `x` with the stored
            codes `(K, N / 2)`, two per byte along `N`. A grouped weight
            has the `(n_groups, N)` multiplier `scale` and the zero point
            `zero`. A `per_channel` weight has the `(N,)` divisor `scale`,
            no `zero` and no `group_size`. `signed` says that the codes
            are int8. `input_scales` (AWQ) divide `x`. A kernel without
            `contiguous_groups` also takes `g_idx`, the `(K,)` group of
            each row, or `None` when row `k` is in group
            `k // group_size`.
        logits: `logits(x, codes, scale, zero, *, group_size, per_channel,
            orientation)` returns the `(M, V)` logits of the `(M, D)`
            activations with a stored lookup table that packs two codes
            per byte along `D`: `(V, D / 2)` for a `"tied"` table, or
            `(D / 2, V)` for an `"untied"` one. `None` when the kernel has
            no logits variant.
        dtypes: The compute dtypes that the kernel supports.
        supports_device: `supports_device(device_class)` says whether the
            kernel runs on a device class with the installed library. It
            reads no tensor and no policy.
        requirements: The device and the library that the kernel needs,
            as a noun phrase for the error of the `"require"` policy, for
            example `"a CUDA device of compute capability 8.0 or newer"`.
        available: `available(device_class, m, k, n, group_size, dtype)`
            says whether the kernel runs one matmul: the shape, the group
            size and the compute dtype on a device class that
            `supports_device` passes.
        logits_available: `logits_available(device_class, m, d, v,
            group_size, dtype)`, the same check for `logits`.
        tested: `tested()` says whether the installed version of an
            experimental compiler is in the range that a run on hardware
            passed. `None` when the kernel has no such range. The
            `"auto"` policy refuses a kernel outside its range.
        contiguous_groups: `True` when the kernel reads the group of row
            `k` as `k // group_size` and takes no `g_idx`. Keras then
            runs it on a view with a `g_idx` only when `g_idx` holds
            these groups.
    """

    name: str
    matmul: object
    dtypes: tuple
    supports_device: object
    requirements: str
    available: object
    logits: object = None
    logits_available: object = None
    tested: object = None
    contiguous_groups: bool = False
