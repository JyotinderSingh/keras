"""Quantized matmul kernels of the torch backend.

`w4a16_matmul` is the Triton matmul kernel: the contraction of `(M, K)`
activations with a 4-bit weight that stays packed. The kernel reads the
stored codes `(K, ceil(N/2))`, two per byte: the low nibble of byte `j`
is column `2j` and the high nibble is column `2j + 1`. So each byte tile
holds two planes, the even and the odd columns, and each plane has its
own accumulator.

The kernel uses sum-then-scale. Per K tile, the small integers
`d = q - z` (in [-15, 15], exact in every dot dtype) go into a `tl.dot`
with a float32 result, and the group scale multiplies that partial sum.
One formula covers signed and unsigned codes: with `XOR = 8` for int8
storage and `XOR = 0` for uint8 storage, `q = nibble ^ XOR` and
`z' = z + XOR` give `d = q - z'`.

A program runs the K tiles of its split in one of two loops:

- Fast loop: each tile lies in the group `k0 // G`, and that group
  exists. Per tile, the kernel reads one scale row and one zero row at
  that static index.
- General loop: any other program, for example under GPTQ's activation
  order. The kernel clamps the group of each row to `[0, n_groups - 1]`,
  gathers its scale and zero point, dequantizes each weight in float32
  and casts it to the dot dtype, as the reference path rounds it.
- Per-channel int4: no zero point and no group index. The fast loop
  gives `d = q - XOR`, and the epilogue divides each column by its scale.

A weight whose mode fixes its groups (int4, AWQ, GPTQ with one group)
has no `g_idx`: row `k` is in group `k // G`. When each K tile lies in
one group, the constexpr `CONTIGUOUS` compiles the fast loop only, with
no check and no `g_idx` load. Any other grouped call has a `g_idx`, and
one check per program reads the `g_idx` rows of all its tiles before the
loops to select one of them.

No loop body has a runtime branch, so the compiler can pipeline the
loads of the next tiles.

The epilogue interleaves the two accumulators into column order, casts to
the compute dtype and stores with a column mask. With split-K, each split
stores its float32 partial, and a second kernel in the same op adds the
partials in a fixed order and casts the sum once. The kernels are
forward only: their caller runs them inside a custom gradient.

Triton ships with torch on Linux. Where it does not import, this module
has no kernel.
"""

import contextlib
import dataclasses
import functools
import platform

import torch
from torch.distributed.tensor import DTensor

from keras.src.backend.common.backend_utils import m_bucket
from keras.src.backend.common.quantized_kernels import Kernel
from keras.src.backend.torch.ops.core import get_device

try:
    import triton
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction
except ImportError:
    triton = None

# The outputs of one program of the split-K sum.
_SUM_BLOCK = 1024


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    """A launch configuration of `w4a16_matmul`.

    A program computes `block_m` rows and `2 * block_nh` columns. Per K
    tile of `block_k` rows, it reads `block_nh` bytes of codes per row.
    `block_k` is `None` for the tile that the group size gives. With
    `split_k > 1`, that number of programs shares the K tiles of an output
    tile, and a second kernel adds their float32 partials in a fixed
    order.
    """

    block_m: int
    block_nh: int
    block_k: int = None
    num_warps: int = 4
    num_stages: int = 3
    split_k: int = 1


# (compute capability, compute dtype, M bucket, G) -> `KernelConfig`. A key
# with `None` in place of the capability, the dtype or `G` is a default row
# for every value of that field. There is no autotuning: it selects by
# noisy timings, and so it can change the summation order from one run to
# the next. The frontend fits a row to the shared memory of the device
# (`fit_config`).
_CONFIGS = {
    (None, None, "1-16", None): KernelConfig(16, 32, num_stages=4),
    (None, None, "17-64", None): KernelConfig(32, 32),
    (None, None, "65-256", None): KernelConfig(64, 64),
    (None, None, "257+", None): KernelConfig(128, 64),
    # A float32 tile needs two times the shared memory of a 16-bit tile,
    # and the general loop stages its scale and zero point tiles too. With
    # a K tile of 128, these rows fit in the 99 KB of one block on sm_86
    # and sm_89. With 64 rows of `x`, a float32 call needs 108 KB.
    (None, "float32", "65-256", None): KernelConfig(32, 32, num_stages=2),
    (None, "float32", "257+", None): KernelConfig(32, 32, num_stages=2),
    # sm_89 rows from the sweeps on an RTX 4090. The decode rows (M 1-16)
    # are measured on this kernel, the prefill rows on the kernel before
    # the two loops. The float16 prefill rows take the bfloat16 values: the
    # tiles have the same bytes. Split-K is on at decode only. A row with
    # `None` in place of `G` also serves per-channel int4 and each `G` with
    # no row of its own, so its K tile is `None` (the tile of `_block_k`)
    # or 32, which divides each `G` of 32 or more.
    ((8, 9), "bfloat16", "1-16", 128): KernelConfig(16, 32, 64, 4, 2, 8),
    ((8, 9), "bfloat16", "1-16", None): KernelConfig(16, 32, None, 4, 2, 8),
    ((8, 9), "float16", "1-16", 128): KernelConfig(16, 32, 64, 4, 2, 8),
    ((8, 9), "float16", "1-16", None): KernelConfig(16, 32, None, 4, 2, 8),
    ((8, 9), "bfloat16", "65-256", 128): KernelConfig(128, 32, 32, 4, 4, 1),
    ((8, 9), "bfloat16", "65-256", None): KernelConfig(64, 32, None, 4, 3, 1),
    ((8, 9), "bfloat16", "257+", 128): KernelConfig(256, 32, 32, 4, 3, 1),
    ((8, 9), "bfloat16", "257+", None): KernelConfig(256, 32, 32, 4, 4, 1),
    ((8, 9), "float16", "65-256", 128): KernelConfig(128, 32, 32, 4, 4, 1),
    ((8, 9), "float16", "65-256", None): KernelConfig(64, 32, None, 4, 3, 1),
    ((8, 9), "float16", "257+", 128): KernelConfig(256, 32, 32, 4, 3, 1),
    ((8, 9), "float16", "257+", None): KernelConfig(256, 32, 32, 4, 4, 1),
}


def kernel_config(capability, dtype, m, group_size):
    """The configuration of the most specific row of `_CONFIGS`."""
    bucket = m_bucket(m)
    for key in (
        (capability, dtype, bucket, group_size),
        (capability, dtype, bucket, None),
        (None, dtype, bucket, None),
        (None, None, bucket, None),
    ):
        if key in _CONFIGS:
            return _CONFIGS[key]


def _block_k(group_size, k):
    """The K tile: `G` for `G` in 32, 64 and 128, else up to 128.

    A tile divides `G` where it can, so that a tile with contiguous groups
    lies in one group and takes the fast path. One group (`G >= K`, or a
    per-channel scale) takes the smallest tile that covers `K`, at most
    128.
    """
    if group_size is None or group_size >= k:
        return max(16, min(128, triton.next_power_of_2(k)))
    # The largest power of two that divides `G`.
    return max(16, min(128, group_size & -group_size))


def _contiguous(g_idx, group_size, k, block_k):
    """Whether the kernel takes the group of a K tile from its start.

    Without `g_idx`, row `k` is in group `k // G`. When each K tile lies in
    one group (`G >= K`, or the tile divides `G`), the kernel takes the
    group of the tile and reads no `g_idx` (`CONTIGUOUS`). Else the general
    loop reads the groups. A call with a `g_idx` reads its groups.
    """
    return g_idx is None and (group_size >= k or group_size % block_k == 0)


def shared_bytes(config, block_k, sizes, per_channel, capability):
    """An estimate of the shared memory of one block, in bytes.

    `sizes` holds the bytes of one element of `x`, the scale, the zero
    point and `g_idx`. The estimate follows the pipeliner of Triton 3.8.
    Each load that the fast loop or the per-channel loop stages has
    `num_stages - 1` buffers. The general loop has one buffer for each of
    its tiles: `x`, the codes, the scale, the zero point and `g_idx`. A
    float32 dot also stages the weight tiles. On sm_90, a 16-bit dot with
    64 rows or more (wgmma) stages `x` and the weight tiles. A
    `CONTIGUOUS` variant compiles the fast loop only, and the estimate
    keeps the general loop for it, so its rows fit as with a `g_idx`. The
    compile test checks that no compiled row needs more than its estimate.
    """
    x_size, scale_size, zero_size, g_idx_size = sizes
    block_m, block_nh = config.block_m, config.block_nh
    x = block_m * block_k * x_size
    codes = block_k * block_nh
    weights = 2 * block_k * block_nh * x_size
    dot = 0
    if x_size == 4:
        dot = weights
    elif capability >= (9, 0) and block_m >= 64:
        dot = x + weights
    buffers = max(1, config.num_stages - 1)
    # The epilogue converts the output tile: float32 partials with split-K.
    out_size = 4 if config.split_k > 1 else x_size
    epilogue = block_m * 2 * block_nh * out_size
    if per_channel:
        # The per-channel codes load into registers, not into shared memory.
        return max(buffers * x + codes + dot, epilogue)
    rows = config.num_stages * 2 * block_nh * zero_size
    fast = buffers * (x + codes) + rows
    tiles = 2 * block_k * block_nh * (scale_size + zero_size)
    general = x + codes + tiles + block_k * g_idx_size
    return max(max(fast, general) + dot, epilogue)


def fit_config(config, block_k, sizes, per_channel, capability, limit):
    """`config`, changed until `shared_bytes` is at most `limit`.

    Each step takes one stage less, when that makes the estimate smaller
    and at least 2 stages stay. Else it halves the columns, and then the
    rows, of the tile, down to 16. A row that fits does not change. The
    K tile does not change, so each output adds the same partial sums in
    the same order. sm_75 has no asynchronous copies, so Triton stages no
    loads there and the estimate does not apply: the row does not change.
    """
    # An estimate of the shared memory that cuts the stages of a row until
    # it fits is an idea of gemlite (`estimate_shared_memory_per_block` and
    # `kernel_config_pruner`).

    def size(config):
        return shared_bytes(config, block_k, sizes, per_channel, capability)

    while capability >= (8, 0) and size(config) > limit:
        fewer = dataclasses.replace(config, num_stages=config.num_stages - 1)
        if config.num_stages > 2 and size(fewer) < size(config):
            config = fewer
        elif config.block_nh > 16:
            config = dataclasses.replace(config, block_nh=config.block_nh // 2)
        elif config.block_m > 16:
            config = dataclasses.replace(config, block_m=config.block_m // 2)
        else:
            break
    return config


# Dynamo cannot trace the getter of `fp32_precision`, so a compiled function
# reads the result once, when it compiles. A precision change after that
# needs a recompile.
@torch.compiler.assume_constant_result
def float32_input_precision():
    """The Triton `input_precision` of a float32 dot.

    It follows `torch.backends.cuda.matmul.fp32_precision`, which every
    float32 precision setter of torch sets: `"tf32"` gives `"tf32"`, and
    any other value (`"ieee"`, `"none"`) gives `"ieee"`. The frontend does
    not call `torch.get_float32_matmul_precision()`: it raises after the
    newer API sets `"tf32"`.
    """
    if torch.backends.cuda.matmul.fp32_precision == "tf32":
        return "tf32"
    return "ieee"


@functools.lru_cache(maxsize=None)
def device_capability(device):
    """The compute capability of a CUDA device, as `(major, minor)`."""
    return torch.cuda.get_device_capability(device)


@functools.lru_cache(maxsize=None)
def device_shared_memory(device):
    """The shared memory that one block can have on a CUDA device."""
    properties = torch.cuda.get_device_properties(device)
    return properties.shared_memory_per_block_optin


def _launch_options(x, group_size, config):
    """The configuration and the dot input precision of a call on `x`.

    `config` is `None` for the most specific row of `_CONFIGS`.
    """
    if config is None:
        capability = None
        if x.device.type == "cuda":
            capability = device_capability(x.device)
        dtype = str(x.dtype).removeprefix("torch.")
        config = kernel_config(capability, dtype, x.shape[0], group_size)
    if x.dtype == torch.float32:
        return config, float32_input_precision()
    return config, "ieee"


def w4a16_matmul(
    x,
    codes,
    scale,
    zero,
    g_idx,
    *,
    group_size,
    signed,
    per_channel,
    config=None,
    input_scales=None,
):
    """Returns the `(M, N)` contraction of `x` with a 4-bit weight.

    Args:
        x: The `(M, K)` activations: bfloat16, float16 or float32. The
            result has the same dtype.
        codes: The `(K, ceil(N/2))` codes, two per byte: int8 (int4
            mode) or uint8 (GPTQ, AWQ).
        scale: The `(n_groups, N)` multiplier, or the `(N,)` divisor of
            per-channel int4.
        zero: The `(n_groups, N)` zero point, or `None` for per-channel
            int4.
        g_idx: The `(K,)` group of each row, or `None` when row `k` is in
            group `k // group_size` (int4, AWQ, GPTQ with one group) and
            for per-channel int4.
        group_size: `G`, or `None` for per-channel int4.
        signed: Whether the codes are int8 (sign-extended nibbles).
        per_channel: Whether the weight is per-channel int4.
        config: A `KernelConfig`, or `None` for the row of `_CONFIGS`,
            fitted to the shared memory of the device.
        input_scales: AWQ's `(K,)` input scales. The activations are
            divided by them in the dtype of `x`.
    """
    if triton is None:
        raise ImportError("`w4a16_matmul` requires Triton.")
    if input_scales is not None:
        x = x / input_scales.to(x.dtype)
    m, k = x.shape
    n = scale.shape[-1]
    if m == 0:
        return x.new_empty((0, n))
    # A row of the table fits the shared memory of the device.
    fit = config is None and x.device.type == "cuda"
    config, input_precision = _launch_options(x, group_size, config)
    if per_channel:
        # The kernel reads no zero point and no group index.
        zero, g_idx, group_size = scale, scale, k
    block_k = config.block_k or _block_k(None if per_channel else group_size, k)
    contiguous = _contiguous(g_idx, group_size, k, block_k)
    if fit:
        # The estimate counts the float32 `g_idx` of the general loop for
        # a contiguous call too, so the row fits as for a stored `g_idx`.
        index = 4 if g_idx is None else g_idx.element_size()
        sizes = tuple(t.element_size() for t in (x, scale, zero)) + (index,)
        capability = device_capability(x.device)
        limit = device_shared_memory(x.device)
        config = fit_config(
            config, block_k, sizes, per_channel, capability, limit
        )
    if contiguous:
        # The kernel reads no group index: the scale stands in for it.
        g_idx = scale
    elif g_idx is None:
        g_idx = torch.arange(k, dtype=torch.float32, device=x.device)
        g_idx = torch.floor_divide(g_idx, group_size)
    return torch.ops.keras.w4a16_matmul(
        x.contiguous(),
        codes.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        g_idx.contiguous(),
        group_size,
        signed,
        per_channel,
        config.block_m,
        config.block_nh,
        block_k,
        config.num_warps,
        config.num_stages,
        config.split_k,
        input_precision,
        contiguous,
    )


# Whether `triton.jit` built the kernels for the Triton interpreter
# (`TRITON_INTERPRET=1`), which runs them on CPU tensors.
INTERPRETED = False
if triton is not None:

    @triton.jit
    def _dot(a, b, UPCAST_DOT: tl.constexpr, INPUT_PRECISION: tl.constexpr):
        if UPCAST_DOT:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        return tl.dot(
            a, b, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

    @triton.jit
    def _w4a16_kernel(
        x_ptr,
        codes_ptr,
        scale_ptr,
        zero_ptr,
        g_idx_ptr,
        y_ptr,
        M,
        N,
        K,
        n_groups,
        group_size,
        NH,
        XOR: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        CONTIGUOUS: tl.constexpr,
        UPCAST_DOT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BM: tl.constexpr,
        BNH: tl.constexpr,
        BK: tl.constexpr,
    ):
        # `NH` is the row stride of the codes, `ceil(N/2)`. It is an
        # argument, so that the compiler knows when it divides by 16. Then
        # the grouped loops load the codes as vectors, and the pipeliner
        # stages them. Per-channel int4 computes the stride from `N` and
        # loads its codes byte by byte: on an RTX 4090, the argument made
        # per-channel int4 4.5% slower at decode.
        if PER_CHANNEL:
            NH = (N + 1) // 2
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        rows = pid_m * BM + tl.arange(0, BM)
        columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
        row_mask = rows < M
        column_mask = columns < N
        dtype = x_ptr.dtype.element_ty
        # Byte `j` holds the columns `2j` and `2j + 1`.
        bytes_ = pid_n * BNH + tl.arange(0, BNH)
        acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
        acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
        # The K tiles of this split.
        tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
        first = pid_k * tiles
        last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
        uniform = True
        if not (PER_CHANNEL or CONTIGUOUS):
            # One check per program, `CHECK` rows of `g_idx` per step: each
            # row of tile `t` is in the group `t * BK // G`, and the group
            # of the last tile exists.
            CHECK: tl.constexpr = 1024
            bad = tl.zeros((CHECK,), dtype=tl.int32)
            for start in tl.range(first * BK, last * BK, CHECK):
                ks = start + tl.arange(0, CHECK)
                k_mask = ks < tl.minimum(last * BK, K)
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
                want = (ks // BK) * BK // group_size
                bad = bad | tl.where(k_mask & (g.to(tl.int32) != want), 1, 0)
            uniform = (tl.max(bad, axis=0) == 0) & (
                (last - 1) * BK // group_size < n_groups
            )
        # The pipeliner does not stage a loop body with a runtime branch
        # that holds a dot. So the branch is outside the two loops. The
        # grouped loop with no branch, at a K tile that is at most `G`, is
        # an idea of gemlite (`gemm_splitK_kernels.py`) and vLLM
        # (`triton_w4a16.py`). Triton makes a tensor of `uniform = True`,
        # so a constexpr in the condition keeps the general loop out of the
        # variants that never run it.
        if PER_CHANNEL or CONTIGUOUS or uniform:
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    codes_ptr + ks[:, None] * NH + bytes_[None, :],
                    mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                    other=0,
                ).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                # An integer goes to the dot dtype through float32, which
                # holds it exactly: the Triton interpreter converts an int32
                # to a wrong bfloat16.
                if PER_CHANNEL:
                    d_even = (even - XOR).to(tl.float32).to(dtype)
                    d_odd = (odd - XOR).to(tl.float32).to(dtype)
                    acc_even += _dot(x, d_even, UPCAST_DOT, INPUT_PRECISION)
                    acc_odd += _dot(x, d_odd, UPCAST_DOT, INPUT_PRECISION)
                else:
                    # The scale and the zero row of the group of the tile.
                    row = k0 // group_size * N + columns
                    s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                    z_row = tl.load(zero_ptr + row, column_mask, other=0)
                    s_row_even, s_row_odd = tl.split(
                        tl.reshape(s_row.to(tl.float32), (BNH, 2))
                    )
                    z_row_even, z_row_odd = tl.split(
                        tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2))
                    )
                    d_even = (even - z_row_even[None, :]).to(tl.float32)
                    d_odd = (odd - z_row_odd[None, :]).to(tl.float32)
                    d_even, d_odd = d_even.to(dtype), d_odd.to(dtype)
                    partial = _dot(x, d_even, UPCAST_DOT, INPUT_PRECISION)
                    acc_even += partial * s_row_even[None, :]
                    partial = _dot(x, d_odd, UPCAST_DOT, INPUT_PRECISION)
                    acc_odd += partial * s_row_odd[None, :]
        else:
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    codes_ptr + ks[:, None] * NH + bytes_[None, :],
                    mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                    other=0,
                ).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                # The scale and the zero point of each row. The clamp keeps
                # a corrupt `g_idx` in bounds.
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
                g = tl.minimum(tl.maximum(g.to(tl.int32), 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_even, s_odd = tl.split(
                    tl.reshape(s.to(tl.float32), (BK, BNH, 2))
                )
                z_even, z_odd = tl.split(
                    tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2))
                )
                w_even = ((even - z_even).to(tl.float32) * s_even).to(dtype)
                w_odd = ((odd - z_odd).to(tl.float32) * s_odd).to(dtype)
                acc_even += _dot(x, w_even, UPCAST_DOT, INPUT_PRECISION)
                acc_odd += _dot(x, w_odd, UPCAST_DOT, INPUT_PRECISION)
        acc = tl.interleave(acc_even, acc_odd)
        if PER_CHANNEL:
            s = tl.load(scale_ptr + columns, mask=column_mask, other=1.0)
            acc = acc / s.to(tl.float32)[None, :]
        if SPLIT_K > 1:
            # `y_ptr` is the `(SPLIT_K, M, N)` float32 workspace. Its offsets
            # can pass 2^31, so they are int64.
            rows = pid_k.to(tl.int64) * M + rows
        offsets = rows[:, None] * N + columns[None, :]
        tl.store(
            y_ptr + offsets,
            acc.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & column_mask[None, :],
        )

    # The interpreter computes a wrong bfloat16 `tl.dot`, so the op casts
    # the dot operands to float32 there.
    INTERPRETED = isinstance(_w4a16_kernel, InterpretedFunction)

    @triton.jit
    def _splitk_sum(
        ws_ptr, y_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
    ):
        # The sum of the `(SPLIT_K, M, N)` float32 partials, in the order
        # 0 to `SPLIT_K - 1`, with one cast to the dtype of `y_ptr`. The
        # offsets of the programs and of the splits can pass 2^31, so they
        # are int64. `tl.cast` also takes `MN = 1`, which a launch gives as
        # a constant. The reduce kernel in the same op as the matmul, over
        # a float32 scratchpad, is an idea of OpenAI `triton_kernels`
        # (`matmul.py`).
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < MN
        acc = tl.load(ws_ptr + offsets, mask=mask, other=0.0)
        for s in tl.static_range(1, SPLIT_K):
            step = s * tl.cast(MN, tl.int64)
            acc += tl.load(ws_ptr + step + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty), mask=mask)

    def _launch(
        x,
        codes,
        scale,
        zero,
        g_idx,
        n,
        n_groups,
        group_size,
        nh,
        *,
        block_m,
        block_nh,
        block_k,
        num_warps,
        num_stages,
        split_k,
        **constants,
    ):
        """Launches `_w4a16_kernel` and returns its `(M, n)` result.

        `nh` is the row stride of N-packed codes, and `constants` its
        `XOR`, `PER_CHANNEL`, `CONTIGUOUS`, `UPCAST_DOT` and
        `INPUT_PRECISION`. With split-K, `_splitk_sum` adds the float32
        partials of the splits.
        """
        m, k = x.shape
        y = x.new_empty((m, n))
        out = y
        if split_k > 1:
            # The float32 partials of the splits.
            out = x.new_empty((split_k, m, n), dtype=torch.float32)
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, 2 * block_nh), split_k)
        # The Triton interpreter runs on CPU tensors.
        device = contextlib.nullcontext()
        if x.device.type == "cuda":
            device = torch.cuda.device(x.device)
        with device:
            torch.library.wrap_triton(_w4a16_kernel)[grid](
                x,
                codes,
                scale,
                zero,
                g_idx,
                out,
                m,
                n,
                k,
                n_groups,
                group_size,
                nh,
                SPLIT_K=split_k,
                BM=block_m,
                BNH=block_nh,
                BK=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                **constants,
            )
            if split_k > 1:
                # A sum in a fixed order: the result is deterministic.
                torch.library.wrap_triton(_splitk_sum)[
                    (triton.cdiv(m * n, _SUM_BLOCK),)
                ](out, y, m * n, SPLIT_K=split_k, BLOCK=_SUM_BLOCK)
        return y

    @torch.library.triton_op("keras::w4a16_matmul", mutates_args=())
    def _w4a16_matmul(
        x: torch.Tensor,
        codes: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        g_idx: torch.Tensor,
        group_size: int,
        signed: bool,
        per_channel: bool,
        block_m: int,
        block_nh: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
        split_k: int,
        input_precision: str,
        contiguous: bool,
    ) -> torch.Tensor:
        return _launch(
            x,
            codes,
            scale,
            zero,
            g_idx,
            scale.shape[-1],
            scale.shape[0],
            group_size,
            codes.shape[1],
            block_m=block_m,
            block_nh=block_nh,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            split_k=split_k,
            XOR=8 if signed else 0,
            PER_CHANNEL=per_channel,
            CONTIGUOUS=contiguous,
            UPCAST_DOT=INTERPRETED,
            INPUT_PRECISION=input_precision,
        )

    @_w4a16_matmul.register_fake
    def _(x, codes, scale, *args):
        return x.new_empty((x.shape[0], scale.shape[-1]))


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


def _triton_supports_device(device_class):
    """A CUDA device of compute capability 8.0 or newer, or 7.5.

    A ROCm device has the class `"cuda"`: `pytorch-triton-rocm` imports
    as `triton`, but the kernel is not tested there. The Triton
    interpreter runs on CPU in the tests only.
    """
    if not device_class.startswith("cuda-sm"):
        return False
    capability = int(device_class.removeprefix("cuda-sm"))
    return capability >= 80 or capability == 75


def _triton_available(device_class, m, k, n, group_size, dtype):
    """The int32 offsets, and float16 compute on compute capability 7.5."""
    # The kernel indexes with int32 offsets. The split-K workspace, which
    # can hold more elements, has int64 offsets.
    if max(m * k, m * n, k * n) >= 2**31:
        return False
    return device_class != "cuda-sm75" or dtype == "float16"


# The kernels of this backend. Triton ships with torch on Linux. Where it
# does not import, the backend has none. The GPU job of CI runs the kernel
# under `"require"` on the torch release that it installs.
KERNELS = ()
if triton is not None:
    KERNELS = (
        Kernel(
            name="triton",
            matmul=w4a16_matmul,
            dtypes=("bfloat16", "float16", "float32"),
            supports_device=_triton_supports_device,
            requirements=(
                "a CUDA device of compute capability 8.0 or newer, or 7.5 "
                "with float16 compute"
            ),
            available=_triton_available,
        ),
    )
