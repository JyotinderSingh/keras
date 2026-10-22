"""Quantized matmul kernels of the torch backend.

`w4a16_matmul` is Triton K1: the contraction of `(M, K)` activations
with a 4-bit weight that stays packed. The kernel reads the stored codes
`(K, ceil(N/2))`, two per byte: the low nibble of byte `j` is column
`2j` and the high nibble is column `2j + 1`. So each byte tile holds two
planes, the even and the odd columns, and each plane has its own
accumulator.

The kernel uses sum-then-scale. Per K tile, the small integers
`d = q - z` (in [-15, 15], exact in every dot dtype) go into a `tl.dot`
with a float32 result, and the group scale multiplies that partial sum.
One formula covers signed and unsigned codes: with `XOR = 8` for int8
storage and `XOR = 0` for uint8 storage, `q = nibble ^ XOR` and
`z' = z + XOR` give `d = q - z'`.

A program runs the K tiles of its split in one of two loops. Before the
loop, one check reads the `g_idx` rows of all these tiles:

- Fast loop: each tile lies in the group `k0 // G` (`g_idx` holds
  contiguous groups), and that group exists. Per tile, the kernel reads
  one scale row and one zero row at that static index.
- General loop: any other program, for example under GPTQ's activation
  order. The kernel clamps the group of each row to `[0, n_groups - 1]`,
  gathers its scale and zero point, dequantizes each weight in float32
  and casts it to the dot dtype, as the reference path rounds it.
- Per-channel int4: no zero point and no group index. The fast loop
  gives `d = q - XOR`, and the epilogue divides each column by its scale.

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

import torch

try:
    import triton
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction
except ImportError:
    triton = None

# The largest M of each bucket; a larger M is in the bucket "257+".
_M_BUCKETS = ((16, "1-16"), (64, "17-64"), (256, "65-256"))
# The outputs of one program of the split-K sum.
_SUM_BLOCK = 1024


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    """A launch configuration of `w4a16_matmul`.

    A program computes `block_m` rows and `2 * block_nh` columns, from
    `block_nh` bytes of codes per row. `block_k` is the K tile, or `None`
    for the tile that the group size gives. With `split_k > 1`, that
    number of programs shares the K tiles of an output tile, and a sum in
    a fixed order adds their float32 partials.
    """

    block_m: int
    block_nh: int
    block_k: int = None
    num_warps: int = 4
    num_stages: int = 3
    split_k: int = 1


# (compute capability, compute dtype, M bucket, G) -> `KernelConfig`. A key
# with `None` in place of the capability, the dtype or `G` is a default row
# for every value of that field. The GPU sessions add measured rows. There
# is no autotuning: it selects by noisy timings, and so it can change the
# summation order from one run to the next.
_CONFIGS = {
    (None, None, "1-16", None): KernelConfig(16, 32, num_stages=4),
    (None, None, "17-64", None): KernelConfig(32, 32),
    (None, None, "65-256", None): KernelConfig(64, 64),
    (None, None, "257+", None): KernelConfig(128, 64),
    # A float32 tile needs two times the shared memory of a 16-bit tile.
    # With a K tile of 128, these rows fit in the 99 KB of one block on
    # sm_86 and sm_89. The general loop also stages its scale and zero
    # point tiles: with 64 rows of `x`, it needs 108 KB on sm_89.
    (None, "float32", "65-256", None): KernelConfig(32, 32, num_stages=2),
    (None, "float32", "257+", None): KernelConfig(32, 32, num_stages=2),
    # sm_89 rows from the sweeps on an RTX 4090: t1a-20261022-2 (bfloat16)
    # and t1b-20261022-2 (float16, decode). The float16 prefill rows take
    # the bfloat16 values: the tiles have the same bytes. Split-K is on at
    # decode only. A row with `None` in place of `G` also serves per-channel
    # int4 and each `G` with no row of its own, so its K tile is `None` (the
    # tile of `_block_k`) or 32, which divides each `G` of 32 or more.
    ((8, 9), "bfloat16", "1-16", 128): KernelConfig(16, 32, 64, 4, 3, 8),
    ((8, 9), "bfloat16", "1-16", None): KernelConfig(16, 32, None, 4, 2, 8),
    ((8, 9), "float16", "1-16", 128): KernelConfig(16, 32, 64, 4, 3, 8),
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
    bucket = next((name for top, name in _M_BUCKETS if m <= top), "257+")
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
    awq_scales=None,
    upcast_dot=False,
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
        g_idx: The `(K,)` group of each row, or `None` for per-channel
            int4.
        group_size: `G`, or `None` for per-channel int4.
        signed: Whether the codes are int8 (sign-extended nibbles).
        per_channel: Whether the weight is per-channel int4.
        config: A `KernelConfig`, or `None` for the row of `_CONFIGS`.
        awq_scales: AWQ's `(K,)` input scales. The activations are
            divided by them in the dtype of `x`.
        upcast_dot: For the interpreter tests only: casts both dot
            operands to float32. The Triton interpreter computes a wrong
            bfloat16 `tl.dot`, so with this switch it checks every
            bfloat16 step except the bfloat16 dot itself.
    """
    if triton is None:
        raise ImportError("`w4a16_matmul` requires Triton.")
    if awq_scales is not None:
        x = x / awq_scales.to(x.dtype)
    m, k = x.shape
    n = scale.shape[-1]
    if m == 0:
        return x.new_empty((0, n))
    if config is None:
        capability = None
        if x.device.type == "cuda":
            capability = device_capability(x.device)
        dtype = str(x.dtype).removeprefix("torch.")
        config = kernel_config(capability, dtype, m, group_size)
    if x.dtype == torch.float32:
        input_precision = float32_input_precision()
    else:
        input_precision = "ieee"
    if per_channel:
        # The kernel reads no zero point and no group index.
        zero, g_idx, group_size = scale, scale, k
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
        config.block_k or _block_k(None if per_channel else group_size, k),
        config.num_warps,
        config.num_stages,
        config.split_k,
        input_precision,
        upcast_dot,
    )


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
        UPCAST_DOT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BM: tl.constexpr,
        BNH: tl.constexpr,
        BK: tl.constexpr,
    ):
        # `NH` is the row stride of the codes, `ceil(N/2)`. It is an
        # argument, so that the compiler knows when it divides by 16. Then
        # the codes load as vectors.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        rows = pid_m * BM + tl.arange(0, BM)
        # Byte `j` holds the columns `2j` and `2j + 1`.
        bytes_ = pid_n * BNH + tl.arange(0, BNH)
        columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
        row_mask = rows < M
        column_mask = columns < N
        dtype = x_ptr.dtype.element_ty
        acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
        acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
        # The K tiles of this split.
        tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
        first = pid_k * tiles
        last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
        uniform = True
        if not PER_CHANNEL:
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
        # In compiled Triton, a name that both branches of a runtime `if`
        # assign must have one type. The fast loop holds rows and the
        # general loop holds tiles, so their names are different. The
        # interpreter does not check this.
        if uniform:
            if not PER_CHANNEL:
                # The pipeliner stages only the loads of dot operands, and
                # the scale multiplies a dot result. So the loop loads the
                # scale row of the next tile one tile ahead. A program with
                # no tiles reads no row.
                s_next = tl.load(
                    scale_ptr + first * BK // group_size * N + columns,
                    column_mask & (first < last),
                    other=0.0,
                )
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
                    s_row = s_next
                    group = tl.minimum(tile + 1, last - 1) * BK // group_size
                    s_next = tl.load(
                        scale_ptr + group * N + columns, column_mask, other=0.0
                    )
                    row = k0 // group_size * N + columns
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

    @triton.jit
    def _splitk_sum(
        ws_ptr, y_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
    ):
        # The sum of the `(SPLIT_K, M, N)` float32 partials, in the order
        # 0 to `SPLIT_K - 1`, with one cast to the dtype of `y_ptr`. The
        # offsets of a split can pass 2^31, so they are int64. `tl.cast`
        # also takes `MN = 1`, which a launch gives as a constant.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < MN
        acc = tl.load(ws_ptr + offsets, mask=mask, other=0.0)
        for s in tl.static_range(1, SPLIT_K):
            step = s * tl.cast(MN, tl.int64)
            acc += tl.load(ws_ptr + step + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty), mask=mask)

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
        upcast_dot: bool,
    ) -> torch.Tensor:
        m, k = x.shape
        n = scale.shape[-1]
        y = x.new_empty((m, n))
        out = y
        if split_k > 1:
            # The float32 partials of the splits.
            out = x.new_empty((split_k, m, n), dtype=torch.float32)
        grid = (
            triton.cdiv(m, block_m),
            triton.cdiv(codes.shape[1], block_nh),
            split_k,
        )
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
                scale.shape[0],
                group_size,
                codes.shape[1],
                XOR=8 if signed else 0,
                PER_CHANNEL=per_channel,
                UPCAST_DOT=upcast_dot,
                INPUT_PRECISION=input_precision,
                SPLIT_K=split_k,
                BM=block_m,
                BNH=block_nh,
                BK=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            if split_k > 1:
                # A sum in a fixed order: the result is deterministic.
                torch.library.wrap_triton(_splitk_sum)[
                    (triton.cdiv(m * n, _SUM_BLOCK),)
                ](out, y, m * n, SPLIT_K=split_k, BLOCK=_SUM_BLOCK)
        return y

    @_w4a16_matmul.register_fake
    def _(x, codes, scale, *args):
        return x.new_empty((x.shape[0], scale.shape[-1]))

    # Whether `triton.jit` built the kernels for the Triton interpreter
    # (`TRITON_INTERPRET=1`), which runs them on CPU tensors.
    INTERPRETED = isinstance(_w4a16_kernel, InterpretedFunction)
else:
    INTERPRETED = False
