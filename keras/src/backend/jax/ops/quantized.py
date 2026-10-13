"""Quantized contraction kernels of the JAX backend.

`keras.src.quantizers.kernels` runs each kernel as a route. A kernel runs
forward only: its caller runs it inside a custom gradient whose backward
does not call the kernel.
"""

import functools

import jax
import jax.numpy as jnp
import jaxlib
from jax import lax
from jax.experimental import pallas as pl

from keras.src.utils.module_utils import LazyModule

# Pallas Mosaic GPU loads on first use, because its import takes more than
# 100 ms. JAX 0.10.2 has no `plgpu.mma`, so `plgpu.available` is False.
plgpu = LazyModule("jax.experimental.pallas.mosaic_gpu", required_attr="mma")

# The rows of a K step. With 16 activation rows, the MMA left operand
# needs four (8, 64) tiles, so 64 rows do not lower.
_BK = 128
# float32 2**23 and its bits. Its ulp is 1, so the float32 with the bits
# `_MAGIC_BITS | q` is 2**23 + q, exactly, for 0 <= q < 2**23.
_MAGIC = 2.0**23
_MAGIC_BITS = 0x4B000000
# Compute capability -> the SM count of its card: an A100, an A40, an RTX
# 4090 and an H100 SXM.
_SMS = {"8.0": 108, "8.6": 84, "8.9": 128, "9.0": 132}
# (compute capability, M bucket) -> the row of a sweep with JAX 0.12.0 on
# an RTX 4090 (8.9, bfloat16 and float16) or an H100 SXM (9.0, bfloat16).
# In a confirmation pass over the cells of its key, each row is faster
# than the start row. float16 takes the 9.0 row of bfloat16: on the RTX
# 4090, the sweeps give the same rows for the two dtypes.
_TUNED = {
    ("8.9", "1-16"): (16, 32, 2, 128),
    ("8.9", "65-256"): (32, 32, 2, 1024),
    ("8.9", "257+"): (32, 32, 2, 1),
    ("9.0", "1-16"): (16, 64, 4, 132),
}
# (compute capability, compute dtype, M bucket) -> (BM, BNH, stages,
# min_blocks). `BNH` is the number of code columns of a block, and
# `min_blocks` the number of blocks that the split-K axis aims for. A key
# with no row in `_TUNED` takes a start value of the SM count.
MOSAIC_GPU_CONFIGS = {
    (capability, dtype, bucket): _TUNED.get((capability, bucket), row)
    for capability, sms in _SMS.items()
    for dtype in ("bfloat16", "float16")
    for bucket, row in (
        ("1-16", (16, 32, 3, 2 * sms)),
        ("17-64", (32, 64, 2, 2 * sms)),
        ("65-256", (64, 64, 2, 1)),
        ("257+", (64, 64, 2, 1)),
    )
}


def mosaic_gpu_versions():
    """The `(jax, jaxlib)` pair. The Mosaic GPU compiler ships in jaxlib."""
    return jax.__version__, jaxlib.__version__


# The kernel compiles once per signature (the shapes, the dtypes and the
# static arguments), so a repeated eager call does not trace or compile
# again. Under a trace, the function inlines.
@functools.partial(
    jax.jit,
    static_argnames=(
        "group_size",
        "signed",
        "per_channel",
        "config",
        "interpret",
    ),
    inline=True,
)
def mosaic_gpu_w4a16_matmul(
    x,
    codes,
    scale,
    zero,
    *,
    group_size,
    signed,
    per_channel,
    config,
    input_scales=None,
    interpret=False,
):
    """Returns the `(M, N)` contraction of `x` against the 4-bit weight, in
    `x.dtype`.

    Mosaic GPU K1, with sum-then-scale. `x` is `(M, K)` in bfloat16 or
    float16. `codes` is the stored `(K, N / 2)` tensor: byte `j` holds
    column `2j` in its low nibble and column `2j + 1` in its high nibble,
    with `signed` nibbles in int8 storage. A grouped weight has the
    `(n_groups, N)` multiplier `scale` and the int8 or uint8 zero point
    `zero`, and row `k` is in group `k // group_size`. A `per_channel`
    weight has the `(N,)` divisor `scale` and no `zero`. AWQ's
    `input_scales` divide `x` in its dtype. `config` is a row
    `(BM, BNH, stages, min_blocks)` of `MOSAIC_GPU_CONFIGS`. The caller
    checks the alignment: `K % 128 == 0`, `N % 64 == 0`, and a
    `group_size` that 128 divides.

    `interpret=True` (tests only) runs the kernel in GPU interpret mode,
    with `jnp.dot` in place of `plgpu.mma`, which has no interpret rule.
    """
    block_m, block_n, stages, min_blocks = config
    if input_scales is not None:
        x = x / input_scales.astype(x.dtype)
    m, k = x.shape
    half = codes.shape[1]
    # `N % 64 == 0`, so 32 code columns divide `N / 2`.
    while half % block_n:
        block_n //= 2
    padded = -(-m // block_m) * block_m
    blocks = padded // block_m * (half // block_n)
    split = _split_k(k // _BK, blocks, min_blocks)
    xor = 8 if signed else 0
    inputs = [jnp.pad(x, ((0, padded - m), (0, 0))), codes]
    if not per_channel:
        # The kernel subtracts 2**23 + z + XOR in float32, which is exact
        # only for an integer zero point `z`.
        if zero.dtype not in (jnp.int8, jnp.uint8):
            raise ValueError(
                "`mosaic_gpu_w4a16_matmul` needs an int8 or uint8 zero "
                f"point. Received: zero.dtype={zero.dtype}"
            )
        # Axis 2 holds the scale of the even and the odd columns, then
        # their zero point plus `XOR` plus 2**23. Mosaic GPU has no
        # strided slices, so the columns split here. A block
        # `(1, 1, 4, BNH)` is one contiguous run, as an untiled `cp.async`
        # copy needs.
        scale_rows = scale.astype("float32")
        zero_rows = zero.astype("float32") + (xor + _MAGIC)
        shape = (scale.shape[0], half // block_n, block_n)
        inputs.append(
            jnp.stack(
                [
                    rows[:, i::2].reshape(shape)
                    for rows in (scale_rows, zero_rows)
                    for i in (0, 1)
                ],
                axis=2,
            )
        )
    planes = _w4a16_kernel(
        *inputs,
        xor=xor,
        per_group=None if per_channel else group_size // _BK,
        block_m=block_m,
        block_n=block_n,
        stages=stages,
        split=split,
        interpret=interpret,
    )
    # The partials of the splits sum in a fixed order, so a call is
    # deterministic.
    sums = []
    for plane in planes:
        total = plane[0, :m]
        for i in range(1, split):
            total = total + plane[i, :m]
        sums.append(total)
    y = jnp.stack(sums, axis=-1).reshape(m, 2 * half)
    if per_channel:
        y = y / scale.astype("float32")
    return y.astype(x.dtype)


def _split_k(steps, blocks, min_blocks):
    """The number of splits of the K steps.

    One warpgroup per block underfills a large GPU at decode. So the grid
    takes the smallest split of 1, 2, 4 and 8 that divides `steps` and
    gives at least `min_blocks` blocks, or else the largest that divides
    `steps`.
    """
    splits = [s for s in (1, 2, 4, 8) if steps % s == 0]
    return next((s for s in splits if blocks * s >= min_blocks), splits[-1])


def _w4a16_kernel(
    x, codes, *rows, xor, per_group, block_m, block_n, stages, split, interpret
):
    """The float32 partials `(split, Mp, N / 2)` of the even and the odd
    columns.

    Each block runs an input-only pipeline over its K steps. A step widens
    the code tile and splits it into the two planes `(b & 15) ^ XOR` and
    `((b >> 4) & 15) ^ XOR`. A bitcast of each plane with the bits of the
    float32 2**23 gives 2**23 plus the plane. The step subtracts the zero
    row plus `XOR` plus 2**23 (or `XOR` plus 2**23 per channel). The
    results are integers in [-15, 15], exact in the compute dtype. Each
    plane goes through an SMEM tile into one MMA with the activation tile,
    and its scale row multiplies the float32 result.
    """
    dtype = x.dtype
    half = codes.shape[1]
    steps = x.shape[1] // _BK // split
    plane = jax.ShapeDtypeStruct((split, x.shape[0], half), jnp.float32)
    stage = plgpu.SMEM(
        (_BK, block_n), dtype, transforms=_transforms(block_n, dtype)
    )
    # A row that broadcasts across the rows of the accumulator.
    scale_layout = plgpu.Layout.MMA_ACC(dtype, m_warps=block_m // 16)
    scale_layout = scale_layout.reduce(0)

    def mma(acc, a, b):
        if interpret:
            return acc + jnp.dot(a, b, preferred_element_type=jnp.float32)
        return plgpu.mma(acc, a, b)

    # The structure of the kernel is an idea of the sm80 `ragged_dot`
    # kernel of Tokamax
    # (`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm80.py`): an
    # input-only `plgpu.emit_pipeline` over the K steps of a block,
    # `plgpu.mma` on swizzled SMEM tiles, and a store of the accumulators
    # after the pipeline.
    @plgpu.kernel(
        out_type=(plane, plane),
        scratch_types=[stage, stage],
        grid=(x.shape[0] // block_m, half // block_n, split),
        grid_names=("m", "n", "s"),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )
    def kernel(x_g, codes_g, *refs):
        *rows_g, even_g, odd_g, even_s, odd_s = refs
        mi, ni, si = (lax.axis_index(name) for name in ("m", "n", "s"))
        k0 = si * steps

        def body(_, x_s, codes_s, *refs):
            *rows_s, carry = refs
            b = codes_s[...].astype(jnp.int32)
            # One load per step: the staging stores would keep two loads
            # in the plane loop apart.
            a = x_s[...]
            out = []
            for p, (acc, w_s) in enumerate(zip(carry, (even_s, odd_s))):
                # The bits of the float32 2**23 + (nibble ^ XOR), so a
                # bitcast takes the place of an int-to-float conversion.
                # The magic number is an idea of FasterTransformer and
                # TensorRT-LLM (`interleaved_numeric_conversion.h`) and
                # of Marlin (`dequant`), which use a 16-bit form.
                q = ((b >> (4 * p)) & 15) ^ (xor | _MAGIC_BITS)
                w = lax.bitcast_convert_type(q, jnp.float32)
                if rows_s:
                    zero = rows_s[0][0, 0, 2 + p, :]
                    w = w - lax.broadcast_in_dim(zero, w.shape, (1,))
                else:
                    w = w - (xor + _MAGIC)
                w_s[...] = w.astype(dtype)
                part = mma(jnp.zeros(acc.shape, jnp.float32), a, w_s[...])
                if rows_s:
                    s = plgpu.load(
                        rows_s[0].at[0, 0, p, :],
                        layout=scale_layout,
                        optimized=False,
                    )
                    part = part * lax.broadcast_in_dim(s, part.shape, (1,))
                out.append(acc + part)
            return tuple(out)

        specs = [
            _tiled((block_m, _BK), dtype, lambda s: (mi, k0 + s)),
            _tiled((_BK, block_n), codes.dtype, lambda s: (k0 + s, ni)),
        ]
        if rows_g:
            specs.append(
                plgpu.BlockSpec(
                    (1, 1, 4, block_n),
                    lambda s: ((k0 + s) // per_group, ni, 0, 0),
                )
            )
        accs = plgpu.emit_pipeline(
            body,
            grid=(steps,),
            in_specs=specs,
            max_concurrent_steps=stages,
            init_carry=(jnp.zeros((block_m, block_n), jnp.float32),) * 2,
        )(x_g, codes_g, *rows_g)
        block = (si, pl.ds(mi * block_m, block_m), pl.ds(ni * block_n, block_n))
        even_g[block], odd_g[block] = accs

    return kernel(x, codes, *rows)


def _transforms(columns, dtype):
    """The tiling and the swizzle of an SMEM tile with `columns` columns."""
    bits = jnp.dtype(dtype).itemsize * 8
    swizzle = plgpu.find_swizzle(columns * bits)
    return (
        plgpu.TilingTransform((8, 8 * swizzle // bits)),
        plgpu.SwizzleTransform(swizzle),
    )


def _tiled(shape, dtype, index_map):
    """A block of a pipeline input, tiled and swizzled in SMEM."""
    return plgpu.BlockSpec(
        shape, index_map, transforms=_transforms(shape[-1], dtype)
    )


# (compute capability, compute dtype, M bucket) -> (BM, BV, stages,
# min_blocks) of the logits, for both orientations: the rows of K1, with
# `BV = 128` table columns per block in place of `BNH`. A block of a row
# input needs at least 128 elements, and with `BV = 256` two stages do
# not fit in 99 KB of SMEM. On JAX 0.12.0, with `BM = 16`, a tied table
# lowers only for one block of rows: a block of 16 columns of the
# transposed activations `(D, Mp)` fails the GMEM transfer plan of Mosaic
# GPU, which needs whole 128-byte cache lines. `BM` 32 and 64 lower. The
# rows give `BM = 16` only for `M <= 16`, and the routes refuse an
# allow-list row with `BM = 16` for a tied call above 16 rows.
MOSAIC_GPU_LOGITS_CONFIGS = {
    key: (block_m, 128, stages, min_blocks)
    for key, (block_m, _, stages, min_blocks) in MOSAIC_GPU_CONFIGS.items()
}


@functools.partial(
    jax.jit,
    static_argnames=(
        "group_size",
        "per_channel",
        "orientation",
        "config",
        "interpret",
    ),
    inline=True,
)
def mosaic_gpu_w4a16_logits(
    x,
    codes,
    scale,
    zero,
    *,
    group_size,
    per_channel,
    orientation,
    config,
    interpret=False,
):
    """Returns the `(M, V)` logits of `x` against the 4-bit table, in
    `x.dtype`.

    Mosaic GPU K-packed, with sum-then-scale. `x` is `(M, D)` in bfloat16
    or float16. `codes` is the stored table, packed along `D`: byte `j`
    holds `d = 2j` in its low nibble and `d = 2j + 1` in its high
    nibble, sign-extended in int8 storage. A `"tied"` table is
    `(V, D / 2)`, with a grouped `(V, n_groups)` multiplier `scale` and
    zero point `zero`. An `"untied"` table is `(D / 2, V)`, with
    `(n_groups, V)`. `d` is in group `d // group_size`. A `per_channel`
    table has the `(V,)` divisor `scale` and no `zero`. `config` is a
    row `(BM, BV, stages, min_blocks)` of `MOSAIC_GPU_LOGITS_CONFIGS`.
    The caller checks the alignment: `D % 128 == 0`, `V % 128 == 0`,
    and a `group_size` that 128 divides. For a tied table with more than
    16 rows, it also checks that `BM` is 32 or 64.

    `interpret=True` (tests only) runs the kernel in GPU interpret mode,
    with `jnp.dot` in place of `plgpu.mma`.
    """
    tied = orientation == "tied"
    block_m, block_v, stages, min_blocks = config
    (m, d), v = x.shape, codes.shape[0 if tied else 1]
    # `V % 128 == 0`, so 128 table columns divide `V`.
    if v % block_v:
        block_v = 128
    padded = -(-m // block_m) * block_m
    blocks = padded // block_m * (v // block_v)
    split = _split_k(d // _BK, blocks, min_blocks)
    xor = 8 if codes.dtype == jnp.int8 else 0
    # Inside each K step of 128 columns: the even columns, then the odd
    # ones, as the staging tile holds the low plane, then the high plane.
    # The even and the odd columns of the activations against the two
    # nibble planes are an idea of TritonBench (`int4_gemm/kernel.py`,
    # `matmul_kernel`), which runs one dot per plane.
    a = jnp.pad(x, ((0, padded - m), (0, 0)))
    a = a.reshape(padded, d // _BK, _BK // 2, 2).swapaxes(2, 3)
    a = a.reshape(padded, d)
    inputs = [a.T if tied else a, codes]
    if not per_channel:
        # Two `(n_groups, V)` float32 inputs: the scale, and the zero point
        # plus `XOR`. An untied float32 scale goes in as stored. A block
        # `(1, BV)` is one contiguous run, as an untiled `cp.async` copy
        # needs. Mosaic GPU has no strided slices, so the tied
        # `(V, n_groups)` arrays transpose here.
        rows = (scale.T, zero.T) if tied else (scale, zero)
        inputs += [rows[0].astype("float32"), rows[1].astype("float32") + xor]
    partials = _logits_kernel(
        *inputs,
        xor=xor,
        per_group=None if per_channel else group_size // _BK,
        tied=tied,
        block_m=block_m,
        block_v=block_v,
        stages=stages,
        split=split,
        interpret=interpret,
    )
    # The partials of the splits sum in a fixed order, so a call is
    # deterministic.
    partials = partials[:, :, :m] if tied else partials[:, :m]
    y = partials[0]
    for i in range(1, split):
        y = y + partials[i]
    if tied:
        y = y.T
    if per_channel:
        y = y / scale.astype("float32")
    return y.astype(x.dtype)


def _logits_kernel(
    x,
    codes,
    *rows,
    xor,
    per_group,
    tied,
    block_m,
    block_v,
    stages,
    split,
    interpret,
):
    """The float32 partials of the logits: `(split, Mp, V)` for an untied
    table, `(split, V, Mp)` for a tied one.

    Each K step reads 64 code bytes, 128 rows of `D`. It widens them,
    splits the planes `(b & 15) ^ XOR` and `((b >> 4) & 15) ^ XOR`, and
    subtracts the zero point plus `XOR` (or `XOR` per channel). It stores
    the low plane in the first half of one SMEM tile along `D`, and the
    high plane in the second half. One MMA contracts the tile with the
    permuted activations, and the scale multiplies the float32 result.
    An untied table is the right operand, `x @ W`. A tied table is the
    left operand, `W @ x^T`, because its tile has `D` along its minor
    axis.
    """
    dtype, half = x.dtype, _BK // 2
    if tied:
        (d, padded), v = x.shape, codes.shape[0]
        stage, acc_shape, out = (block_v, _BK), (block_v, block_m), (v, padded)
        grid, names = (v // block_v, padded // block_m), ("v", "m")
        # A column of the scale that broadcasts across the accumulator.
        scale_layout = plgpu.Layout.MMA_ACC(dtype, m_warps=4).reduce(1)
    else:
        (padded, d), v = x.shape, codes.shape[1]
        stage, acc_shape, out = (_BK, block_v), (block_m, block_v), (padded, v)
        grid, names = (padded // block_m, v // block_v), ("m", "v")
        scale_layout = plgpu.Layout.MMA_ACC(dtype, m_warps=block_m // 16)
        scale_layout = scale_layout.reduce(0)
    # The axis of `V` in the accumulator and in the planes.
    axis = 0 if tied else 1
    steps = d // _BK // split

    def mma(acc, a, b):
        if interpret:
            return acc + jnp.dot(a, b, preferred_element_type=jnp.float32)
        return plgpu.mma(acc, a, b)

    # As for `_w4a16_kernel`, the structure of the kernel is an idea of
    # the sm80 `ragged_dot` kernel of Tokamax
    # (`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm80.py`): an
    # input-only `plgpu.emit_pipeline` over the K steps of a block,
    # `plgpu.mma` on a swizzled SMEM tile, and a store of the accumulator
    # after the pipeline.
    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct((split,) + out, jnp.float32),
        scratch_types=[
            plgpu.SMEM(stage, dtype, transforms=_transforms(stage[1], dtype))
        ],
        grid=grid + (split,),
        grid_names=names + ("s",),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )
    def kernel(x_g, codes_g, *refs):
        *rows_g, out_g, w_s = refs
        mi, vi, si = (lax.axis_index(name) for name in ("m", "v", "s"))
        k0 = si * steps

        def body(_, x_s, codes_s, *refs):
            *rows_s, acc = refs
            b = codes_s[...].astype(jnp.int32)
            for p in (0, 1):
                w = (((b >> (4 * p)) & 15) ^ xor).astype(jnp.float32)
                if rows_s:
                    zero = rows_s[1][0, :]
                    w = w - lax.broadcast_in_dim(zero, w.shape, (axis,))
                else:
                    w = w - xor
                plane = pl.ds(p * half, half)
                w_s[(slice(None), plane) if tied else plane] = w.astype(dtype)
            zeros = jnp.zeros(acc_shape, jnp.float32)
            if tied:
                part = mma(zeros, w_s[...], x_s[...])
            else:
                part = mma(zeros, x_s[...], w_s[...])
            if rows_s:
                s = plgpu.load(
                    rows_s[0].at[0], layout=scale_layout, optimized=False
                )
                part = part * lax.broadcast_in_dim(s, acc_shape, (axis,))
            return acc + part

        if tied:
            specs = [
                _tiled((_BK, block_m), dtype, lambda s: (k0 + s, mi)),
                _tiled((block_v, half), codes.dtype, lambda s: (vi, k0 + s)),
            ]
            block = (pl.ds(vi * block_v, block_v), pl.ds(mi * block_m, block_m))
        else:
            specs = [
                _tiled((block_m, _BK), dtype, lambda s: (mi, k0 + s)),
                _tiled((half, block_v), codes.dtype, lambda s: (k0 + s, vi)),
            ]
            block = (pl.ds(mi * block_m, block_m), pl.ds(vi * block_v, block_v))
        for _ in rows_g:
            specs.append(
                plgpu.BlockSpec(
                    (1, block_v), lambda s: ((k0 + s) // per_group, vi)
                )
            )
        out_g[(si,) + block] = plgpu.emit_pipeline(
            body,
            grid=(steps,),
            in_specs=specs,
            max_concurrent_steps=stages,
            init_carry=jnp.zeros(acc_shape, jnp.float32),
        )(x_g, codes_g, *rows_g)

    return kernel(x, codes, *rows)
