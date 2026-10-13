"""Quantized contraction kernels of the JAX backend.

`keras.src.quantizers.kernels` runs each kernel of `KERNELS` as a route
(`keras.src.backend.common.quantized_kernels`). A kernel runs forward
only: its caller runs it inside a custom gradient whose backward does not
call the kernel.
"""

import functools
import platform

import jax
import jax.numpy as jnp
import jaxlib
from jax import lax
from jax.experimental import pallas as pl
from packaging.version import Version

from keras.src.backend.common.backend_utils import m_bucket
from keras.src.backend.common.quantized_kernels import Kernel
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


# Under `"auto"`, the Mosaic GPU kernels run only with equal jax and
# jaxlib versions in this range: the Mosaic GPU compiler ships in jaxlib,
# and a jax release accepts an older jaxlib. A version enters the range
# after the tests pass on it under `"require"` on a GPU.
MOSAIC_GPU_TESTED_JAX = ("0.11.2", "0.12.0")
# The CUDA compute capabilities that the Mosaic GPU kernels run on. A ROCm
# device reports a `gfx` name.
_MOSAIC_GPU_CAPABILITIES = ("8.0", "8.6", "8.9", "9.0")


def mosaic_gpu_tested(versions=None, prefixes=MOSAIC_GPU_TESTED_JAX):
    """Whether a version is in a tested range: by default, the installed
    jax and jaxlib in `MOSAIC_GPU_TESTED_JAX`.

    `versions` is a version string, or a pair of versions that must be
    equal. `prefixes` holds release prefixes: `"0.12"` covers every 0.12.x
    release. A pre-release or a development release is never in the
    range.
    """
    if versions is None:
        versions = (jax.__version__, jaxlib.__version__)
    versions = {versions} if isinstance(versions, str) else set(versions)
    if len(versions) != 1:
        return False
    version = Version(versions.pop())
    return not version.is_prerelease and any(
        version.release[: len(prefix.split("."))]
        == tuple(int(part) for part in prefix.split("."))
        for prefix in prefixes
    )


def _capability(interpret):
    """The compute capability of the rows of a call: that of the default
    device, or 8.9 in interpret mode, which runs on CPU."""
    return "8.9" if interpret else jax.devices()[0].compute_capability


def mosaic_gpu_w4a16_matmul(
    x,
    codes,
    scale,
    zero,
    *,
    group_size,
    signed,
    per_channel,
    config=None,
    input_scales=None,
    interpret=False,
):
    """Returns the `(M, N)` contraction of `x` against the 4-bit weight, in
    `x.dtype`.

    The Mosaic GPU matmul kernel, with sum-then-scale. `x` is `(M, K)` in
    bfloat16 or float16. `codes` is the stored `(K, N / 2)` tensor: byte
    `j` holds column `2j` in its low nibble and column `2j + 1` in its
    high nibble, with `signed` nibbles in int8 storage. A grouped weight
    has the `(n_groups, N)` multiplier `scale` and the int8 or uint8 zero
    point `zero`, and row `k` is in group `k // group_size` (a
    `group_size` of `K` or more is one group). A `per_channel` weight has
    the `(N,)` divisor `scale` and no `zero`. AWQ's `input_scales` divide
    `x` in its dtype. `config` is a row `(BM, BNH, stages, min_blocks)`,
    or `None` for the row of `MOSAIC_GPU_CONFIGS` for the device, the
    dtype of `x` and the M bucket. The caller checks the alignment:
    `K % 128 == 0`, `N % 64 == 0`, and a `group_size` that 128 divides.

    `interpret=True` (tests only) runs the kernel in GPU interpret mode,
    with `jnp.dot` in place of `plgpu.mma`, which has no interpret rule.
    It takes the rows of compute capability 8.9.
    """
    m, k = x.shape
    if config is None:
        key = (_capability(interpret), jnp.dtype(x.dtype).name, m_bucket(m))
        config = MOSAIC_GPU_CONFIGS[key]
    return _mosaic_gpu_w4a16_matmul(
        x,
        codes,
        scale,
        zero,
        group_size=None if per_channel else min(group_size, k),
        signed=signed,
        per_channel=per_channel,
        config=config,
        input_scales=input_scales,
        interpret=interpret,
    )


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
def _mosaic_gpu_w4a16_matmul(
    x,
    codes,
    scale,
    zero,
    *,
    group_size,
    signed,
    per_channel,
    config,
    input_scales,
    interpret,
):
    """`mosaic_gpu_w4a16_matmul` with its configuration."""
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


@functools.lru_cache(maxsize=None)
def device_class():
    """The class of the default device: `cuda-sm<major><minor>`,
    `tpu-<kind>`, `cpu-<machine>`, or the platform."""
    device = jax.devices()[0]
    capability = getattr(device, "compute_capability", "")
    if capability.replace(".", "").isdigit():
        return "cuda-sm" + capability.replace(".", "")
    if device.platform == "tpu":
        return f"tpu-{device.device_kind}"
    if device.platform == "cpu":
        return f"cpu-{platform.machine()}"
    return device.platform


def _mosaic_gpu_supports_device(device_class):
    """An NVIDIA GPU of a compute capability in the list, and a JAX with
    `plgpu.mma`."""
    capabilities = {
        "cuda-sm" + capability.replace(".", "")
        for capability in _MOSAIC_GPU_CAPABILITIES
    }
    return device_class in capabilities and plgpu.available


def _mosaic_gpu_aligned(k, n, group_size):
    """The alignment gate: whole K steps of 128 rows, whole blocks of 32
    code columns, and groups of whole K steps. A per-channel weight has
    no `group_size`."""
    group = k if group_size is None else min(group_size, k)
    return n % 64 == 0 and k % 128 == 0 and group % 128 == 0


def _mosaic_gpu_available(device_class, m, k, n, group_size, dtype):
    """The alignment gate of `mosaic_gpu_w4a16_matmul`."""
    return _mosaic_gpu_aligned(k, n, group_size)


# The kernels of this backend.
KERNELS = (
    Kernel(
        name="mosaic_gpu",
        matmul=mosaic_gpu_w4a16_matmul,
        dtypes=("bfloat16", "float16"),
        supports_device=_mosaic_gpu_supports_device,
        requirements=(
            "an NVIDIA GPU of compute capability "
            f"{', '.join(_MOSAIC_GPU_CAPABILITIES)} and a JAX with "
            "`plgpu.mma` (0.11.2 or newer)"
        ),
        available=_mosaic_gpu_available,
        tested=mosaic_gpu_tested,
        contiguous_groups=True,
    ),
)
