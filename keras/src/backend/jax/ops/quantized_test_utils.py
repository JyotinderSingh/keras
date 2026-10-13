"""Test helpers of the Mosaic GPU kernels of the JAX backend.

`mosaic_gpu_lowering` lowers the kernels for a GPU on a CPU host.
`mosaic_gpu_routes` gives routes of the kernels that run on a CPU host,
in GPU interpret mode, and `register_interpret_routes` adds the interpret
routes `"mosaic_gpu_interpret"` to the route tables of
`keras.src.quantizers.kernels`. `kernels_test_utils` calls it at import.
"""

import contextlib
import dataclasses
import functools

import jax

from keras.src.backend.jax.ops import quantized
from keras.src.quantizers import kernels


@contextlib.contextmanager
def mosaic_gpu_lowering(compute_capability):
    """Lowers the Mosaic GPU kernels for a GPU on a CPU host.

    Inside the block, `jax.export.export(..., platforms=["cuda"])` lowers
    a Mosaic GPU kernel for `compute_capability`, for example `"8.9"`. It
    runs the Pallas lowering, the pipeline of that capability (`cp.async`
    below 9.0, TMA from 9.0) and the Mosaic GPU layout inference, but no
    `ptxas`. `AbstractDevice` has no compute capability, so the helper
    sets the private `jax.experimental.mosaic.gpu.utils._infer_arch`.

    It also wraps the private `LaunchContext.async_copy`. An untiled copy
    from GMEM reads its block as the base address plus the flat SMEM
    index, so a block that is not one contiguous run of its GMEM array
    lowers, passes interpret mode, and reads the wrong bytes on a GPU.
    The wrap raises `ValueError` for such a block. It changes nothing
    else.

    The pipeline reads the capability when `mosaic_gpu_w4a16_matmul` or
    `mosaic_gpu_w4a16_logits` traces, and the functions keep their
    traces. So the helper clears them on entry and on exit: a trace for
    another capability does not serve the block, and a trace of the block
    does not serve a later call.

    Raises `NotImplementedError` when JAX has no `plgpu.mma`, or when a
    private name moved.
    """
    try:
        if not quantized.plgpu.available:
            raise AttributeError("`plgpu.mma`")
        mosaic = jax.experimental.mosaic.gpu
        utils, context = mosaic.utils, mosaic.launch_context.LaunchContext
        infer_arch, async_copy = utils._infer_arch, context.async_copy
    except AttributeError as e:
        raise NotImplementedError(
            "`mosaic_gpu_lowering` needs a Mosaic GPU name that JAX "
            f"{jax.__version__} does not have: {e}"
        ) from e
    arch = tuple(int(part) for part in compute_capability.split("."))

    def checked_copy(self, *, src_ref, gmem_slice=(), **kwargs):
        source, transform = src_ref.type, kwargs.get("gmem_transform")
        if source.memory_space is None and not transform:
            shape = tuple(source.shape)
            if not _contiguous_block(shape, gmem_slice):
                raise ValueError(
                    "An untiled GMEM to SMEM copy reads a block that is not "
                    f"one contiguous run of its array. Received: array "
                    f"shape={shape}, block={_block_shape(shape, gmem_slice)}"
                )
        return async_copy(
            self, src_ref=src_ref, gmem_slice=gmem_slice, **kwargs
        )

    traced = (
        quantized._mosaic_gpu_w4a16_matmul,
        quantized._mosaic_gpu_w4a16_logits,
    )
    for function in traced:
        function.clear_cache()
    utils._infer_arch = lambda: arch
    context.async_copy = checked_copy
    try:
        yield
    finally:
        utils._infer_arch, context.async_copy = infer_arch, async_copy
        for function in traced:
            function.clear_cache()


def _block_shape(shape, gmem_slice):
    """The length of a GMEM block along each axis of its array."""
    index = gmem_slice if isinstance(gmem_slice, tuple) else (gmem_slice,)
    index += (slice(None),) * (len(shape) - len(index))
    return tuple(
        len(range(size)[item])
        if isinstance(item, slice)
        else getattr(item, "length", 1)
        for size, item in zip(shape, index)
    )


def _contiguous_block(shape, gmem_slice):
    """Whether a block is one contiguous run of a row-major array: each
    axis after the first axis longer than 1 spans its whole array axis."""
    longer = False
    for length, size in zip(_block_shape(shape, gmem_slice), shape):
        if longer and length != size:
            return False
        longer = longer or length != 1
    return True


def mosaic_gpu_routes(interpret=True, config=None, **changes):
    """The matmul and the logits route of the Mosaic GPU kernels, for a
    CPU host.

    The routes are those of `kernels._kernel_routes`, with the device
    check `plgpu.available` in place of the GPU check and no `kernel`, so
    `"require"` does not run them. With `interpret`, they run the kernels
    in GPU interpret mode. With `config`, the kernels take that launch
    configuration, else the row of their table: on a CPU host, the rows
    of compute capability 8.9 in interpret mode. `changes` replace fields
    of both routes, for example `tested=None`.
    """
    (kernel,) = [k for k in quantized.KERNELS if k.name == "mosaic_gpu"]
    options = {"interpret": interpret}
    if config is not None:
        options["config"] = config
    kernel = dataclasses.replace(
        kernel,
        matmul=functools.partial(kernel.matmul, **options),
        logits=functools.partial(kernel.logits, **options),
        supports_device=lambda device_class: quantized.plgpu.available,
    )
    matmul, logits = kernels._kernel_routes(kernel)
    changes.setdefault("kernel", None)
    return (
        dataclasses.replace(matmul, **changes),
        dataclasses.replace(logits, **changes),
    )


def register_interpret_routes():
    """Adds `"mosaic_gpu_interpret"` to the matmul and the logits routes,
    and returns `{"mosaic_gpu_interpret": "mosaic_gpu"}`.

    The routes run the kernels in GPU interpret mode on CPU, with
    `jnp.dot` in place of `plgpu.mma`, through `force_route` or a
    measured row. They have no version range. The matmul route does not
    support `jax.vmap` of a guarded call (a view with a `g_idx`):
    interpret mode has an IO effect, which JAX does not support in a
    vmap-of-cond.
    """
    matmul, logits = mosaic_gpu_routes(tested=None)
    kernels._ROUTES["mosaic_gpu_interpret"] = matmul
    kernels._LOGITS_ROUTES["mosaic_gpu_interpret"] = logits
    return {"mosaic_gpu_interpret": "mosaic_gpu"}
