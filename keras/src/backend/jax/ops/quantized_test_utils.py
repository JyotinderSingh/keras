"""Test helpers of the kernels of the JAX backend.

`mosaic_gpu_lowering` lowers the Mosaic GPU kernels for a GPU on a CPU
host, and `pallas_tpu_lowering` the Pallas TPU kernel for a TPU.
`pallas_tpu_interpret` gives TPU interpret mode a TPU on a host without
one. `mosaic_gpu_routes` and `pallas_tpu_route` give routes of the kernels
that run on a CPU host, in interpret mode, and `register_interpret_routes`
adds the interpret routes `"mosaic_gpu_interpret"` and
`"pallas_interpret"` to the route tables of
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


def kernel_record(name):
    """The `Kernel` record `name` of `KERNELS`."""
    (kernel,) = [k for k in quantized.KERNELS if k.name == name]
    return kernel


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
    kernel = kernel_record("mosaic_gpu")
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


@contextlib.contextmanager
def pallas_tpu_lowering(device_kind):
    """Lowers the Pallas TPU kernel for a TPU on a CPU host.

    Inside the block, `jax.export.export(..., platforms=["tpu"])` lowers a
    Pallas TPU kernel for one TPU of `device_kind`, for example
    `"TPU v5 lite"` (v5e) or `"TPU v6 lite"` (v6e). The block runs under
    an abstract mesh of that device. The lowering runs the Pallas TPU
    lowering, but not the libtpu compiler.

    Raises `NotImplementedError` when `jax.sharding.AbstractDevice` has no
    `platform`, as in JAX 0.9.
    """
    try:
        device = jax.sharding.AbstractDevice(
            device_kind=device_kind, num_cores=1, platform="tpu"
        )
    except TypeError as e:
        raise NotImplementedError(
            "`pallas_tpu_lowering` needs an `AbstractDevice` with a "
            f"`platform`, which JAX {jax.__version__} does not have: {e}"
        ) from e
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        yield


@contextlib.contextmanager
def pallas_tpu_interpret():
    """The TPU of Pallas TPU interpret mode on a host without a TPU.

    A Pallas TPU kernel reads the TPU generation when it traces. JAX
    0.11.2 and newer read it from the private chip registry
    `jax._src.tpu_info.registry`, under the device kind of the default
    device, so inside the block that device kind (`"cpu"`, or a GPU kind
    such as `"NVIDIA L4"`) gives a TPU v5e. `get_tpu_info` caches its
    result, so the block clears that cache on entry and on exit: the TPU
    v5e does not stay after the block. Older JAX reads the TPU from an
    abstract mesh, so the block runs inside
    `pallas_tpu_lowering("TPU v5 lite")`. The route `"pallas_interpret"`
    enters the block for each call.

    Raises `NotImplementedError` when neither form works.
    """
    tpu_info = getattr(jax._src, "tpu_info", None)
    if tpu_info is None:
        with pallas_tpu_lowering("TPU v5 lite"):
            yield
        return
    try:
        chip = tpu_info.ChipVersion.TPU_V5E
        info = tpu_info.get_tpu_info_for_chip(chip, 1)
        registry = tpu_info.registry
        kind = tpu_info.get_device_kind()
        clear_cache = tpu_info.get_tpu_info.cache_clear
    except AttributeError as e:
        raise NotImplementedError(
            "`pallas_tpu_interpret` needs a TPU chip registry name that JAX "
            f"{jax.__version__} does not have: {e}"
        ) from e
    previous = registry.get(kind)
    registry[kind] = lambda: info
    clear_cache()
    try:
        yield
    finally:
        if previous is None:
            del registry[kind]
        else:
            registry[kind] = previous
        clear_cache()


@functools.lru_cache(maxsize=None)
def pallas_tpu_interpret_available():
    """Whether `pallas_tpu_interpret` works with the installed JAX."""
    try:
        with pallas_tpu_interpret():
            return True
    except NotImplementedError:
        return False


def pallas_tpu_route(interpret=True, precision=None, **changes):
    """The matmul route of the Pallas TPU kernel, for a host without a TPU.

    The route is that of `kernels._kernel_routes`, with no `kernel`, so
    `"require"` does not run it. With `interpret`, it runs the kernel in
    TPU interpret mode inside `pallas_tpu_interpret`, and its device check
    is `pallas_tpu_interpret_available`. Without `interpret`, it runs the
    kernel, and its device check passes on each device: for a lowering
    inside `pallas_tpu_lowering`, or on a TPU. With `precision`, each call
    runs under `jax.default_matmul_precision(precision)`. `changes`
    replace fields of the route, for example `tested=None`.
    """
    kernel = kernel_record("pallas_tpu")
    function = kernel.matmul

    def matmul(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            if precision is not None:
                stack.enter_context(jax.default_matmul_precision(precision))
            if interpret:
                stack.enter_context(pallas_tpu_interpret())
                kwargs["interpret"] = quantized.pltpu.InterpretParams()
            return function(*args, **kwargs)

    def supports_device(device_class):
        return not interpret or pallas_tpu_interpret_available()

    kernel = dataclasses.replace(
        kernel, matmul=matmul, supports_device=supports_device
    )
    route, _ = kernels._kernel_routes(kernel)
    changes.setdefault("kernel", None)
    return dataclasses.replace(route, **changes)


def register_interpret_routes():
    """Adds the interpret routes to the route tables, and returns the
    name of the kernel route that each one stands in for.

    `"mosaic_gpu_interpret"` runs the Mosaic GPU kernels in GPU interpret
    mode on CPU, with `jnp.dot` in place of `plgpu.mma`, as a matmul and a
    logits route. The matmul route does not support `jax.vmap` of a
    guarded call (a view with a `g_idx`): interpret mode has an IO effect,
    which JAX does not support in a vmap-of-cond.

    `"pallas_interpret"` runs the Pallas TPU kernel in TPU interpret mode
    on a host without a TPU, as a matmul route. On such a host the
    reference path contracts float32 in float32, so the route runs its
    float32 dots at `HIGHEST` (`"highest"`). It does not support
    `jax.vmap`, as the kernel.

    The routes run through `force_route` or a measured row. They have no
    version range.
    """
    matmul, logits = mosaic_gpu_routes(tested=None)
    kernels._ROUTES["mosaic_gpu_interpret"] = matmul
    kernels._LOGITS_ROUTES["mosaic_gpu_interpret"] = logits
    kernels._ROUTES["pallas_interpret"] = pallas_tpu_route(
        precision="highest", tested=None
    )
    return {
        "mosaic_gpu_interpret": "mosaic_gpu",
        "pallas_interpret": "pallas_tpu",
    }
