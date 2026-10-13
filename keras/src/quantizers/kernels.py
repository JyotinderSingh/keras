"""Routes for a 4-bit weight-only contraction.

A route runs one int4 contraction in place of the reference path, which
dequantizes the weight and contracts it. A route is a backend kernel or
a formulation in `keras.ops`. It runs forward only: its
caller owns the gradient, and the backward does not call the route. A
mode's projection reaches it through `contract_quantized` of
`keras.src.quantizers.modes.common`, and the reverse lookup through
`LookupHandlers._contract_routed`.

Static checks select the route and never read a tensor value. In order:
the policy allows routes (`_policy_allows`); the codes pack two per byte
in a grouped or per-channel scheme (`_int4_pairs_layout`); the geometry
describes the contraction as a matmul
(`ProjectionGeometry.matrix_contraction`); the contraction has at least
one row (`_has_rows`); no distribution is active and no operand is
sharded (`_not_distributed`); under `"auto"`, a measured row names the
route (`_measured_row`); and the route supports the compute dtype
(`_dtype_supported`), the device and the shape (`_route_available`). The
first route of the row whose checks pass runs.

The policy lives in `backend.config`. `"auto"` (the default) runs the
routes of the measured rows. `"off"` runs the reference path:
`keras.config.disable_quantized_kernels()` sets it. `"require"` is
private, for CI (`pytest --kernel-policy=require`, from the root
`conftest.py`). It runs the platform's kernels and ignores the measured
rows and the version range. A device that no kernel supports raises when
`"require"` is set. A call whose shape or dtype each kernel refuses runs
the reference path, and so does a reverse lookup on a device with no
logits kernel.

A backend's kernels are the `Kernel` records of `KERNELS` in its
`ops/quantized.py` (`keras.src.backend.common.quantized_kernels`).
`_kernel_routes` turns each record into a route of `_ROUTES` and, when it
has a logits variant, a route of `_LOGITS_ROUTES`.

`route_counts` counts the selections per route name, `"reference"`
included, for tests. On JAX a selection happens at trace time. Under
`torch.compile` the counter does not count, because Dynamo would guard on
its value and compile again on each call.
"""

import collections
import contextlib
import dataclasses
import functools
import math
import platform

from keras.src import backend
from keras.src import ops
from keras.src.backend import config
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import m_bucket
from keras.src.distribution import distribution_lib
from keras.src.quantizers.quantized_weight import Int4Pairs

# The kernel module of the backend (`KERNELS`, `device_class`,
# `is_sharded`, `is_compiling`), or `None` for a backend with no kernels.
_KERNEL_MODULE = getattr(getattr(backend, "ops", None), "quantized", None)
# (backend, device class, compute dtype, M bucket) -> (route name, ...)
# A row names, in order of preference, the routes that ran faster than the
# reference path at each measured shape of its cell and passed the route
# tests there. `"auto"` runs a route only where a row names it.
_MEASURED_ROWS = {}
# The rows of the reverse lookup, as `_MEASURED_ROWS`. The compute dtype is
# the reverse dtype of the layer, when it has one.
_MEASURED_LOGITS_ROWS = {}
route_counts = collections.Counter()


class _State:
    # The route that `force_route` forces, or `None`.
    route = None


_state = _State()


@dataclasses.dataclass(frozen=True)
class Route:
    """The code of one route and the checks that it adds.

    A `matrix` route runs `fn(quantized_weight, inputs, dtype)` forward:
    it takes the inputs as `(M, K)` in the compute dtype `dtype` and
    returns `(M, N)` in stored column order. A route of the reverse
    lookup takes the stored table and the `(M, D)` inputs and returns
    `(M, V)`. Any other route runs `fn(layer, geometry, quantized_weight,
    inputs)` and takes and returns the shapes of the layer.

    `dtypes` holds the compute dtypes that the route supports (empty for
    every dtype). `available(device_class, m, k, n, quantized_weight,
    dtype)` gives its device and shape checks. `tested()` says whether
    the installed compiler is in the tested range of the route. `kernel`
    is the backend `Kernel` of the route, or `None`: the `"require"`
    policy runs kernels only.
    """

    fn: object
    matrix: bool = True
    dtypes: tuple = ()
    available: object = None
    tested: object = None
    kernel: object = None


@dataclasses.dataclass(frozen=True)
class Selection:
    """A selected route.

    `rows` is the contracted size `K`, `contracted` the number of
    trailing input axes that it spans, `columns` the free kernel
    dimensions in kernel order, and `dtype` the compute dtype.
    """

    name: str
    route: Route
    rows: int
    contracted: int
    columns: tuple
    dtype: str


def _reference(layer, geometry, quantized_weight, inputs):
    """The reference path: `dequantize`, then `geometry.contract`."""
    float_kernel = quantized_weight.dequantize(layer.compute_dtype)
    return geometry.contract(inputs, float_kernel)


# Route name -> `Route` of a projection.
_ROUTES = {}
# Route name -> `Route` of a reverse lookup.
_LOGITS_ROUTES = {}


@contextlib.contextmanager
def policy(value):
    """Runs a block, or decorates a function, under the policy `value`:
    `"auto"`, `"off"` or `"require"`.

    A route is selected at trace time. On JAX and TensorFlow, a policy
    change applies only to functions that are traced after the change.
    """
    _check_policy(value)
    previous = config._quantized_kernels_policy()
    config._set_quantized_kernels_policy(value)
    try:
        yield
    finally:
        config._set_quantized_kernels_policy(previous)


@contextlib.contextmanager
def force_route(name):
    """Runs a block under `"auto"` with the route `name` forced, for tests
    and benchmarks.

    The forced route runs wherever its own checks pass: the layout, the
    contraction, the rows, the dtype, the device and the shape. It skips
    the distribution check, the measured rows and the version range.
    """
    if name not in _ROUTES and name not in _LOGITS_ROUTES:
        raise ValueError(
            "Unknown route. Expected one of "
            f"{sorted(set(_ROUTES) | set(_LOGITS_ROUTES))}. "
            f"Received: name={name!r}"
        )
    previous = _state.route
    _state.route = name
    try:
        with policy("auto"):
            yield
    finally:
        _state.route = previous


def _check_policy(value):
    """Under `"require"`, raises when no kernel supports the device. The
    value itself is checked by `config`."""
    if value != "require" or _required(_ROUTES, _device_class()):
        return
    kernels = [
        f"{route.kernel.name} needs {route.kernel.requirements}"
        for route in _ROUTES.values()
        if route.kernel is not None
    ]
    raise ValueError(
        "The kernel policy 'require' needs a platform with a kernel. "
        f"The {backend.backend()} backend has no kernel for the device "
        f"class {_device_class()!r}."
        + (f" Kernels: {'; '.join(kernels)}." if kernels else "")
    )


def _required(routes, device_class):
    """The routes of `routes` that `"require"` runs on `device_class`: the
    kernels that support it, in table order."""
    return tuple(
        name
        for name, route in routes.items()
        if route.kernel is not None
        and route.kernel.supports_device(device_class)
    )


def select_route(layer, geometry, quantized_weight, inputs):
    """Returns the `Selection` of a projection, or `None` for the
    reference path."""

    def contraction():
        if not _int4_pairs_layout(quantized_weight):
            return None
        return geometry.matrix_contraction()

    selection = _select(
        _ROUTES,
        _MEASURED_ROWS,
        quantized_weight,
        inputs,
        layer.compute_dtype,
        contraction,
    )
    _count(selection)
    return selection


def select_logits_route(layer, table, inputs):
    """Returns the `Selection` of a reverse lookup, or `None` for the
    reference path.

    The caller, the reverse lookup of `ReversibleEmbedding` with no
    activation quantizer, gives the stored table: the forward table
    `(V, D)` of a tied layer, or the reverse table `(D, V)` of an untied
    one. `inputs` are `(..., D)` in the reverse dtype.
    """

    def contraction():
        if not isinstance(table.layout, Int4Pairs):
            return None
        vocabulary = table.shape[0 if _tied(table) else 1]
        return table.layout.orig_len, 1, (vocabulary,)

    selection = _select(
        _LOGITS_ROUTES,
        _MEASURED_LOGITS_ROWS,
        table,
        inputs,
        layer.reverse_dtype or layer.compute_dtype,
        contraction,
    )
    _count(selection)
    return selection


def _select(routes, measured_rows, view, inputs, dtype, contraction):
    """The first route of `routes` whose checks pass, as a `Selection`.

    `contraction()` gives `(rows, contracted, columns)`, or `None` when
    no route reproduces the contraction. It runs after the policy check,
    so the reference path reads nothing more of the call.
    """
    forced, current = _state.route, config._quantized_kernels_policy()
    if not _policy_allows(current, forced, measured_rows):
        return None
    described = contraction()
    if described is None:
        return None
    rows, contracted, columns = described
    free = tuple(inputs.shape)[: len(inputs.shape) - contracted]
    if not _has_rows(free):
        return None
    m, n = math.prod(free), math.prod(columns)
    device_class = _device_class()
    if forced:
        candidates = (forced,)
    elif not _not_distributed(inputs, view):
        return None
    elif current == "require":
        candidates = _required(routes, device_class)
    else:
        candidates = _measured_row(measured_rows, device_class, dtype, m)
    # A forced route and `"require"` skip the version range.
    measured = not forced and current == "auto"
    for name in candidates:
        # A forced name can belong to the other table.
        route = routes.get(name)
        if (
            route is not None
            and _dtype_supported(route, dtype, inputs)
            and _route_available(
                route, device_class, m, rows, n, view, dtype, measured
            )
        ):
            return Selection(name, route, rows, contracted, columns, dtype)
    return None


def _policy_allows(current, forced, measured_rows):
    """`"off"` runs the reference path, and so does `"auto"` with no
    measured row and no forced route."""
    return current != "off" and bool(
        forced or current == "require" or measured_rows
    )


def _has_rows(free):
    """The free input axes are known and hold at least one row."""
    return all(dim is not None for dim in free) and math.prod(free) >= 1


def _int4_pairs_layout(view):
    """The codes are 2-D and pack two 4-bit codes per byte along the
    columns. The scheme is grouped (a multiplier and a zero point), or
    per-channel (a divisor and no zero point)."""
    scheme = view.scheme
    # `WeightScheme` gives a group size only to a multiplier scheme with a
    # zero point.
    grouped = scheme.group_size is not None and view.axis == 0
    per_channel = (
        scheme.scale_form == "divisor"
        and not scheme.has_zero_point
        and view.scale_axes == (1,)
    )
    return (
        isinstance(view.layout, Int4Pairs)
        and view.layout.axis in (-1, 1)
        and len(view.codes.shape) == 2
        and (grouped or per_channel)
    )


def _not_distributed(inputs, view):
    """No Keras distribution is active, and no operand is sharded."""
    if distribution_lib.distribution() is not None:
        return False
    is_sharded = getattr(_KERNEL_MODULE, "is_sharded", None)
    if is_sharded is None:
        return True
    # The view of a call holds the layer's variables.
    codes = getattr(view.codes, "value", view.codes)
    return not any(is_sharded(tensor) for tensor in (inputs, codes))


def _dtype_supported(route, dtype, inputs):
    """The route supports the compute dtype. The inputs of a `matrix`
    route have the compute dtype: a layer with `autocast=False` leaves
    its inputs uncast."""
    if route.dtypes and dtype not in route.dtypes:
        return False
    return not route.matrix or standardize_dtype(inputs.dtype) == dtype


def _route_available(route, device_class, m, k, n, view, dtype, measured):
    """The device and shape checks of the route, and, for a `measured`
    selection, the tested version range of its compiler."""
    if route.available is not None:
        if not route.available(device_class, m, k, n, view, dtype):
            return False
    return not measured or route.tested is None or route.tested()


def _measured_row(measured_rows, device_class, dtype, m):
    """The route names of the measured row of the call.

    The M bucket comes from the input shape, so it is read only where the
    platform has a row. Elsewhere, a compiled call does not guard on it
    and does not compile again for a new bucket.
    """
    platform = (backend.backend(), device_class, dtype)
    if not any(key[:3] == platform for key in measured_rows):
        return ()
    return measured_rows.get(platform + (m_bucket(m),), ())


def _is_compiling():
    """Whether a compiler that guards on Python state traces the call:
    Dynamo under `torch.compile`."""
    is_compiling = getattr(_KERNEL_MODULE, "is_compiling", None)
    return is_compiling is not None and is_compiling()


def _count(selection):
    """Counts a selection in `route_counts`, except under a compiler that
    guards on Python state."""
    if not _is_compiling():
        name = "reference" if selection is None else selection.name
        route_counts[name] += 1


def _device_class():
    """The device class of the call: `cuda-sm<major><minor>`,
    `mps`, `gpu` or `cpu-<machine>`.

    The kernel module of the backend says it. A backend with no kernel
    module reads its default device.
    """
    device_class = getattr(_KERNEL_MODULE, "device_class", None)
    if device_class is not None:
        return device_class()
    return _default_device_class()


@functools.lru_cache(maxsize=None)
def _default_device_class():
    """`gpu` when the default devices are GPUs, else `cpu-<machine>`."""
    if getattr(backend, "distribution_lib", None) is not None:
        if distribution_lib.list_devices()[0].startswith("gpu"):
            return "gpu"
    return f"cpu-{platform.machine()}"


def _tied(table):
    """Whether a lookup table is the forward table `(V, D)` of a tied layer:
    it packs its codes along axis 1."""
    return table.layout.axis % 2 == 1


def run_route(selection, layer, geometry, quantized_weight, inputs):
    """Runs `selection` forward only."""
    route = selection.route
    if not route.matrix:
        return route.fn(layer, geometry, quantized_weight, inputs)
    free = ops.shape(inputs)[: len(inputs.shape) - selection.contracted]
    x = ops.reshape(inputs, (-1, selection.rows))
    y = route.fn(quantized_weight, x, selection.dtype)
    return ops.reshape(y, tuple(free) + selection.columns)


def _kernel_routes(kernel):
    """The routes of a backend `Kernel`: its matmul, and its logits or
    `None`.

    Each route unpacks the stored tensors of a `QuantizedWeight` for the
    kernel.
    """

    def matmul(view, inputs, dtype):
        return kernel.matmul(
            inputs,
            view.codes,
            view.scale,
            view.zero_point,
            group_size=view.scheme.group_size,
            signed=standardize_dtype(view.codes.dtype) == "int8",
            per_channel=view.scheme.group_size is None,
            input_scales=view.input_scales,
            g_idx=view.g_idx,
        )

    def logits(table, inputs, dtype):
        return kernel.logits(
            inputs,
            table.codes,
            table.scale,
            table.zero_point,
            group_size=table.scheme.group_size,
            per_channel=table.scheme.group_size is None,
            orientation="tied" if _tied(table) else "untied",
        )

    def available(kernel_available):
        def check(device_class, m, k, n, view, dtype):
            return kernel.supports_device(device_class) and kernel_available(
                device_class, m, k, n, view.scheme.group_size, dtype
            )

        return check

    route = Route(
        matmul,
        dtypes=kernel.dtypes,
        available=available(kernel.available),
        tested=kernel.tested,
        kernel=kernel,
    )
    if kernel.logits is None:
        return route, None
    return route, dataclasses.replace(
        route, fn=logits, available=available(kernel.logits_available)
    )


def _register_kernels():
    """Adds the routes of the backend's `KERNELS` to the route tables."""
    for kernel in getattr(_KERNEL_MODULE, "KERNELS", ()):
        _ROUTES[kernel.name], logits = _kernel_routes(kernel)
        if logits is not None:
            _LOGITS_ROUTES[kernel.name] = logits


_register_kernels()
