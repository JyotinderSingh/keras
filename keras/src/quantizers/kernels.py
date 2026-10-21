"""Routes for a quantized contraction.

A route runs one 4-bit weight-only contraction in place of the reference
path, which dequantizes the weight and calls `geometry.contract`. A route
is a kernel, a formulation in `keras.ops` or a native change. It runs
forward only: its caller runs it inside a custom gradient whose backward
dequantizes the weight and does not call the route.

Static gates select the route. They read the backend, the device, the
dtypes, the shapes, the layout and scheme of the weight, the geometry,
the policy and the library versions, and never a tensor value:

1. The policy is not `"off"`. The export entry points trace under
   `"off"`.
2. The contraction has at least one row (`M >= 1`).
3. The geometry has the standard `contract` and `contract_grad`: a
   `ProjectionGeometry` with the default kernel axes, or an
   `EinsumProjectionGeometry` whose equation is `reshape_compatible`.
4. The codes are 2-D and pack two 4-bit codes per byte along the
   columns. The scheme is grouped (a multiplier and a zero point), or
   per-channel (a divisor, no zero point and no activation quantizer).
5. The route is available: its device and shape checks pass, and under
   `"auto"` its library version is in its tested range (`tested`).
6. No Keras distribution is active, and no operand is a DTensor.
7. The route supports the compute dtype.
8. Under `"auto"`, `_ALLOW_LIST` has a row for the backend, the device
   class, the compute dtype and the M bucket. A row holds measured
   routes only, as an ordered tuple of `(route, configuration)`. The
   selection takes the first route of the row whose gates pass.

The policy is private. `"auto"` (the default) runs the route of the
allow-list row, `"off"` runs the reference path, and `"require"` runs
the platform's kernel. `"require"` ignores the allow-list and the
version range, and `set_policy("require")` raises on a platform without
a kernel.

`route_counts` counts the selections per route name for tests. On JAX a
selection happens at trace time. Under `torch.compile` the counter does
not count, because Dynamo would guard on its value and recompile on each
call.
"""

import collections
import contextlib
import dataclasses
import functools
import math
import platform

from packaging.version import Version

from keras.src import backend
from keras.src import ops
from keras.src.distribution import distribution_lib
from keras.src.quantizers.geometry import EinsumProjectionGeometry
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.utils.module_utils import tensorflow as tf

if backend.backend() == "torch":
    import torch
    from torch.distributed.tensor import DTensor
elif backend.backend() == "jax":
    import jax

POLICIES = ("auto", "off", "require")
# The largest M of each bucket; a larger M is in the bucket "257+".
_M_BUCKETS = ((16, "1-16"), (64, "17-64"), (256, "65-256"))
# (backend, device class, compute dtype, M bucket) -> ((route, config), ...)
_ALLOW_LIST = {}
# backend -> {device type of `list_devices`: the kernel of `"require"`}
_REQUIRED = {}
# The selections per route name, "reference" included. On JAX a selection
# happens at trace time. Under `torch.compile` the counter does not count.
# Tests read the counter.
route_counts = collections.Counter()


class _State:
    policy = "auto"
    # A route that `policy(..., route=...)` forces, for tests only.
    route = None


_state = _State()


@dataclasses.dataclass(frozen=True)
class RouteSource:
    """The code of one route and the gates that it adds.

    `fn(layer, geometry, quantized_weight, inputs, config)` runs the
    contraction forward. A `matrix` source takes the inputs as `(M, K)`
    and returns `(M, N)` in stored column order. Any other source takes
    and returns the shapes of the layer. `dtypes` holds the compute dtypes
    that the source supports (gate 7, empty for every dtype), and
    `available(m, k, n, quantized_weight)` its device and shape checks
    (gate 5). `version()` returns the installed library version, or the
    `(jax, jaxlib)` pair, and `tested_versions` holds the release
    prefixes of the tested range.
    """

    fn: object
    matrix: bool = True
    dtypes: tuple = ()
    available: object = None
    version: object = None
    tested_versions: tuple = ()


@dataclasses.dataclass(frozen=True)
class Route:
    """A selected route.

    `rows` is the contracted size `K`, `contracted` the number of
    trailing input axes that it spans, and `columns` the free kernel
    dimensions in kernel order.
    """

    name: str
    source: RouteSource
    config: object
    rows: int
    contracted: int
    columns: tuple


def _reference(layer, geometry, quantized_weight, inputs, config=None):
    """The reference path: `dequantize`, then `geometry.contract`."""
    float_kernel = quantized_weight.dequantize(layer.compute_dtype)
    return geometry.contract(inputs, float_kernel)


# Route name -> `RouteSource`. "reference_wrapped" runs the reference path
# as a route, through `force_route` only.
_ROUTES = {"reference_wrapped": RouteSource(_reference, matrix=False)}


def set_policy(value):
    """Sets the kernel policy: `"auto"`, `"off"` or `"require"`.

    A route is selected at trace time. On JAX and TensorFlow, a policy
    change applies only to functions that are traced after the change.
    """
    _check_policy(value, None)
    _state.policy = value


@contextlib.contextmanager
def policy(value, route=None):
    """Runs a block, or decorates a function, under the policy `value`.

    `route` is for tests only: it forces that route where gates 2, 3, 4
    and 7 pass.
    """
    _check_policy(value, route)
    previous = (_state.policy, _state.route)
    _state.policy, _state.route = value, route
    try:
        yield
    finally:
        _state.policy, _state.route = previous


def force_route(name):
    """Test helper: runs the route `name` where gates 2, 3, 4 and 7 pass."""
    return policy("auto", route=name)


def _check_policy(value, route):
    if value not in POLICIES:
        raise ValueError(
            f"The kernel policy must be one of {POLICIES}. "
            f"Received: policy={value!r}"
        )
    if route is not None and route not in _ROUTES:
        raise ValueError(
            f"Unknown route. Expected one of {sorted(_ROUTES)}. "
            f"Received: route={route!r}"
        )
    if value == "require" and _required_route() is None:
        raise ValueError(
            "The kernel policy 'require' needs a platform with a kernel. "
            f"The {backend.backend()} backend has no kernel on this "
            "device."
        )


def _required_route():
    """The kernel that `"require"` runs on this platform, or `None`."""
    kernels = _REQUIRED.get(backend.backend())
    if not kernels:
        return None
    return kernels.get(distribution_lib.list_devices()[0].split(":")[0])


def select_route(layer, geometry, quantized_weight, inputs):
    """Returns the route that the static gates select, or `None`."""
    route = _select(layer, geometry, quantized_weight, inputs)
    # Under `torch.compile`, Dynamo would guard on the count and recompile
    # on every call.
    if not (backend.backend() == "torch" and torch.compiler.is_compiling()):
        route_counts["reference" if route is None else route.name] += 1
    return route


def _select(layer, geometry, view, inputs):
    """`select_route` without the count."""
    forced = _state.route
    # Gate 1, and gate 8 when the allow-list has no row: the reference path
    # runs before any other gate reads the call.
    if _state.policy == "off" or not (
        forced or _state.policy == "require" or _ALLOW_LIST
    ):
        return None
    contraction = _contraction(layer, geometry, view)  # Gates 3 and 4.
    if contraction is None:
        return None
    rows, contracted, columns = contraction
    free = tuple(inputs.shape)[: len(inputs.shape) - contracted]
    if any(dim is None for dim in free) or math.prod(free) < 1:  # Gate 2.
        return None
    m, dtype = math.prod(free), layer.compute_dtype
    # A forced route skips gates 5, 6 and 8, and `"require"` gate 8.
    if forced:
        candidates = ((forced, None),)
    elif _distributed(inputs, view):  # Gate 6.
        return None
    elif _state.policy == "require":
        candidates = ((_required_route(), None),)
    else:
        device = str(inputs.device) if backend.backend() == "torch" else None
        key = (backend.backend(), _device_class(device), dtype, _m_bucket(m))
        candidates = _ALLOW_LIST.get(key, ())
    for name, config in candidates:
        source = _ROUTES[name]
        if source.dtypes and dtype not in source.dtypes:  # Gate 7.
            continue
        if forced or _available(source, m, rows, math.prod(columns), view):
            return Route(name, source, config, rows, contracted, columns)
    return None


def _available(source, m, k, n, view):
    """Gate 5: the device and shape checks, and the tested version range."""
    if source.available is not None and not source.available(m, k, n, view):
        return False
    if _state.policy == "require" or source.version is None:
        return True
    return tested(source.version(), source.tested_versions)


def _contraction(layer, geometry, view):
    """Gates 3 and 4: `(rows, contracted, columns)`, or `None`."""
    scheme = view.scheme
    # `WeightScheme` gives a group size only to a multiplier scheme with a
    # zero point.
    grouped = scheme.group_size is not None and view.axis == 0
    per_channel = (
        scheme.scale_form == "divisor"
        and not scheme.has_zero_point
        and view.scale_axes == (1,)
        and not getattr(layer, "inputs_quantizer", None)
    )
    if not (
        isinstance(view.layout, Int4Pairs)
        and view.layout.axis in (-1, 1)
        and len(view.codes.shape) == 2
        and (grouped or per_channel)
    ):
        return None
    axes = getattr(layer, "einsum_axes", None)
    return _matrix(type(geometry), axes, tuple(geometry.weight_shape))


@functools.lru_cache(maxsize=256)
def _matrix(cls, axes, kernel_shape):
    """Gate 3 for a geometry class: `(rows, contracted, columns)`."""
    einsum = issubclass(cls, EinsumProjectionGeometry)
    base = EinsumProjectionGeometry if einsum else ProjectionGeometry
    if not issubclass(cls, ProjectionGeometry) or any(
        getattr(cls, name) is not getattr(base, name)
        for name in ("contract", "contract_grad", "kernel_axes")
    ):
        return None
    if not einsum:
        if len(kernel_shape) != 2:
            return None
        return kernel_shape[0], 1, kernel_shape[1:]
    contracted = reshape_compatible(axes)
    if contracted is None:
        return None
    sizes = dict(zip(axes.kernel, kernel_shape))
    rows = math.prod(sizes[x] for x in axes.kernel if x not in axes.output)
    columns = tuple(sizes[x] for x in axes.kernel if x in axes.output)
    return rows, contracted, columns


def reshape_compatible(axes):
    """How many trailing input axes a 2-D matmul contracts, or `None`.

    `axes` is the `EinsumAxes` of an equation. The stored rows hold the
    contracted kernel axes and the columns the free kernel axes, each in
    kernel order. The inputs reshaped to `(M, K)` line up with them when
    the equation has no batch axis, every input label is in the output or
    in the kernel, the contracted labels are the last input labels, in
    kernel order, and the output labels are the free input labels in
    input order, then the free kernel labels in kernel order.
    """
    inputs, kernel, output = axes.inputs, axes.kernel, axes.output
    contracted = "".join(label for label in kernel if label not in output)
    free = "".join(label for label in kernel if label in output)
    leading = inputs[: max(len(inputs) - len(contracted), 0)]
    if (
        set(free) & set(inputs)
        or not set(inputs) <= set(output + kernel)
        or inputs[len(leading) :] != contracted
        or output != leading + free
    ):
        return None
    return len(contracted)


def _m_bucket(m):
    return next((name for top, name in _M_BUCKETS if m <= top), "257+")


@functools.lru_cache(maxsize=None)
def _device_class(device):
    """`cuda-sm<major><minor>`, `tpu-<kind>`, `mps` or `cpu-<machine>`.

    `device` is the device of a torch operand. The other backends read
    their default device.
    """
    kind = "cpu"
    if backend.backend() == "torch":
        device = torch.device(device)
        if device.type == "cuda" and torch.version.cuda:
            return "cuda-sm%d%d" % torch.cuda.get_device_capability(device)
        kind = device.type
    elif backend.backend() == "jax":
        device = jax.devices()[0]
        capability = getattr(device, "compute_capability", "")
        if capability.replace(".", "").isdigit():
            return "cuda-sm" + capability.replace(".", "")
        kind = device.platform
        if kind == "tpu":
            kind = f"tpu-{device.device_kind}"
    elif backend.backend() == "tensorflow":
        if tf.config.list_logical_devices("GPU"):
            kind = "gpu"
    return f"cpu-{platform.machine()}" if kind == "cpu" else kind


def _distributed(inputs, view):
    """Gate 6: a Keras distribution is active, or an operand is a DTensor."""
    if distribution_lib.distribution() is not None:
        return True
    return backend.backend() == "torch" and any(
        isinstance(tensor, DTensor) for tensor in (inputs, view.codes)
    )


def tested(installed, prefixes):
    """Whether a library version is in a tested-version constant.

    `installed` is a version string, or the `(jax, jaxlib)` pair, whose
    two versions must be equal. `prefixes` holds release prefixes:
    `"3.8"` covers every 3.8.x release. A pre-release or a development
    release is never in the range.
    """
    versions = {installed} if isinstance(installed, str) else set(installed)
    if len(versions) != 1:
        return False
    version = Version(versions.pop())
    return not version.is_prerelease and any(
        version.release[: len(prefix.split("."))]
        == tuple(int(part) for part in prefix.split("."))
        for prefix in prefixes
    )


def run_route(route, layer, geometry, quantized_weight, inputs):
    """Runs `route` forward only."""
    source = route.source
    if not source.matrix:
        return source.fn(
            layer, geometry, quantized_weight, inputs, route.config
        )
    free = ops.shape(inputs)[: len(inputs.shape) - route.contracted]
    x = ops.reshape(inputs, (-1, route.rows))
    y = source.fn(layer, geometry, quantized_weight, x, route.config)
    return ops.reshape(y, tuple(free) + route.columns)


def contract_quantized(layer, geometry, quantized_weight, inputs):
    """Contracts `inputs` against the weight that `quantized_weight` holds.

    Runs the route that the static gates select, and the reference path
    (`dequantize`, then `geometry.contract`) when no route applies. It
    defines no gradient: its caller runs it inside a custom gradient.
    """
    route = select_route(layer, geometry, quantized_weight, inputs)
    if route is None:
        return _reference(layer, geometry, quantized_weight, inputs)
    return run_route(route, layer, geometry, quantized_weight, inputs)


def reverse_lookup_view(table, orientation):
    """The `(D, V)` view of a stored int4 lookup table.

    An untied table is stored `(D, V)`. A tied table is the forward table
    `(V, D)`, so the view transposes its codes, and its scale and zero
    point when they are grouped.
    """
    if orientation == "untied":
        return table
    grouped = table.zero_point is not None
    return QuantizedWeight(
        codes=ops.transpose(table.codes),
        scale=ops.transpose(table.scale) if grouped else table.scale,
        zero_point=ops.transpose(table.zero_point) if grouped else None,
        g_idx=table.g_idx,
        layout=Int4Pairs(axis=0, orig_len=table.layout.orig_len),
        scheme=table.scheme,
        shape=table.shape[::-1],
        axis=0 if grouped else None,
        scale_axes=None if grouped else (1,),
    )


def _reference_lookup(layer, table, orientation, inputs, config=None):
    """The reference reverse contraction, with no activation quantizer.

    A grouped table is dequantized in the precision of its scale. A
    per-channel table contracts its codes, then divides by its scale.
    """
    dtype = layer.reverse_dtype or layer.compute_dtype
    view = reverse_lookup_view(table, orientation)
    if view.zero_point is not None:
        precision = backend.result_type(dtype, view.scale.dtype)
        return ops.matmul(inputs, view.dequantize(precision))
    return view.apply_scale(ops.cast(ops.matmul(inputs, view.unpack()), dtype))


# Route name -> the `RouteSource` of its reverse lookup. Its `fn(layer,
# table, orientation, inputs, config)` takes the stored table. A `matrix`
# source takes the inputs as `(M, D)` and returns `(M, V)`.
_LOOKUP_ROUTES = {
    "reference_wrapped": RouteSource(_reference_lookup, matrix=False)
}
# (backend, device class, compute dtype, M bucket) -> ((route, config), ...)
_LOOKUP_ALLOW_LIST = {}


def select_lookup_route(layer, table, orientation, inputs):
    """Returns the route of a reverse lookup that the gates select, or `None`.

    The caller, the reverse lookup of `ReversibleEmbedding` with no
    activation quantizer, takes the place of gate 3. `table` is the
    stored table: `(V, D)` for a `"tied"` orientation, `(D, V)` for an
    `"untied"` one. `inputs` are `(..., D)`, in the reverse dtype. Gate 4
    asks for two 4-bit codes per byte along `D`; `LookupHandlers` stores
    a grouped scheme or a per-channel divisor. The routes are those of
    `_LOOKUP_ROUTES`, and gate 8 reads `_LOOKUP_ALLOW_LIST`.
    """
    route = _select_lookup(layer, table, orientation, inputs)
    if not (backend.backend() == "torch" and torch.compiler.is_compiling()):
        route_counts["reference" if route is None else route.name] += 1
    return route


def _select_lookup(layer, table, orientation, inputs):
    """`select_lookup_route` without the count."""
    forced = _state.route
    if _state.policy == "off" or not (  # Gates 1 and 8, as in `_select`.
        forced or _state.policy == "require" or _LOOKUP_ALLOW_LIST
    ):
        return None
    d_axis = 1 if orientation == "tied" else 0
    if not (  # Gate 4.
        isinstance(table.layout, Int4Pairs) and table.layout.axis % 2 == d_axis
    ):
        return None
    free = tuple(inputs.shape)[:-1]
    if any(dim is None for dim in free) or math.prod(free) < 1:  # Gate 2.
        return None
    m, dtype = math.prod(free), layer.reverse_dtype or layer.compute_dtype
    d, v = table.layout.orig_len, table.shape[1 - d_axis]
    if forced:
        candidates = ((forced, None),)
    # Gate 6, on the tensors that the stored variables hold.
    elif _distributed(inputs, table.with_tensors(table.read_tensors())):
        return None
    elif _state.policy == "require":
        candidates = ((_required_route(), None),)
    else:
        device = str(inputs.device) if backend.backend() == "torch" else None
        key = (backend.backend(), _device_class(device), dtype, _m_bucket(m))
        candidates = _LOOKUP_ALLOW_LIST.get(key, ())
    for name, config in candidates:
        source = _LOOKUP_ROUTES.get(name)
        if source is None or (source.dtypes and dtype not in source.dtypes):
            continue  # No reverse lookup in the route, or gate 7.
        if forced or _available(source, m, d, v, table):
            return Route(name, source, config, d, 1, (v,))
    return None


def run_lookup_route(route, layer, table, orientation, inputs):
    """Runs the reverse lookup `route` forward only."""
    source = route.source
    if not source.matrix:
        return source.fn(layer, table, orientation, inputs, route.config)
    x = ops.reshape(inputs, (-1, route.rows))
    y = source.fn(layer, table, orientation, x, route.config)
    return ops.reshape(y, tuple(ops.shape(inputs)[:-1]) + route.columns)
