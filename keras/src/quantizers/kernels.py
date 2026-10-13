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
5. The route is available: its device and shape checks pass, it accepts
   the configuration of the allow-list row for the call, and under
   `"auto"` its library version is in its tested range (`tested`).
6. No Keras distribution is active, and no operand is a DTensor.
7. The route supports the compute dtype. The inputs of a `matrix` route
   have the compute dtype: a layer with `autocast=False` leaves its
   inputs uncast.
8. Under `"auto"`, `_ALLOW_LIST` has a row for the backend, the device
   class, the compute dtype and the M bucket. A row holds measured
   routes only, as an ordered tuple of `(route, configuration)`. The
   selection takes the first route of the row whose gates pass.

N5 (`"n5"`) is the one route in backend-agnostic `keras.ops`. It
contracts the inputs with the exact small integers `q - z` of each
group, and the group scale multiplies the float32 partial sum. It reads
the group of row `k` as `k // G`. A grouped view with no `g_idx` (int4,
AWQ, GPTQ with one group) has these groups by construction, so it needs
no guard. For a view with a `g_idx` (GPTQ with several groups),
`run_route` runs N5 only when `g_idx` holds those groups, and the
reference path otherwise (GPTQ's activation order).

The policy is private. `"auto"` (the default) runs the route of the
allow-list row, `"off"` runs the reference path, and `"require"` runs
the platform's kernel. `"require"` ignores the allow-list and the
version range, and `set_policy("require")` raises on a platform without
a kernel. Under `"require"`, a call on a device that the kernel does
not support raises too.

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
from keras.src.backend.common import standardize_dtype
from keras.src.distribution import distribution_lib
from keras.src.quantizers.geometry import EinsumProjectionGeometry
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantized_weight import Int4Pairs
from keras.src.quantizers.quantized_weight import QuantizedWeight
from keras.src.utils.module_utils import tensorflow as tf

if backend.backend() == "torch":
    import torch
    from torch.distributed.tensor import DTensor

    from keras.src.backend.torch.ops.core import get_device
elif backend.backend() == "jax":
    import jax

POLICIES = ("auto", "off", "require")
# The largest M of each bucket; a larger M is in the bucket "257+".
_M_BUCKETS = ((16, "1-16"), (64, "17-64"), (256, "65-256"))
# (backend, device class, compute dtype, M bucket) -> ((route, config), ...)
# Each row names the JSON lines file of `quant-tooling/kernels/results/`
# that measured it.
_ALLOW_LIST = {
    # 2026-10-22-m1pro-torch-cpu-n5.jsonl (Apple M1 Pro, torch eager). Six
    # shapes per row; the bfloat16 rows above `M = 64` have three.
    ("torch", "cpu-arm64", "float32", "1-16"): (("n5", None),),
    ("torch", "cpu-arm64", "float16", "1-16"): (("n5", None),),
    ("torch", "cpu-arm64", "bfloat16", "1-16"): (("n5", None),),
    ("torch", "cpu-arm64", "bfloat16", "17-64"): (("n5", None),),
    ("torch", "cpu-arm64", "bfloat16", "65-256"): (("n5", None),),
    ("torch", "cpu-arm64", "bfloat16", "257+"): (("n5", None),),
}
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
    (gate 5). `accepts(m, quantized_weight, config)` refuses the
    configuration of an allow-list row for a call that the kernel cannot
    run with it (gate 5): one row serves every shape of its M bucket.
    `version()` returns the installed library version, or the
    `(jax, jaxlib)` pair, and `tested_versions` holds the release
    prefixes of the tested range. A source with `contiguous_groups`
    reads the group of row `k` as `k // G`, so `run_route` guards it on a
    view with a `g_idx`.
    """

    fn: object
    matrix: bool = True
    dtypes: tuple = ()
    available: object = None
    accepts: object = None
    version: object = None
    tested_versions: tuple = ()
    contiguous_groups: bool = False


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


def _wait_for_inputs(inputs, view):
    """Under a JAX trace, makes the work on the weight wait for `inputs`.

    The work on the weight alone (the unpacked and cast codes of N5) does
    not depend on the inputs. Without a barrier, XLA runs it for every
    layer of a program at the start and keeps each result live: on CPU,
    a chain of 8 int4 layers of 1024 x 1024 then reserves 32 MiB of temp
    memory in place of 4 MiB. One `jax.lax.optimization_barrier` over the
    inputs and the tensors of the weight orders the work after the
    inputs. Outside a JAX trace, the inputs and the view return as they
    are.
    """
    if backend.backend() != "jax" or not isinstance(inputs, jax.core.Tracer):
        return inputs, view
    inputs, tensors = jax.lax.optimization_barrier(
        (inputs, view.read_tensors())
    )
    return inputs, view.with_tensors(tensors)


def _sum_then_scale(layer, geometry, quantized_weight, inputs, config=None):
    """N5: the `(M, K)` inputs against the 4-bit codes, scaled per group.

    The low and high nibbles of the codes hold the even and the odd
    columns. For each plane, a batched float32 matmul contracts the
    inputs of each group, in the values of the compute dtype, with the
    small integers `q - z`, so each product is exact. The group scale
    multiplies the partial sum of its group. Per-channel int4 is one group
    with no zero point, and its scale divides the result. AWQ divides the
    inputs by its input scales in the compute dtype. Under a JAX trace,
    the work on the weight waits for the inputs (`_wait_for_inputs`).
    """
    inputs, view = _wait_for_inputs(inputs, quantized_weight)
    dtype = layer.compute_dtype
    rows, columns = view.codes.shape[0], view.layout.orig_len
    grouped = view.scheme.group_size is not None
    groups = rows // min(view.scheme.group_size, rows) if grouped else 1
    if standardize_dtype(view.codes.dtype) == "int8":
        # The arithmetic shifts sign-extend each nibble.
        low = ops.right_shift(ops.left_shift(view.codes, 4), 4)
    else:
        low = ops.bitwise_and(view.codes, 15)
    planes = (low, ops.right_shift(view.codes, 4))
    if view.input_scales is not None:
        inputs = ops.divide(inputs, view.input_scales)
    # `(groups, M, G)`: the inputs of each group.
    x = ops.reshape(inputs, (-1, groups, rows // groups))
    x = ops.cast(ops.transpose(x, (1, 0, 2)), "float32")
    scale = ops.cast(view.scale, "float32")
    if grouped:
        # `(groups, 1, N / 2, 2)`: the even and odd columns of each group.
        halves = (groups, 1, columns // 2, 2)
        zero = ops.cast(ops.reshape(view.zero_point, halves), "float32")
        group_scale = ops.reshape(scale, halves)
    outputs = []
    for i, plane in enumerate(planes):
        w = ops.reshape(plane, (groups, rows // groups, columns // 2))
        w = ops.cast(w, "float32")
        if grouped:
            # After the cast: a subtraction in uint8 would wrap.
            w = ops.subtract(w, zero[..., i])
            partial = ops.multiply(ops.matmul(x, w), group_scale[..., i])
        else:
            partial = ops.matmul(x, w)
        outputs.append(ops.sum(partial, axis=0))
    y = ops.reshape(ops.stack(outputs, axis=-1), (-1, columns))
    if not grouped:
        y = ops.divide(y, scale)
    return ops.cast(y, dtype)


def _n5_available(m, k, n, view):
    """Gate 5 of N5: whole groups or one group, an even `N`, and eager
    execution on torch and TensorFlow."""
    group = view.scheme.group_size
    if n % 2 or (group is not None and k % group and k > group):
        return False
    if backend.backend() == "torch":
        return not torch.compiler.is_compiling()
    return backend.backend() != "tensorflow" or tf.executing_eagerly()


# Route name -> `RouteSource`. "reference_wrapped" runs the reference path
# as a route, through `force_route` only.
_ROUTES = {
    "reference_wrapped": RouteSource(_reference, matrix=False),
    "n5": RouteSource(
        _sum_then_scale,
        dtypes=("bfloat16", "float16", "float32"),
        available=_n5_available,
        contiguous_groups=True,
    ),
}


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

    `route` is for tests only: it forces that route where gates 2, 3, 4,
    5 (without the version range) and 7 pass.
    """
    _check_policy(value, route)
    previous = (_state.policy, _state.route)
    _state.policy, _state.route = value, route
    try:
        yield
    finally:
        _state.policy, _state.route = previous


def force_route(name):
    """Test helper: runs the route `name` where gates 2, 3, 4, 5 (without
    the version range) and 7 pass."""
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
    # A forced route skips gates 6 and 8, and `"require"` gate 8. Both
    # skip the version range of gate 5.
    if forced:
        candidates = ((forced, None),)
    elif _distributed(inputs, view):  # Gate 6.
        return None
    elif _state.policy == "require":
        candidates = ((_required_route(), None),)
    else:
        candidates = _candidates(_ALLOW_LIST, dtype, m)
    for name, config in candidates:
        source = _ROUTES[name]
        if (source.dtypes and dtype not in source.dtypes) or (  # Gate 7.
            source.matrix and standardize_dtype(inputs.dtype) != dtype
        ):
            continue
        if _available(source, m, rows, math.prod(columns), view, config):
            return Route(name, source, config, rows, contracted, columns)
    return None


def _available(source, m, k, n, view, config):
    """Gate 5: the device and shape checks, the configuration of the
    allow-list row (`None` without one), and the tested version range."""
    if source.available is not None and not source.available(m, k, n, view):
        return False
    if config is not None and source.accepts is not None:
        if not source.accepts(m, view, config):
            return False
    if _state.route or _state.policy == "require" or source.version is None:
        return True
    return tested(source.version(), source.tested_versions)


def _candidates(allow_list, dtype, m):
    """Gate 8: the `(route, config)` pairs of the row of this call.

    The M bucket comes from the input shape, so it is read only where the
    platform has a row. Elsewhere, a compiled call does not guard on it
    and does not compile again for a new bucket.
    """
    # torch ops move every operand to the device of `get_device()`.
    device = get_device() if backend.backend() == "torch" else None
    platform = (backend.backend(), _device_class(device), dtype)
    if not any(key[:3] == platform for key in allow_list):
        return ()
    return allow_list.get(platform + (_m_bucket(m),), ())


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

    `device` is the torch device that the call runs on. The other
    backends read their default device.
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
    if backend.backend() != "torch":
        return False
    # The view of a GPTQ or AWQ call, or of a lookup table, holds the
    # layer's variables.
    codes = getattr(view.codes, "value", view.codes)
    return any(isinstance(tensor, DTensor) for tensor in (inputs, codes))


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
    """Runs `route` forward only, with the group guard when it needs one.

    A route with `contiguous_groups` runs on a view with a `g_idx` (GPTQ
    with several groups) only when `g_idx[k] == k // G` for every row
    `k`, and the reference path runs otherwise. Under a JAX trace,
    `ops.cond` traces both. An eager call branches in Python: an eager
    `lax.cond` would trace and compile both branches again on each call,
    because each call gives it new functions. A grouped view with no
    `g_idx` (int4, AWQ, GPTQ with one group) puts row `k` in group
    `k // G` on every path, so it needs no guard.
    """
    source = route.source
    if not source.matrix:
        return source.fn(
            layer, geometry, quantized_weight, inputs, route.config
        )

    def routed():
        free = ops.shape(inputs)[: len(inputs.shape) - route.contracted]
        x = ops.reshape(inputs, (-1, route.rows))
        y = source.fn(layer, geometry, quantized_weight, x, route.config)
        return ops.reshape(y, tuple(free) + route.columns)

    def reference():
        return _reference(layer, geometry, quantized_weight, inputs)

    # Whether the groups are contiguous is a fact of the weight, decided
    # where `g_idx` is written or loaded and never per call. Serving stacks
    # decide it the same way: vLLM (`marlin_utils.py`, an empty `g_idx`
    # without activation order), SGLang (`gptq.py`), TGI
    # (`layers/gptq/__init__.py`, `arange // G` without `desc_act`) and
    # TensorRT-LLM (which loads no `g_idx`).
    g_idx = quantized_weight.g_idx
    if not source.contiguous_groups or g_idx is None:
        return routed()
    group = min(quantized_weight.scheme.group_size, route.rows)
    groups = ops.floor_divide(ops.arange(route.rows), group)
    contiguous = ops.all(ops.equal(ops.cast(g_idx, "int32"), groups))
    if backend.backend() == "jax" and not isinstance(
        contiguous, jax.core.Tracer
    ):
        return routed() if contiguous else reference()
    # Under a JAX trace this is a `lax.cond`; torch and TensorFlow eager
    # branch in Python.
    return ops.cond(contiguous, routed, reference)


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
    elif _distributed(inputs, table):  # Gate 6.
        return None
    elif _state.policy == "require":
        candidates = ((_required_route(), None),)
    else:
        candidates = _candidates(_LOOKUP_ALLOW_LIST, dtype, m)
    for name, config in candidates:
        source = _LOOKUP_ROUTES.get(name)
        if source is None or (source.dtypes and dtype not in source.dtypes):
            continue  # No reverse lookup in the route, or gate 7.
        if _available(source, m, d, v, table, config):
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


# Triton K1 (torch with CUDA): `backend.ops.quantized.w4a16_matmul`, a
# `matrix` route. Its K-packed variant `w4a16_logits` is the reverse lookup
# route of the same name. They are in `_ROUTES` and `_LOOKUP_ROUTES` only
# where Triton imports, and they have no allow-list row until the GPU
# sessions measure one. They are the kernels of `"require"` on a CUDA
# device (device type "gpu" in `list_devices`), but not on ROCm, where
# `pytorch-triton-rocm` imports as `triton`.
# The Triton releases that a run under `"require"` passed on (GPU CI or a
# pod).
TRITON_TESTED = ("3.8",)


def _triton(layer, geometry, quantized_weight, inputs, config=None):
    """Triton K1: the `(M, K)` inputs with the stored 4-bit codes."""
    view = quantized_weight
    return backend.ops.quantized.w4a16_matmul(
        inputs,
        view.codes,
        view.scale,
        view.zero_point,
        view.g_idx,
        group_size=view.scheme.group_size,
        signed=ops.dtype(view.codes) == "int8",
        per_channel=view.scheme.group_size is None,
        config=config,
        awq_scales=view.input_scales,
    )


def _triton_available(m, k, n, view):
    """Gate 5 of Triton K1: the device gate and the int32 offsets.

    The device is CUDA (not ROCm) with compute capability 8.0 or newer,
    or 7.5 with float16 compute. Under `"require"`, a CUDA device that
    fails this check raises. On CPU, the gate passes only under the
    Triton interpreter, for `force_route` in the tests: no CPU row exists.
    """
    # The kernel indexes with int32 offsets. The split-K workspace, which
    # can hold more elements, has int64 offsets.
    if max(m * k, m * n, k * n) >= 2**31:
        return False
    # The view of a GPTQ or AWQ call holds the layer's variables.
    device = getattr(view.codes, "value", view.codes).device
    if device.type == "cpu":
        return backend.ops.quantized.INTERPRETED
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    capability = backend.ops.quantized.device_capability(device)
    # Inside the autocast scope, the scale reads in the compute dtype.
    dtype = ops.dtype(view.scale)
    if capability >= (8, 0) or (capability == (7, 5) and dtype == "float16"):
        return True
    if _state.policy == "require":
        raise ValueError(
            "The kernel policy 'require' needs a CUDA device with compute "
            "capability 8.0 or newer, or 7.5 with float16 compute. "
            f"Received: capability={capability}, compute dtype={dtype}"
        )
    return False


def _triton_logits(layer, table, orientation, inputs, config=None):
    """Triton K-packed: the `(M, D)` inputs with the stored 4-bit table."""
    return backend.ops.quantized.w4a16_logits(
        inputs,
        table.codes,
        table.scale,
        table.zero_point,
        group_size=table.scheme.group_size,
        per_channel=table.scheme.group_size is None,
        orientation=orientation,
        config=config,
    )


def _triton_logits_available(m, d, v, table):
    """Gate 5 of the Triton logits: the alignment, K tiles in one group,
    and the gate of K1 on compute capability 8.0 or newer.

    `D` and `V` must be multiples of 128. A K tile is a power of two from
    32 to 128 rows of `D`, and it must lie in one group. So a grouped
    table has groups of 128 or 256 rows, or one group (`G >= D`). On
    compute capability 7.5, the gate refuses the call, also under
    `"require"`, and the call takes the reference path. The gate of K1
    reads the dtype of the scale there, and the scale of a table reads
    float32 in a model, because an `Embedding` does not autocast.
    """
    group = table.scheme.group_size
    if d % 128 or v % 128 or (group not in (None, 128, 256) and group < d):
        return False
    device = getattr(table.codes, "value", table.codes).device
    if device.type == "cuda":
        if backend.ops.quantized.device_capability(device) == (7, 5):
            return False
    return _triton_available(m, d, v, table)


if backend.backend() == "torch" and backend.ops.quantized.triton is not None:
    _ROUTES["triton"] = RouteSource(
        _triton,
        dtypes=("bfloat16", "float16", "float32"),
        available=_triton_available,
        version=lambda: backend.ops.quantized.triton.__version__,
        tested_versions=TRITON_TESTED,
    )
    _LOOKUP_ROUTES["triton"] = RouteSource(
        _triton_logits,
        dtypes=("bfloat16", "float16", "float32"),
        available=_triton_logits_available,
        version=lambda: backend.ops.quantized.triton.__version__,
        tested_versions=TRITON_TESTED,
    )
    if torch.version.hip is None:
        _REQUIRED.setdefault("torch", {})["gpu"] = "triton"


# Mosaic GPU K1 (JAX, NVIDIA): `mosaic_gpu_w4a16_matmul` of the JAX
# backend's `ops.quantized`. Under `"auto"`, it runs only with equal jax
# and jaxlib versions in `MOSAIC_GPU_TESTED_JAX`: the Mosaic GPU compiler
# ships in jaxlib, and a jax release accepts an older jaxlib. A version
# enters the constant after a correctness run on a GPU passes on it.
MOSAIC_GPU_TESTED_JAX = ("0.11.2", "0.12.0")
# The device gate: CUDA compute capabilities. A ROCm device reports a
# `gfx` name.
_MOSAIC_GPU_CAPABILITIES = ("8.0", "8.6", "8.9", "9.0")


def _mosaic_gpu(layer, geometry, view, inputs, config, interpret=False):
    """Mosaic GPU K1: the `(M, K)` inputs against the 4-bit codes.

    With no configuration in the allow-list row, the row of
    `MOSAIC_GPU_CONFIGS` for the device gives it. Interpret mode runs on
    CPU and takes the rows of compute capability 8.9.
    """
    quantized = backend.ops.quantized
    m, k = inputs.shape
    if config is None:
        if interpret:
            capability = "8.9"
        else:
            capability = jax.devices()[0].compute_capability
        key = (capability, layer.compute_dtype, _m_bucket(m))
        config = quantized.MOSAIC_GPU_CONFIGS[key]
    grouped = view.scheme.group_size is not None
    return quantized.mosaic_gpu_w4a16_matmul(
        inputs,
        view.codes,
        view.scale,
        view.zero_point,
        group_size=min(view.scheme.group_size, k) if grouped else None,
        signed=standardize_dtype(view.codes.dtype) == "int8",
        per_channel=not grouped,
        config=config,
        input_scales=view.input_scales,
        interpret=interpret,
    )


def _mosaic_gpu_aligned(k, n, group_size):
    """The alignment gate: whole K steps of 128 rows, whole blocks of 32
    code columns, and groups of whole K steps. A per-channel weight has
    no `group_size`."""
    group = k if group_size is None else min(group_size, k)
    return n % 64 == 0 and k % 128 == 0 and group % 128 == 0


def _mosaic_gpu_available(m, k, n, view):
    """Gate 5 of Mosaic GPU K1: the device gate and the alignment gate.

    Under `"require"`, a device or a JAX that cannot run the kernel raises.
    """
    if backend.backend() != "jax":
        return False
    device = jax.devices()[0]
    capability = getattr(device, "compute_capability", None)
    if not (
        device.platform == "gpu"
        and capability in _MOSAIC_GPU_CAPABILITIES
        and backend.ops.quantized.plgpu.available
    ):
        if _state.policy == "require":
            raise ValueError(
                "The kernel policy 'require' runs Mosaic GPU K1, which needs "
                "an NVIDIA GPU of compute capability "
                f"{', '.join(_MOSAIC_GPU_CAPABILITIES)} and a JAX with "
                f"`plgpu.mma` (0.11.2 or newer). Received: device={device}, "
                f"compute_capability={capability}, jax={jax.__version__}"
            )
        return False
    return _mosaic_gpu_aligned(k, n, view.scheme.group_size)


def _mosaic_gpu_interpret_available(m, k, n, view):
    """The interpret route: a JAX with Mosaic GPU, and the alignment gate."""
    return (
        backend.backend() == "jax"
        and backend.ops.quantized.plgpu.available
        and _mosaic_gpu_aligned(k, n, view.scheme.group_size)
    )


_ROUTES["mosaic_gpu"] = RouteSource(
    _mosaic_gpu,
    dtypes=("bfloat16", "float16"),
    available=_mosaic_gpu_available,
    version=lambda: backend.ops.quantized.mosaic_gpu_versions(),
    tested_versions=MOSAIC_GPU_TESTED_JAX,
    contiguous_groups=True,
)
# Test helper, through `force_route` only: Mosaic GPU K1 in GPU interpret
# mode on CPU, with `jnp.dot` in place of `plgpu.mma`. It does not support
# `jax.vmap` of a guarded call (a view with a `g_idx`): interpret mode has
# an IO effect, which JAX does not support in a vmap-of-cond.
_ROUTES["mosaic_gpu_interpret"] = RouteSource(
    functools.partial(_mosaic_gpu, interpret=True),
    dtypes=("bfloat16", "float16"),
    available=_mosaic_gpu_interpret_available,
    contiguous_groups=True,
)
_REQUIRED.setdefault("jax", {})["gpu"] = "mosaic_gpu"


@contextlib.contextmanager
def mosaic_gpu_lowering(compute_capability):
    """Test helper: lowers the Mosaic GPU kernels for a GPU on a CPU host.

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
        if not backend.ops.quantized.plgpu.available:
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

    quantized = backend.ops.quantized
    traced = (
        quantized.mosaic_gpu_w4a16_matmul,
        quantized.mosaic_gpu_w4a16_logits,
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


# Mosaic GPU K-packed (JAX, NVIDIA): `mosaic_gpu_w4a16_logits` of the JAX
# backend's `ops.quantized`, the reverse lookup routes `"mosaic_gpu"` and
# `"mosaic_gpu_interpret"`. They have the device gate, the version range
# and the dtypes of Mosaic GPU K1. The groups of a table are `d // G` by
# construction, so they need no guard.
def _mosaic_gpu_logits(
    layer, table, orientation, inputs, config, interpret=False
):
    """Mosaic GPU K-packed: the `(M, D)` inputs against the stored table.

    With no configuration in the allow-list row, the row of
    `MOSAIC_GPU_LOGITS_CONFIGS` for the device gives it. Interpret mode
    runs on CPU and takes the rows of compute capability 8.9.
    """
    quantized = backend.ops.quantized
    m, d = inputs.shape
    if config is None:
        if interpret:
            capability = "8.9"
        else:
            capability = jax.devices()[0].compute_capability
        dtype = layer.reverse_dtype or layer.compute_dtype
        key = (capability, dtype, _m_bucket(m))
        config = quantized.MOSAIC_GPU_LOGITS_CONFIGS[key]
    grouped = table.zero_point is not None
    return quantized.mosaic_gpu_w4a16_logits(
        inputs,
        table.codes,
        table.scale,
        table.zero_point,
        group_size=min(table.scheme.group_size, d) if grouped else None,
        per_channel=not grouped,
        orientation=orientation,
        config=config,
        interpret=interpret,
    )


def _mosaic_gpu_logits_available(m, d, v, table, interpret=False):
    """Gate 5 of Mosaic GPU K-packed: blocks of 128 columns of `V`, then
    gate 5 of K1 for `K = D` and `N = V`.

    The shape check comes before the device gate, as for the Triton
    logits. So under `"require"`, a `V` that 128 does not divide takes the
    reference path on each device.
    """
    if v % 128:
        return False
    if interpret:
        return _mosaic_gpu_interpret_available(m, d, v, table)
    return _mosaic_gpu_available(m, d, v, table)


def _mosaic_gpu_logits_accepts(m, table, config):
    """The configuration check of Mosaic GPU K-packed: a tied table with
    more than 16 rows needs `BM` 32 or 64.

    On JAX 0.12.0, a block of 16 columns of the transposed activations
    `(D, Mp)` fails the GMEM transfer plan of Mosaic GPU when `Mp > 16`.
    One allow-list row serves both orientations. So the check refuses the
    tied call, which takes the reference path, and keeps the untied one.
    A tied table packs its codes along axis 1.
    """
    tied = table.layout.axis % 2 == 1
    return not (tied and config[0] == 16 and m > 16)


_LOOKUP_ROUTES["mosaic_gpu"] = RouteSource(
    _mosaic_gpu_logits,
    dtypes=("bfloat16", "float16"),
    accepts=_mosaic_gpu_logits_accepts,
    available=_mosaic_gpu_logits_available,
    version=lambda: backend.ops.quantized.mosaic_gpu_versions(),
    tested_versions=MOSAIC_GPU_TESTED_JAX,
)
# Test helper, through `force_route` only: Mosaic GPU K-packed in GPU
# interpret mode on CPU.
_LOOKUP_ROUTES["mosaic_gpu_interpret"] = RouteSource(
    functools.partial(_mosaic_gpu_logits, interpret=True),
    dtypes=("bfloat16", "float16"),
    accepts=_mosaic_gpu_logits_accepts,
    available=functools.partial(_mosaic_gpu_logits_available, interpret=True),
)


# Pallas TPU K1 (JAX, TPU): `pallas_tpu_w4a16_matmul` of the JAX backend's
# `ops.quantized`. Under `"auto"`, it runs only with equal jax and jaxlib
# versions in `PALLAS_TPU_TESTED_JAX`: the Pallas TPU lowering ships in
# jaxlib. A version enters the constant after a correctness run on a TPU
# passes on it. No run has passed yet, so the constant is empty, the route
# has no allow-list row, and `"auto"` does not run it.
PALLAS_TPU_TESTED_JAX = ()


def _pallas_tpu(layer, geometry, view, inputs, config, interpret=False):
    """Pallas TPU K1: the `(M, K)` inputs against the 4-bit codes.

    With no configuration in the allow-list row, `pallas_tpu_blocks` gives
    the blocks of the shape. Interpret mode runs the kernel inside
    `pallas_tpu_interpret`.
    """
    quantized = backend.ops.quantized
    m, k = inputs.shape
    grouped = view.scheme.group_size is not None
    group_size = min(view.scheme.group_size, k) if grouped else None
    if config is None:
        n = view.layout.orig_len
        config = quantized.pallas_tpu_blocks(m, k, n, group_size)
    call = functools.partial(
        quantized.pallas_tpu_w4a16_matmul,
        inputs,
        view.codes,
        view.scale,
        view.zero_point,
        group_size=group_size,
        signed=standardize_dtype(view.codes.dtype) == "int8",
        per_channel=not grouped,
        config=config,
        input_scales=view.input_scales,
    )
    if not interpret:
        return call()
    with pallas_tpu_interpret():
        return call(interpret=quantized.pltpu.InterpretParams())


def _pallas_tpu_aligned(m, k, n, group_size):
    """The alignment gate: an even `N`, and a `BK` that divides `K` and
    that the group size divides. A per-channel weight has no
    `group_size`."""
    group = None if group_size is None else min(group_size, k)
    blocks = backend.ops.quantized.pallas_tpu_blocks(m, k, n, group)
    return blocks is not None


def _pallas_tpu_available(m, k, n, view):
    """Gate 5 of Pallas TPU K1: the device gate and the alignment gate.

    Under `"require"`, a device that is not a TPU raises.
    """
    if backend.backend() != "jax":
        return False
    device = jax.devices()[0]
    if device.platform != "tpu":
        if _state.policy == "require":
            raise ValueError(
                "The kernel policy 'require' runs Pallas TPU K1, which needs "
                f"a TPU. Received: device={device}"
            )
        return False
    return _pallas_tpu_aligned(m, k, n, view.scheme.group_size)


def _pallas_tpu_interpret_available(m, k, n, view):
    """The interpret route: the JAX backend and the alignment gate."""
    return backend.backend() == "jax" and _pallas_tpu_aligned(
        m, k, n, view.scheme.group_size
    )


# The route does not support `jax.vmap`: the batching rule of `pallas_call`
# gives each operand in `pl.ANY` a batched block, which the Pallas TPU
# lowering refuses.
_ROUTES["pallas_tpu"] = RouteSource(
    _pallas_tpu,
    dtypes=("bfloat16", "float32"),
    available=_pallas_tpu_available,
    # The `(jax, jaxlib)` pair.
    version=lambda: backend.ops.quantized.mosaic_gpu_versions(),
    tested_versions=PALLAS_TPU_TESTED_JAX,
    contiguous_groups=True,
)
# Test helper, through `force_route` only: Pallas TPU K1 in TPU interpret
# mode on a host without a TPU. It does not support `jax.vmap` either.
_ROUTES["pallas_interpret"] = RouteSource(
    functools.partial(_pallas_tpu, interpret=True),
    dtypes=("bfloat16", "float32"),
    available=_pallas_tpu_interpret_available,
    contiguous_groups=True,
)
_REQUIRED.setdefault("jax", {})["tpu"] = "pallas_tpu"


@contextlib.contextmanager
def pallas_tpu_lowering(device_kind):
    """Test helper: lowers the Pallas TPU kernels for a TPU on a CPU host.

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
    """Test helper: the TPU of Pallas TPU interpret mode on a host without
    a TPU.

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
