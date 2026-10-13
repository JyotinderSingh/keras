"""Test routes and fixtures of `kernels.py`.

Importing this module registers the test route `"reference_wrapped"`:
the reference path of a projection, run as a route through `force_route`
only. A wrapped call takes the path of a selected route (the selection
and the count), so its outputs must equal those of the reference path
bit for bit.

`route_tables` adds the `"matrix"` route, which contracts the `(M, K)`
inputs with the stored matrix.

The route tests check, by label:

- G1: the error against the float64 oracle is at most 1.25 times that
  of the reference path plus a floor of the compute dtype.
- G2: the error is at most an absolute bound of the compute dtype.
- G3: small integer codes and power-of-two scales give the oracle
  exactly.
- G4: the outputs are finite.
- G5: two calls give the same outputs.
- G6: the reference path (`"off"`) is unchanged, bit for bit.
- G7: the input gradient equals that of `"reference_wrapped"`.
- G8: the layer output equals a direct call of the selected route.
"""

import contextlib
import math

from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src.backend.common import AutocastScope
from keras.src.quantizers import kernels


def _matrix(quantized_weight, inputs, dtype):
    """`(M, K)` inputs times the stored matrix."""
    kernel = quantized_weight.dequantize(dtype)
    if quantized_weight.permutation is not None:
        kernel = ops.transpose(kernel, quantized_weight.permutation)
    return ops.matmul(inputs, ops.reshape(kernel, (inputs.shape[-1], -1)))


def matrix_route(**changes):
    """The `"matrix"` route of a projection, with the fields `changes`."""
    return kernels.Route(_matrix, **changes)


kernels._ROUTES["reference_wrapped"] = kernels.Route(
    kernels._reference, matrix=False
)


@contextlib.contextmanager
def route_tables(rows=None, routes=None, required=None):
    """Runs a block with test route tables, under `"auto"`.

    The block adds the `"matrix"` route, then `routes`, to the route
    table, and sets the measured rows to `rows` (none by default). With
    `required`, a tuple of route names, the other kernel routes leave the
    table, so `"require"` runs only those.

    Only `"auto"` reads the measured rows, and `pytest --kernel-policy`
    can set another policy for the process. So the block runs under
    `"auto"`. A block inside this block can set a different policy. The
    tables and the policy are as before when the block ends.
    """
    tables = (kernels._ROUTES, kernels._MEASURED_ROWS)
    saved = [dict(table) for table in tables]
    if required is not None:
        for name, route in list(kernels._ROUTES.items()):
            if route.kernel is not None and name not in required:
                del kernels._ROUTES[name]
    kernels._ROUTES.update({"matrix": matrix_route(), **(routes or {})})
    kernels._MEASURED_ROWS.clear()
    kernels._MEASURED_ROWS.update(rows or {})
    try:
        with kernels.policy("auto"):
            yield
    finally:
        for table, values in zip(tables, saved):
            table.clear()
            table.update(values)


def row_key(dtype, bucket):
    """The measured row key of this backend and device."""
    return (backend.backend(), kernels._device_class(), dtype, bucket)


def counts(layer, x):
    """The routes that one call `layer(x)` selects."""
    kernels.route_counts.clear()
    layer(x)
    return dict(kernels.route_counts)


def device_type():
    """The device type of `list_devices`: `"cpu"`, `"gpu"`, `"tpu"`..."""
    if backend.backend() == "numpy":
        return "cpu"
    return distribution.list_devices()[0].split(":")[0]


def floor_and_bound(dtype):
    """The G1 floor and the G2 bound of the compute dtype `dtype`."""
    if dtype == "float32":
        return 1e-6, 1e-5
    if dtype == "float16":
        return 2e-4, 2**-9
    return 1e-3, 2**-7


def admitted(layer, x):
    """The routes that `force_route` runs for a call of `layer` on `x`.

    They are the routes of the table whose dtypes, device check and shape
    check pass.
    """
    routes, view = kernels._ROUTES, layer._quantized_weight()
    dtype = layer.compute_dtype
    geometry = layer._quantization_geometry()
    k, contracted, columns = geometry.matrix_contraction()
    n = math.prod(columns)
    shape = tuple(x.shape)
    m = math.prod(shape[: len(shape) - contracted])
    device_class = kernels._device_class()
    names = []
    for name, route in routes.items():
        if route.dtypes and dtype not in route.dtypes:
            continue
        if route.available is None or route.available(
            device_class, m, k, n, view, dtype
        ):
            names.append(name)
    return names


def direct(layer, x):
    """A direct call of the route that a call of `layer` on `x` selects,
    on the tensors that the layer reads: `kernels.run_route`."""
    geometry = layer._quantization_geometry()
    with AutocastScope(layer.compute_dtype):
        view = layer._quantized_weight()
        view = view.with_tensors(view.read_tensors())
        inputs = ops.cast(x, layer.compute_dtype)
        selection = kernels.select_route(layer, geometry, view, inputs)
        return kernels.run_route(selection, layer, geometry, view, inputs)
