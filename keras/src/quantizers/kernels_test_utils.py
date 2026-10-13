"""Test routes and fixtures of `kernels.py`.

Importing this module registers test routes in both route tables:

- `"reference_wrapped"`: the reference path of a projection and of a
  reverse lookup, run as a route through `force_route` only. A wrapped
  call takes the path of a selected route (the selection, the count and,
  for the reverse lookup, the custom gradient), so its outputs must equal
  those of the reference path bit for bit.
- The interpret routes of the backend: `"mosaic_gpu_interpret"` on JAX
  and `"triton_interpret"` on torch under the Triton interpreter. They
  run the kernels of the backend on a CPU host and have no `kernel`, so
  `"require"` does not run them.

`route_tables` adds the `"matrix"` routes, which contract the `(M, K)`
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
from keras.src.backend.common.backend_utils import m_bucket
from keras.src.quantizers import kernels
from keras.src.quantizers import strategy_registry

if backend.backend() == "jax":
    from keras.src.backend.jax.ops import quantized_test_utils
elif backend.backend() == "torch":
    from keras.src.backend.torch.ops import quantized_test_utils
else:
    quantized_test_utils = None

# Whether the kernels run under the Triton interpreter, which rounds
# float32 to bfloat16 toward zero: up to one bfloat16 ulp of the output.
INTERPRETED = (
    backend.backend() == "torch" and quantized_test_utils.quantized.INTERPRETED
)


def _reference_logits(layer, geometry, table, inputs):
    """The reference reverse contraction, with no activation quantizer.

    A grouped table is dequantized in the precision of its scale. A
    per-channel table contracts its codes, then divides by its scale.
    """
    dtype = layer.reverse_dtype or layer.compute_dtype
    view = table
    if layer.tie_weights:
        strategy = strategy_registry.get_strategy(layer.quantization_mode)
        view = strategy._transposed_lookup_view(layer, table)
    if view.zero_point is not None:
        precision = backend.result_type(dtype, view.scale.dtype)
        return ops.matmul(inputs, view.dequantize(precision))
    return view.apply_scale(ops.cast(ops.matmul(inputs, view.unpack()), dtype))


def _matrix(quantized_weight, inputs, dtype):
    """`(M, K)` inputs times the stored matrix."""
    kernel = quantized_weight.dequantize(dtype)
    if quantized_weight.permutation is not None:
        kernel = ops.transpose(kernel, quantized_weight.permutation)
    return ops.matmul(inputs, ops.reshape(kernel, (inputs.shape[-1], -1)))


def _matrix_logits(table, inputs, dtype):
    """The `(M, D)` inputs times the table as stored."""
    weight = table.dequantize(dtype)
    if kernels._tied(table):
        weight = ops.transpose(weight)
    return ops.matmul(inputs, weight)


def matrix_route(**changes):
    """The `"matrix"` route of a projection, with the fields `changes`."""
    return kernels.Route(_matrix, **changes)


def matrix_logits_route(**changes):
    """The `"matrix"` route of a reverse lookup, with the fields
    `changes`."""
    return kernels.Route(_matrix_logits, **changes)


kernels._ROUTES["reference_wrapped"] = kernels.Route(
    kernels._reference, matrix=False
)
kernels._LOGITS_ROUTES["reference_wrapped"] = kernels.Route(
    _reference_logits, matrix=False
)
# Interpret route name -> the name of the kernel route that it stands in
# for.
INTERPRET_ROUTES = {}
if quantized_test_utils is not None:
    INTERPRET_ROUTES = quantized_test_utils.register_interpret_routes()


@contextlib.contextmanager
def route_tables(
    rows=None, logits_rows=None, routes=None, logits_routes=None, required=None
):
    """Runs a block with test route tables, under `"auto"`.

    The block adds the `"matrix"` routes, then `routes` and
    `logits_routes`, to the route tables, and sets the measured rows to
    `rows` and `logits_rows` (none by default). With `required`, a tuple
    of route names, the other kernel routes leave both tables, so
    `"require"` runs only those.

    Only `"auto"` reads the measured rows, and `pytest --kernel-policy`
    can set another policy for the process. So the block runs under
    `"auto"`. A block inside this block can set a different policy. The
    tables and the policy are as before when the block ends.
    """
    tables = (
        kernels._ROUTES,
        kernels._LOGITS_ROUTES,
        kernels._MEASURED_ROWS,
        kernels._MEASURED_LOGITS_ROWS,
    )
    saved = [dict(table) for table in tables]
    if required is not None:
        for table in tables[:2]:
            for name, route in list(table.items()):
                if route.kernel is not None and name not in required:
                    del table[name]
    kernels._ROUTES.update({"matrix": matrix_route(), **(routes or {})})
    kernels._LOGITS_ROUTES.update(
        {"matrix": matrix_logits_route(), **(logits_routes or {})}
    )
    for table, measured_rows in zip(tables[2:], (rows, logits_rows)):
        table.clear()
        table.update(measured_rows or {})
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


def row(dtype, m, route):
    """A measured row of this backend and device that runs `route` for a
    call with `m` rows."""
    return {row_key(dtype, m_bucket(m)): (route,)}


# An `M` in each M bucket.
BUCKET_ROWS = {"1-16": 4, "17-64": 33, "65-256": 130, "257+": 300}


def first_tested_route(routes, table):
    """The first route of a measured row that `table` holds, with its
    library version in its tested range, else `"reference"`. On the
    device class of the row, this is the route that `"auto"` selects for
    a call that passes the shape gates."""
    for name in routes:
        source = table.get(name)
        if source is None:
            continue
        if source.tested is None or source.tested():
            return name
    return "reference"


def counts(layer, x, **call_kwargs):
    """The routes that one call `layer(x, **call_kwargs)` selects, for
    example with `reverse=True`."""
    kernels.route_counts.clear()
    layer(x, **call_kwargs)
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
    return 2**-7 if INTERPRETED else 1e-3, 2**-7


def lookup_table(layer):
    """The stored table of the reverse lookup: the forward table `(V, D)`
    of a tied layer, or the reverse table `(D, V)` of an untied one."""
    strategy = strategy_registry.get_strategy(layer.quantization_mode)
    if layer.tie_weights:
        return strategy._get_lookup_quantized_weight(layer, None)
    return strategy._get_reverse_lookup_quantized_weight(layer, None)


def admitted(layer, x, reverse=False):
    """The routes that `force_route` runs for a call of `layer` on `x`.

    They are the routes of the table whose dtypes, device check and shape
    check pass. An interpret route stands in for its kernel only on a
    device that the kernel does not support.
    """
    if reverse:
        routes, view = kernels._LOGITS_ROUTES, lookup_table(layer)
        dtype = layer.reverse_dtype or layer.compute_dtype
        k, contracted, n = layer.output_dim, 1, layer.input_dim
    else:
        routes, view = kernels._ROUTES, layer._quantized_weight()
        dtype = layer.compute_dtype
        geometry = layer._quantization_geometry()
        k, contracted, columns = geometry.matrix_contraction()
        n = math.prod(columns)
    shape = tuple(x.shape)
    m = math.prod(shape[: len(shape) - contracted])
    device_class = kernels._device_class()
    kernels_that_run = kernels._required(routes, device_class)
    names = []
    for name, route in routes.items():
        if INTERPRET_ROUTES.get(name) in kernels_that_run:
            continue
        if route.dtypes and dtype not in route.dtypes:
            continue
        if route.available is None or route.available(
            device_class, m, k, n, view, dtype
        ):
            names.append(name)
    return names


def kernel_routes(routes):
    """The routes of `routes` that run a backend kernel: the routes of the
    platform's kernels and the interpret routes."""
    return [
        name
        for name, route in routes.items()
        if route.kernel is not None or name in INTERPRET_ROUTES
    ]


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


def direct_logits(layer, x, in_model=False):
    """A direct call of the route that a reverse lookup of `layer` on `x`
    selects, on the tensors that the layer reads, with the tied LoRA term
    and the soft cap of the reverse lookup after it.

    On its own, a layer under a mixed policy reads its variables in the
    compute dtype. In a model, the `autocast=False` of the layer clears
    the autocast scope of the model.
    """
    dtype = layer.reverse_dtype or layer.compute_dtype
    with AutocastScope(None if in_model else layer.compute_dtype):
        table = lookup_table(layer)
        table = table.with_tensors(table.read_tensors())
        inputs = ops.cast(x, dtype)
        selection = kernels.select_logits_route(layer, table, inputs)
        logits = kernels.run_route(selection, layer, None, table, inputs)
        logits = ops.cast(logits, dtype)
        if layer.tie_weights and layer.lora_enabled:
            lora = ops.matmul(inputs, ops.transpose(layer.lora_embeddings_b))
            lora = ops.matmul(lora, ops.transpose(layer.lora_embeddings_a))
            lora = (layer.lora_alpha / layer.lora_rank) * lora
            logits = ops.add(logits, ops.cast(lora, dtype))
        if layer.logit_soft_cap is not None:
            soft_cap = layer.logit_soft_cap
            logits = ops.multiply(
                ops.tanh(ops.divide(logits, soft_cap)), soft_cap
            )
    return logits
