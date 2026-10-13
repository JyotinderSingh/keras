"""Benchmarks the routes of a 4-bit weight-only quantized contraction.

Run it from the repository root, one backend per process, so that
`keras` is this checkout:

    KERAS_BACKEND=jax python -m benchmarks.quantized_matmul.bench \
        --out=/tmp/results.jsonl

Each cell appends one JSON line per candidate and timing to `--out`.
`make_tables.py` writes the Markdown tables from the lines. `README.md`
lists the levels, the candidates, the flags and the record fields.
"""

import collections
import contextlib
import dataclasses
import datetime
import functools
import gc
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
from absl import app
from absl import flags

import keras
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import kernels
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.utils.dtype_utils import dtype_size

if backend.backend() == "jax":
    import jax
elif backend.backend() == "torch":
    import torch
elif backend.backend() == "tensorflow":
    import tensorflow as tf

SHAPES = ["4096x4096", "4096x14336", "14336x4096", "8192x8192"]
SHAPES += ["2304x9216", "2304x2304"]
# Logits table -> (D, V, tied). The `tiny` tables are for smoke runs.
LOGITS = {
    "gemma2-2b": (2304, 256000, True),
    "gemma3-1b": (1152, 262144, True),
    "llama3.2-1b": (2048, 128256, True),
    "llama3-8b": (4096, 128256, False),
    "mistral-7b-v0.3": (4096, 32768, False),
    "tiny-tied": (256, 1024, True),
    "tiny-untied": (256, 1024, False),
}
# Decoder -> (D, F, heads, KV heads, head size, blocks, V, tied).
MODELS = {
    "gemma2-2b": (2304, 9216, 8, 4, 256, 26, 256000, True),
    "llama3-8b": (4096, 14336, 32, 8, 128, 32, 128256, False),
    "tiny": (256, 512, 4, 2, 64, 2, 1024, True),
}
# Mode -> the mode of its dtype policy name, or `None` for a float layer.
MODES = {
    "float": None,
    "int4_g128": "int4/128",
    "int4_pc": "int4/-1",
    "gptq": "gptq/4/128",
    "gptq_act": "gptq/4/128",  # With a random activation-order `g_idx`.
    "awq": "awq/4/128",
    "int8": "int8",  # W8A8.
    "int8_wo": "int8",  # Weight-only: no activation quantizer.
}
FOUR_BIT = ("int4_g128", "int4_pc", "gptq", "gptq_act", "awq")
# The mode of the logits table of a decoder, by the mode of the decoder.
HEAD_MODES = {
    "float": "float",
    "int4_pc": "int4_pc",
    "int8": "int8",
    "int8_wo": "int8",
}
GROUP = 128
# `err` compares this many rows and columns of a cell with the oracle.
ERR_ROWS, ERR_COLS = 64, 4096
# The harness forms F1 and F3 run as routes of these names.
HARNESS = {"F1": "harness_f1", "F3": "harness_f3"}
TPU_MODES = ["int4_g128", "int4_pc", "gptq", "awq"]
SUITES = {
    "tpu": dict(
        levels=["dense", "logits", "chain", "model"],
        modes=TPU_MODES,
        dtypes=["bfloat16", "float32"],
        models=["gemma2-2b"],
        l2_mb=0.0,
        timing=["graph"],
    ),
    "tpu-decode": dict(
        levels=["dense", "logits"],
        modes=TPU_MODES,
        shapes=["4096x4096"],
        m=["1", "16"],
        logits_m=["1", "16"],
        forms=["float", "F0", "F3"],
        timing=["graph"],
    ),
}
# The packages whose versions `env` holds, when they are installed.
PACKAGES = ["numpy", "ml-dtypes", "jax", "jaxlib", "libtpu", "torch"]
PACKAGES += ["triton", "tensorflow"]

FLAGS = flags.FLAGS
flags.DEFINE_string("out", None, "The JSON lines file to append to.")
flags.DEFINE_enum("suite", "", ["", *SUITES], "A preset of the flags.")
flags.DEFINE_list(
    "levels", ["raw", "dense", "chain", "logits", "model"], "The levels."
)
flags.DEFINE_list("modes", list(MODES)[1:], "The quantization modes.")
flags.DEFINE_list("dtypes", ["bfloat16"], "The compute dtypes.")
flags.DEFINE_list("shapes", SHAPES, "The projections, as `KxN`.")
flags.DEFINE_list("m", ["1", "4", "16", "64", "256", "2048"], "Rows `M`.")
flags.DEFINE_list("logits_shapes", list(LOGITS)[:5], "The logits tables.")
flags.DEFINE_list("logits_m", ["1", "4", "16", "64"], "Logits rows `M`.")
flags.DEFINE_list("models", ["gemma2-2b", "llama3-8b"], "The decoders.")
flags.DEFINE_list("forms", ["float", "F0", "F1", "F3"], "The forms.")
flags.DEFINE_list("routes", None, "Routes of `kernels._ROUTES` to run.")
flags.DEFINE_list("logits_routes", [], "Routes to run on the logits.")
flags.DEFINE_list("timing", ["graph", "eager"], "The timings.")
flags.DEFINE_integer("warmup", 5, "Warm-up calls per candidate.")
flags.DEFINE_integer("calls", 30, "Timed calls per candidate.")
flags.DEFINE_float("l2_mb", -1, "The L2 cache in MiB; below 0: detect.")
flags.DEFINE_integer("max_chain_layers", 64, "The longest decode chain.")
flags.DEFINE_integer("copy_mb", 1024, "The size of the roofline copy.")

# The backend, versions, device, commit, date and copy bandwidth.
ENV = {}


@dataclasses.dataclass
class Candidate:
    """`fn(*args)` under `ctx()`: one callable of a cell.

    `kind` is `"float"`, `"form"` (a path without a kernel), `"route"`
    or `"head"`. `expect` names the route that `kernels.route_counts`
    must count, or the candidate is skipped. A candidate with a
    `skipped` reason does not run: its record holds the reason.
    """

    kind: str
    fn: object
    args: tuple
    ctx: object = functools.partial(kernels.policy, "off")
    bytes: int = 0
    oracle: object = None
    expect: str = None
    config: object = None
    skipped: str = None


class Step(keras.Model):
    """Runs `fn(layers, inputs)` over built layers.

    It does not autocast: the inputs reach `fn` in their own dtype.
    """

    def __init__(self, layers, fn):
        super().__init__(autocast=False)
        self.parts = layers
        self.fn = fn
        self.built = True

    def call(self, inputs):
        return self.fn(self.parts, inputs)


def _parallel(parts, x):
    return [part(x) for part in parts]


def _reverse(parts, x):
    return parts[0](x, reverse=True)


def _matmul(parts, x):
    return ops.matmul(x, parts[0].kernel)


def _contiguous(views, k):
    """Whether the `g_idx` of each view holds the group `k // G` of each
    row `k`, as a route with `contiguous_groups` needs."""
    for view in views.values():
        if view.g_idx is not None:
            group = min(view.scheme.group_size, k)
            g_idx = ops.convert_to_numpy(view.g_idx).astype("int64")
            if not np.array_equal(g_idx, np.arange(k) // group):
                return False
    return True


def _direct(fn, parts, x):
    """`fn` on the stored tensors of `parts[0]`, not through its call."""
    layer = parts[0]
    view = layer._quantized_weight()
    view = view.with_tensors(view.read_tensors())
    return fn(layer, layer._quantization_geometry(), view, x, None)


# The low-nibble mask of uint8 codes, built once as a device tensor: a
# Python scalar in a call is a host-to-device copy, which a CUDA-graph
# capture refuses.
LOW_NIBBLE = ops.convert_to_tensor(15, "uint8")


def _form(layer, geometry, view, x, config=None, sum_then_scale=True):
    """F3 (sum-then-scale), or F1 (dequantized planes, two matmuls).

    `x` is `(M, K)`. The groups must be contiguous and `N` even.
    """
    dtype = layer.compute_dtype
    codes, n = view.codes, view.layout.orig_len
    if standardize_dtype(codes.dtype) == "int8":
        low = ops.right_shift(ops.left_shift(codes, 4), 4)
    else:
        low = ops.bitwise_and(codes, LOW_NIBBLE)
    grouped = view.zero_point is not None
    groups = view.scale.shape[0] if grouped else 1
    size = codes.shape[0] // groups

    def planes(tensor, dtype):
        """The even and odd columns, as `(groups, 1, N / 2)`."""
        tensor = ops.reshape(ops.cast(tensor, dtype), (groups, n // 2, 2))
        return [ops.expand_dims(t, 1) for t in ops.unstack(tensor, axis=-1)]

    scales = planes(view.scale, "float32" if sum_then_scale else dtype)
    zeros = planes(view.zero_point, dtype) if grouped else (None, None)
    apply = ops.multiply if grouped else ops.divide
    if sum_then_scale:
        if view.input_scales is not None:
            x = ops.divide(x, ops.cast(view.input_scales, dtype))
        x = ops.transpose(ops.reshape(x, (-1, groups, size)), (1, 0, 2))
    outputs = []
    plane_codes = (low, ops.right_shift(codes, 4))
    for plane, scale, zero in zip(plane_codes, scales, zeros):
        w = ops.reshape(ops.cast(plane, dtype), (groups, size, n // 2))
        if grouped:
            w = ops.subtract(w, zero)
        if sum_then_scale:
            y = apply(ops.cast(ops.matmul(x, w), "float32"), scale)
            outputs.append(ops.sum(y, axis=0))
            continue
        w = ops.reshape(apply(w, scale), (groups * size, n // 2))
        if view.input_scales is not None:
            input_scales = ops.cast(view.input_scales, dtype)
            w = ops.divide(w, ops.expand_dims(input_scales, 1))
        outputs.append(ops.matmul(x, w))
    return ops.cast(ops.reshape(ops.stack(outputs, -1), (-1, n)), dtype)


ROUTES = {
    "harness_f1": kernels.RouteSource(
        functools.partial(_form, sum_then_scale=False)
    ),
    "harness_f3": kernels.RouteSource(_form),
}


def _configured(name, config):
    """Registers the route `name` with `config` under a name of its own."""
    if config is None:
        return name
    label = name + "@" + ",".join(f"{k}={v}" for k, v in config.items())
    source = kernels._ROUTES[name]

    def fn(layer, geometry, view, x, _):
        return source.fn(layer, geometry, view, x, config)

    kernels._ROUTES[label] = dataclasses.replace(source, fn=fn)
    return label


def _kind(name):
    return "form" if name in ("F0", "F1", "F3", "n5") else "route"


def _policy(mode, dtype):
    return dtype if MODES[mode] is None else f"{MODES[mode]}_from_{dtype}"


def _fill(layer, mode, seed):
    """Assigns random values to the stored tensors of `layer`."""
    rng = np.random.default_rng(seed)
    divisor = not any("zero" in v.path for v in layer.variables)
    for v in layer.variables:
        name, shape = v.path.split("/")[-1], tuple(v.shape)
        dtype = standardize_dtype(v.dtype)
        if name == "g_idx":
            value = np.arange(shape[0]) // GROUP
            if mode == "gptq_act":
                value[rng.permutation(shape[0])] = value.copy()
        elif name == "awq_scales":
            value = rng.uniform(0.5, 2.0, shape)
        elif "scale" in name:
            low, high = (50, 150) if divisor else (5e-3, 1.5e-2)
            value = rng.uniform(low, high, shape)
        elif dtype == "uint8":
            value = rng.integers(0, 16 if "zero" in name else 256, shape)
        elif dtype == "int8" and "zero" in name:
            value = rng.integers(-8, 8, shape)
        elif dtype == "int8":
            value = rng.integers(-127 if "int8" in mode else -128, 128, shape)
        else:
            value = rng.standard_normal(shape, dtype=np.float32) * 0.02
        v.assign(value.astype(dtype if "int" in dtype else "float32"))


def _dense(mode, units, k, dtype, seed):
    """A `Dense` layer of `mode` with random stored tensors."""
    policy = _policy(mode, dtype)
    layer = keras.layers.Dense(units, use_bias=False, dtype=policy)
    layer.build((None, k))
    if mode == "int8_wo":
        layer.inputs_quantizer = None
    _fill(layer, mode, seed)
    return layer


def _table(mode, d, v, tied, dtype, seed):
    """A `ReversibleEmbedding` of `mode` with random stored tensors."""
    layer = keras.layers.ReversibleEmbedding(
        v, d, tie_weights=tied, dtype=_policy(mode, dtype)
    )
    layer.build((None,))
    _fill(layer, mode, seed)
    return layer


def _numpy(tensor):
    return None if tensor is None else np.asarray(ops.convert_to_numpy(tensor))


def _round(x, dtype):
    dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype
    return x[:ERR_ROWS].astype(dtype).astype(np.float64)


def _dense_oracle(layer, mode, x):
    """The float64 outputs of `layer`, in the first rows and columns."""
    x, cols = _round(x, layer.compute_dtype), ERR_COLS
    if MODES[mode] is None:
        return x @ _numpy(layer.kernel)[:, :cols].astype(np.float64)
    if mode.startswith("int8"):
        scale = _numpy(layer.kernel_scale).astype(np.float64)
        return x @ (_numpy(layer._kernel)[:, :cols] / scale[:cols])
    view = layer._quantized_weight()
    zero = _numpy(view.zero_point)
    columns = min(cols, view.layout.orig_len)
    return oracle_contract(
        x,
        _numpy(view.codes)[:, : cols // 2],
        _numpy(view.scale)[..., :cols],
        columns=columns,
        kernel_shape=(x.shape[1], columns),
        zero_point=None if zero is None else zero[:, :cols],
        g_idx=_numpy(view.g_idx),
        input_scales=_numpy(view.input_scales),
    )


def _vocab(tensor, untied):
    """The first `ERR_COLS` vocabulary entries of a table tensor, by row."""
    if tensor is None:
        return None
    tensor = ops.convert_to_tensor(tensor)
    if untied and len(tensor.shape) == 2:
        return _numpy(tensor[:, :ERR_COLS]).T
    return _numpy(tensor[:ERR_COLS])


def _logits_oracle(layer, mode, x):
    """The float64 logits of `layer`, in the first rows and columns."""
    x, untied = _round(x, layer.compute_dtype), not layer.tie_weights
    prefix = "reverse_embeddings" if untied else "embeddings"
    codes = layer.reverse_embeddings if untied else layer._embeddings
    codes = _vocab(codes, untied)
    if MODES[mode] is None:
        return x @ codes.astype(np.float64).T
    scale = _vocab(getattr(layer, f"{prefix}_scale"), untied)
    zero = _vocab(getattr(layer, f"{prefix}_zero"), untied)
    packed = codes.astype(np.int64)
    q = np.stack([packed & 15, (packed >> 4) & 15], -1)
    q = q.reshape(len(packed), -1)
    q = np.where(q > 7, q - 16, q)
    if zero is None:
        return x @ (q / scale.astype(np.float64)[:, None]).T
    g = _numpy(layer.g_idx).astype(np.int64)
    w = (q - zero.astype(np.int64)[:, g]) * scale.astype(np.float64)[:, g]
    return x @ w.T


def _err(outputs, oracle):
    if oracle is None:
        return None
    y = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    y = _numpy(ops.cast(y, "float32"))[:ERR_ROWS, :ERR_COLS]
    return float(np.max(np.abs(y - oracle)) / np.max(np.abs(oracle)))


def _nbytes(tensors, *elements):
    """The bytes of `tensors` and of `(count, dtype)` activations."""
    pairs = [(math.prod(t.shape), t.dtype) for t in tensors]
    pairs += list(elements)
    return sum(n * dtype_size(standardize_dtype(d)) // 8 for n, d in pairs)


def _workload(model, x):
    """`(fn, args)` of a model call. JAX gets the variables as arguments."""
    if backend.backend() != "jax":
        return model, (x,)

    def fn(trainable, non_trainable, x):
        return model.stateless_call(trainable, non_trainable, x)[0]

    trainable = [v.value for v in model.trainable_variables]
    return fn, (trainable, [v.value for v in model.non_trainable_variables], x)


def _platform():
    return ENV["device"]["platform"]


def _sync(outputs):
    if backend.backend() == "jax":
        jax.block_until_ready(outputs)
    elif backend.backend() == "torch" and _platform() == "cuda":
        torch.cuda.synchronize()
    elif backend.backend() == "torch" and _platform() == "mps":
        torch.mps.synchronize()
    elif backend.backend() == "tensorflow" and _platform() == "gpu":
        tf.test.experimental.sync_devices()


def _build(c, timing):
    """`(run, facts)`; `run()` calls the candidate once, or is `None`.

    The graph timing is a `jit` (JAX), a `tf.function` with XLA, or a
    CUDA-graph replay (torch on CUDA only). A failed capture gives
    `None` and the reason in `facts["skipped"]`.
    """
    no_grad = torch.no_grad if backend.backend() == "torch" else None

    def call(fn=c.fn):
        with c.ctx(), (no_grad or contextlib.nullcontext)():
            return fn(*c.args)

    if timing == "eager":
        return call, {}
    if backend.backend() == "jax":
        start = time.perf_counter()
        with c.ctx():
            compiled = jax.jit(c.fn).lower(*c.args).compile()
        memory = compiled.memory_analysis()
        return (lambda: compiled(*c.args)), {
            "compile_s": time.perf_counter() - start,
            "temp_bytes": getattr(memory, "temp_size_in_bytes", None),
            "hlo": _hlo(compiled.as_text()),
        }
    if backend.backend() == "tensorflow":
        fn = tf.function(c.fn, jit_compile=True)
        with c.ctx():
            hlo = fn.experimental_get_compiler_ir(*c.args)("optimized_hlo")
        return functools.partial(call, fn), {"hlo": _hlo(hlo)}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with c.ctx(), torch.no_grad(), torch.cuda.graph(graph):
            outputs = c.fn(*c.args)
    except RuntimeError as error:
        return None, {"skipped": f"the CUDA graph capture failed: {error}"}
    return (lambda: (graph.replay(), outputs)[1]), {}


def _hlo(text):
    """The custom calls, fusion kinds and largest float temp of a module.

    `max_float_elements` is the largest float buffer of the entry
    computation, parameters excluded: `K * N / 2` or more when XLA
    writes a dequantized weight.
    """
    entry = text[text.index("\nENTRY") :]
    entry = entry[: entry.find("\n}") + 1]
    sizes = [
        math.prod(int(d) for d in dims.split(",") if d)
        for line in entry.splitlines()
        if "parameter(" not in line
        for dims in re.findall(r"= (?:bf16|f16|f32)\[([\d,]*)\]", line)
    ]
    fusions = collections.Counter(re.findall(r"kind=(k\w+)", entry))
    return {
        "custom_calls": sorted(set(re.findall(r'_target="([^"]+)"', text))),
        "fusions": dict(fusions),
        "max_float_elements": max(sizes, default=0),
    }


def _torch_facts(run, timing):
    """The 15 profiler events with the most device time (CUDA) or calls,
    and the peak CUDA bytes of an eager call above the bytes before it."""
    cuda = _platform() == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities += [torch.profiler.ProfilerActivity.CUDA] if cuda else []
    with torch.profiler.profile(activities=activities) as profile:
        _sync(run())

    def key(event):
        return getattr(event, "self_device_time_total", 0) if cuda else 0

    events = sorted(profile.key_averages(), key=lambda e: (key(e), e.count))
    facts = {"events": {e.key: e.count for e in events[::-1][:15]}}
    if cuda and timing == "eager":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _sync(run())
        facts["peak_bytes"] = torch.cuda.max_memory_allocated() - base
    return facts


def _measure(cell, candidates):
    """Times the candidates interleaved, call by call. Returns records.

    A candidate that fails to build or to warm up is recorded as skipped,
    with its error, and the other candidates still run.
    """
    # torch on CPU and MPS has the eager timing only.
    eager_only = backend.backend() == "torch" and _platform() != "cuda"
    records = []
    for timing in FLAGS.timing:
        if timing == "graph" and eager_only:
            continue
        runs = {}
        for name, c in candidates.items():
            record = dict(cell, candidate=name, kind=c.kind, timing=timing)
            record.update(config=c.config, bytes=c.bytes)
            if c.skipped:
                records.append(dict(record, skipped=c.skipped))
                continue
            before = collections.Counter(kernels.route_counts)
            try:
                run, facts = _build(c, timing)
                if run is not None:
                    for _ in range(FLAGS.warmup):
                        _sync(run())
            except Exception as error:
                records.append(dict(record, skipped=f"failed: {error}"))
                continue
            if run is None:
                records.append(dict(record, **facts))
                continue
            counts = dict(kernels.route_counts - before)
            record.update(facts, route_counts=counts)
            if c.expect and not counts.get(c.expect):
                records.append(dict(record, skipped="the route did not run"))
                continue
            if backend.backend() == "torch":
                record.update(_torch_facts(run, timing))
            runs[name] = run, record
        times = {name: [] for name in runs}
        for _ in range(FLAGS.calls):
            for name, (run, _) in runs.items():
                start = time.perf_counter()
                _sync(run())
                times[name].append((time.perf_counter() - start) * 1e3)
        for name, (run, record) in runs.items():
            best = min(times[name])
            gbps = record["bytes"] / best / 1e6
            record.update(min_ms=best, median_ms=statistics.median(times[name]))
            record.update(calls=FLAGS.calls, gbps=gbps)
            if ENV.get("copy_gbps"):
                record["pct_copy"] = 100 * gbps / ENV["copy_gbps"]
            record["err"] = _err(run(), candidates[name].oracle)
            records.append(record)
    return records


def _emit(records):
    """Appends the records to `--out` and prints one line for each."""
    with open(FLAGS.out, "a") as f:
        for r in records:
            f.write(json.dumps(dict(r, env=ENV)) + "\n")
            result = r.get("skipped")
            if result is None:
                result = f"{r['min_ms']:.4g} ms, err {r['err']}"
            keys = ("level", "mode", "dtype", "shape", "m", "candidate")
            print(*(r.get(key) for key in keys), r["timing"], result)
    return records


def _routes(m, views, configs=(None,)):
    """`(name, config, skipped)` of each route of `--routes`.

    `views` maps each distinct `(K, N)` of the cell to a weight view. A
    route whose device and shape checks fail at one of them, and a
    configuration whose blocks do not tile one of them, get the reason
    in `skipped`. `configs` applies to the Pallas routes only (the TPU
    block sweep).
    """
    for name in FLAGS.routes:
        available = kernels._ROUTES[name].available
        if available and not all(
            available(m, k, n, view) for (k, n), view in views.items()
        ):
            yield name, None, "not available at this shape"
            continue
        for config in configs if "pallas" in name else (None,):
            yield name, config, config and _untiled(config, views)


def _untiled(config, shapes):
    """Why the blocks of a sweep configuration do not tile one of the
    `(K, N)` of `shapes`, or `None`."""
    for k, n in shapes:
        if config["BNH"] != "full" and (n // 2) % config["BNH"]:
            return f"BNH={config['BNH']} does not tile N / 2 = {n // 2}"
        if k % config["BK"]:
            return f"BK={config['BK']} does not tile K = {k}"
    return None


def _candidate(kind, model, x, out, route=None, **kwargs):
    """A call of `model` on `x`; under `force_route(route)` when given."""
    if route is not None:
        kwargs["ctx"] = functools.partial(kernels.force_route, route)
        kwargs["expect"] = route
    activations = (math.prod(x.shape), x.dtype), (out, x.dtype)
    nbytes = _nbytes(model.variables, *activations)
    return Candidate(kind, *_workload(model, x), bytes=nbytes, **kwargs)


def _candidates(model, float_model, x, out, oracles, forms=(), routes=()):
    """`float`, `F0`, the harness `forms` and the kernel `routes`."""
    found = {}
    if "float" in FLAGS.forms and float_model is not None:
        found["float"] = _candidate(
            "float", float_model, x, out, oracle=oracles[0]
        )
    if "F0" in FLAGS.forms:
        found["F0"] = _candidate("form", model, x, out, oracle=oracles[1])
    for form in forms:
        found[form] = _candidate(
            "form", model, x, out, HARNESS[form], oracle=oracles[1]
        )
    for name, config, skipped in routes:
        label = _configured(name, config)
        kwargs = dict(oracle=oracles[1], config=config, skipped=skipped)
        found[label] = _candidate(_kind(name), model, x, out, label, **kwargs)
    return found


def _forms(mode):
    """F1 and F3 of `--forms`; they need contiguous groups of 4 bits."""
    if mode not in FOUR_BIT or mode == "gptq_act":
        return []
    return [form for form in ("F1", "F3") if form in FLAGS.forms]


def _inputs(m, k, dtype):
    x = np.random.default_rng(m).standard_normal((m, k)).astype("float32")
    return x, ops.cast(ops.convert_to_tensor(x), dtype)


def _projections(modes):
    """`(mode, dtype, K, N)` of the projection levels."""
    shapes = [tuple(int(d) for d in s.split("x")) for s in FLAGS.shapes]
    for mode, dtype, (k, n) in itertools.product(modes, FLAGS.dtypes, shapes):
        yield mode, dtype, k, n


def _ms(decode=False):
    return [int(m) for m in FLAGS.m if not decode or int(m) <= 16]


def _sweep(mode, dtype, m):
    """The Pallas block sweep of the TPU suite: int4 in groups of 128, in
    bfloat16, at M = 1 and M = 256."""
    if (FLAGS.suite, mode, dtype) != ("tpu", "int4_g128", "bfloat16"):
        return (None,)
    if m not in (1, 256):
        return (None,)
    bm = (16,) if m == 1 else (128, 256)
    blocks = itertools.product((128, 256, 512, "full"), (128, 256, 512), bm)
    return [dict(BNH=bnh, BK=bk, BM=b) for bnh, bk, b in blocks]


def raw_level():
    """The contraction on the stored tensors, without the layer call."""
    modes = [mode for mode in FLAGS.modes if mode in FOUR_BIT]
    for mode, dtype, k, n in _projections(modes):
        layer = _dense(mode, n, k, dtype, 1)
        float_layer = _dense("float", n, k, dtype, 2)
        fns = {"F0": kernels._reference} if "F0" in FLAGS.forms else {}
        fns.update({form: ROUTES[HARNESS[form]].fn for form in _forms(mode)})
        for m in _ms():
            x_np, x = _inputs(m, k, dtype)
            found = {}
            if "float" in FLAGS.forms:
                found["float"] = _candidate(
                    "float",
                    Step([float_layer], _matmul),
                    x,
                    m * n,
                    oracle=_dense_oracle(float_layer, "float", x_np),
                )
            views = {(k, n): layer._quantized_weight()}
            sources = {name: (fn, None) for name, fn in fns.items()}
            for name, _, skipped in _routes(m, views):
                source = kernels._ROUTES[name]
                if source.contiguous_groups and not _contiguous(views, k):
                    # The layer runs the reference path here: the group
                    # guard of `run_route` refuses this `g_idx`.
                    skipped = skipped or "g_idx is not k // G"
                sources[name] = source.fn, skipped
            oracle = _dense_oracle(layer, mode, x_np)
            for name, (fn, skipped) in sources.items():
                model = Step([layer], functools.partial(_direct, fn))
                found[name] = _candidate(
                    _kind(name), model, x, m * n, oracle=oracle, skipped=skipped
                )
            cell = dict(level="raw", mode=mode, dtype=dtype, shape=f"{k}x{n}")
            _emit(_measure(dict(cell, k=k, n=n, m=m, layers=1), found))
        gc.collect()


def projection_level(level):
    """`dense`: one `Dense` layer. `chain`: distinct layers, `M <= 16`.

    Each form and route runs through the policy switch of `kernels`. A
    chain calls each layer on the same inputs, with one sync per call.
    It has at least 8 layers, and quantized weights of at least 4x the
    L2 cache. A chain cell above the copy bandwidth runs once more with
    twice the layers.
    """
    chain = level == "chain"
    l2 = FLAGS.l2_mb * 2**20
    if l2 < 0 and _platform() == "cuda":
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
    for mode, dtype, k, n in _projections(FLAGS.modes):
        first = _dense(mode, n, k, dtype, 1)
        count = math.ceil(4 * max(l2, 0) / _nbytes(first.variables))
        count = min(max(8, count), FLAGS.max_chain_layers) if chain else 1
        cell = dict(level=level, mode=mode, dtype=dtype, shape=f"{k}x{n}")
        for rerun in range(2 if chain else 1):
            layers = [first]
            layers += [
                _dense(mode, n, k, dtype, s) for s in range(2, count + 1)
            ]
            floats = [
                _dense("float", n, k, dtype, 1000 + s) for s in range(count)
            ]
            model = Step(layers, _parallel)
            float_model = Step(floats, _parallel)
            views = {(k, n): first._quantized_weight()}
            over = False
            for m in _ms(decode=chain):
                x_np, x = _inputs(m, k, dtype)
                oracles = (None, None)
                if not chain:
                    oracles = (
                        _dense_oracle(floats[0], "float", x_np),
                        _dense_oracle(first, mode, x_np),
                    )
                routes = ()
                if mode in FOUR_BIT:
                    configs = (None,) if chain else _sweep(mode, dtype, m)
                    routes = _routes(m, views, configs)
                found = _candidates(
                    model,
                    float_model,
                    x,
                    m * n * count,
                    oracles,
                    _forms(mode),
                    routes,
                )
                records = _measure(
                    dict(cell, k=k, n=n, m=m, layers=count, rerun=rerun), found
                )
                over |= any(r.get("pct_copy", 0) > 100 for r in _emit(records))
                del found
            del layers, floats, model, float_model
            gc.collect()
            if not over or count >= FLAGS.max_chain_layers:
                break
            count = min(2 * count, FLAGS.max_chain_layers)


def logits_level():
    """The reverse projection of a tied or untied `ReversibleEmbedding`."""
    modes = [mode for mode in ("int4_g128", "int4_pc") if mode in FLAGS.modes]
    routes = [(name, None, None) for name in FLAGS.logits_routes]
    for name, dtype in itertools.product(FLAGS.logits_shapes, FLAGS.dtypes):
        d, v, tied = LOGITS[name]
        float_table = _table("float", d, v, tied, dtype, 2)
        for mode in modes:
            table = _table(mode, d, v, tied, dtype, 1)
            for m in [int(m) for m in FLAGS.logits_m]:
                x_np, x = _inputs(m, d, dtype)
                oracles = (
                    _logits_oracle(float_table, "float", x_np),
                    _logits_oracle(table, mode, x_np),
                )
                found = _candidates(
                    Step([table], _reverse),
                    Step([float_table], _reverse),
                    x,
                    m * v,
                    oracles,
                    routes=routes,
                )
                cell = dict(level="logits", mode=mode, dtype=dtype, shape=name)
                cell.update(k=d, n=v, m=m, layers=1, tied=tied)
                _emit(_measure(cell, found))
                del found
            del table
            gc.collect()
        del float_table
        gc.collect()


def _decoder(spec, mode, dtype, head_ctx):
    """One decode step of a decoder, and its logits alone.

    A stand-in for the attention mixes the query, key and value outputs
    elementwise, so every projection runs and no KV cache is read. The
    logits run under `head_ctx()`.
    """
    d, f, heads, kv, size, blocks, v, tied = spec
    units = [(heads * size, d), (kv * size, d), (kv * size, d)]
    units += [(d, heads * size), (f, d), (f, d), (d, f)]
    layers = [
        _dense(mode, n, k, dtype, 7 * i + j)
        for i in range(blocks)
        for j, (n, k) in enumerate(units)
    ]
    table = _table(HEAD_MODES.get(mode, "int4_g128"), d, v, tied, dtype, 0)
    # A device tensor, as `LOW_NIBBLE` is.
    epsilon = ops.convert_to_tensor(1e-6, dtype)

    def norm(h):
        mean = ops.mean(ops.square(h), axis=-1, keepdims=True)
        return ops.multiply(h, ops.rsqrt(ops.add(mean, epsilon)))

    def step(parts, h):
        for i in range(blocks):
            q, key, value, o, gate, up, down = parts[7 * i : 7 * i + 7]
            x = norm(h)
            a = ops.multiply(
                ops.reshape(q(x), (-1, kv, heads // kv, size)),
                ops.reshape(key(x), (-1, kv, 1, size)),
            )
            a = ops.add(a, ops.reshape(value(x), (-1, kv, 1, size)))
            h = ops.add(h, o(ops.reshape(a, (-1, heads * size))))
            x = norm(h)
            h = ops.add(h, down(ops.multiply(ops.gelu(gate(x)), up(x))))
        return head(parts[-1:], norm(h))

    def head(parts, h):
        with head_ctx():
            return parts[0](h, reverse=True)

    return Step(layers + [table], step), Step([table], head)


def model_level():
    """A decode step of a decoder with random weights, and its logits.

    The `head` candidate runs the logits alone, and each step record
    holds the logits share. The logits run under the first of
    `--logits_routes`, else on the reference path.
    """
    head_ctx = functools.partial(kernels.policy, "off")
    if FLAGS.logits_routes:
        route = FLAGS.logits_routes[0]
        head_ctx = functools.partial(kernels.force_route, route)
    for name, dtype, mode in itertools.product(
        FLAGS.models, FLAGS.dtypes, ["float"] + FLAGS.modes
    ):
        spec = MODELS[name]
        model, head = _decoder(spec, mode, dtype, head_ctx)
        # The distinct `(K, N)` of the projections of a block.
        views = {}
        if mode in FOUR_BIT:
            for layer in model.parts[:7]:
                views[tuple(layer.kernel_shape)] = layer._quantized_weight()
        for m in _ms(decode=True):
            x, out = _inputs(m, spec[0], dtype)[1], m * spec[6]
            if mode == "float":
                found = {"float": _candidate("float", model, x, out)}
            else:
                routes = _routes(m, views) if mode in FOUR_BIT else ()
                found = _candidates(
                    model, None, x, out, (None, None), _forms(mode), routes
                )
            found["head"] = _candidate("head", head, x, out)
            cell = dict(level="model", mode=mode, dtype=dtype, shape=name)
            records = _measure(dict(cell, m=m, layers=spec[5]), found)
            heads = {
                r["timing"]: r["min_ms"]
                for r in records
                if r["candidate"] == "head" and "min_ms" in r
            }
            for r in records:
                head_ms = heads.get(r["timing"]) if "min_ms" in r else None
                if head_ms and r["candidate"] != "head":
                    r["logits_share"] = head_ms / r["min_ms"]
            _emit(records)
            del found
        del model, head, views
        gc.collect()


LEVELS = {
    "raw": raw_level,
    "dense": functools.partial(projection_level, "dense"),
    "chain": functools.partial(projection_level, "chain"),
    "logits": logits_level,
    "model": model_level,
}


def _gpus():
    return tf.config.list_physical_devices("GPU")


def _env():
    """The backend, library versions, device and commit of the run."""
    versions = {"keras": keras.__version__}
    versions["python"] = platform.python_version()
    for name in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    device = dict(platform="cpu", name=platform.processor() or "cpu")
    device["machine"] = platform.machine()
    if backend.backend() == "jax":
        jax_device = jax.devices()[0]
        device.update(platform=jax_device.platform, name=jax_device.device_kind)
        device["count"] = jax.device_count()
        device["capability"] = getattr(jax_device, "compute_capability", None)
    elif backend.backend() == "torch":
        device["platform"] = ops.convert_to_tensor(np.zeros(1)).device.type
        if device["platform"] == "cuda":
            capability = "%d.%d" % torch.cuda.get_device_capability()
            device.update(capability=capability, cuda=torch.version.cuda)
            device["name"] = torch.cuda.get_device_name()
    elif backend.backend() == "tensorflow" and _gpus():
        details = tf.config.experimental.get_device_details(_gpus()[0])
        device.update(platform="gpu", name=details.get("device_name"))
        device["capability"] = details.get("compute_capability")
    if shutil.which("nvidia-smi"):
        query = ["nvidia-smi", "--query-gpu=name,driver_version"]
        device["driver"] = subprocess.run(
            query + ["--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
    root = os.path.dirname(os.path.dirname(os.path.abspath(keras.__file__)))
    commit = None
    # A tarball checkout, or an image with no `git`, uses `KERAS_SHA`.
    if shutil.which("git"):
        git = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        commit = git.stdout.strip() if git.returncode == 0 else None
    return dict(
        backend=backend.backend(),
        keras_file=keras.__file__,
        versions=versions,
        device=device,
        commit=commit or os.environ.get("KERAS_SHA"),
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def main(_):
    for key, value in SUITES.get(FLAGS.suite, {}).items():
        if not FLAGS[key].present:
            FLAGS[key].value = value
    if FLAGS.routes is None:
        # The test routes (the wrapped reference, interpret modes) run the
        # kernels slowly on the host, so the default leaves them out.
        FLAGS.routes = [
            name
            for name in kernels._ROUTES
            if name != "reference_wrapped" and not name.endswith("_interpret")
        ]
    kernels._ROUTES.update(ROUTES)
    ENV.update(_env())
    print(json.dumps(ENV, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    # The roofline: a copy reads and writes `--copy_mb` MiB once each. The
    # added one is a device tensor, as `LOW_NIBBLE` is.
    size = FLAGS.copy_mb * 2**18
    copy = Candidate("float", ops.add, (ops.zeros(size), ops.ones(())))
    copy.bytes = 8 * size
    records = _emit(_measure(dict(level="copy", m=None), {"copy": copy}))
    gbps = [r["gbps"] for r in records if "gbps" in r]
    ENV["copy_gbps"] = max(gbps, default=None)
    for level in FLAGS.levels:
        LEVELS[level]()


if __name__ == "__main__":
    flags.mark_flag_as_required("out")
    app.run(main)
