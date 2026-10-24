"""The allow-list row probe (plan_v3.md 6.4 and 8.3; SESSION.md section 8,
the row sessions). Fork only; not part of a PR.

Run it from the Keras tree, one backend per process (place this file at
benchmarks/quantized_matmul/pod/rows.py):

    python -m benchmarks.quantized_matmul.pod.rows run --route=triton \
        [--level=dense|logits] [--modes=int4_g128,int4_pc,gptq,awq] \
        [--dtypes=bfloat16] [--shapes=4096x4096,...] [--m=1,16] \
        [--logits_shapes=gemma2-2b,...] [--logits_m=1,16] \
        [--candidates=float,F0,n5,triton] [--timing=stream,graph] \
        [--l2_mb=72] [--max_chain_layers=64] [--max_tables=8] \
        [--prefill_layers=8] [--warmup=5] [--samples=5] [--calls=10] \
        [--copy_mb=1024] [--gap_ms=0] [--budget_min=0] [--out=FILE.jsonl]

    python -m benchmarks.quantized_matmul.pod.rows summarize FILE...

`--route` is the route of the row: `triton` (torch, CUDA), `mosaic_gpu`
(JAX, NVIDIA), or `n5` (any backend: the N5 rows). The script ignores
flags that it does not know, and the last value of a flag wins, so the
dry run of run_pod.sh can append the small grid of bench.py.

One cell is a (level, mode, dtype, shape, M). Each cell times these
candidates of one workload, interleaved:
- `float`: float `Dense` layers (or tables) of the same shapes, in the
  compute dtype. For the ratio to the float layer only.
- `F0`: the reference path (the policy `"off"`).
- `F1`, `F3`: the harness forms of bench.py (JAX only by default).
- `n5`: N5 under `force_route("n5")`, the group guard included.
- the route of the row under `force_route(route)`, with the configuration
  of its table row (an allow-list row with no configuration runs the
  same code).
`--candidates` replaces this list. `gptq_act` (a random activation order)
runs only F0, `float` and Triton (the general path): N5 and Mosaic GPU take
the reference branch of the guard there.

The workload: `level=dense` calls a chain of distinct quantized `Dense`
layers on the same inputs (bench.py `--levels=chain`): at least 8 layers,
and weights of at least 4x `--l2_mb` (at most `--max_chain_layers`). So
the weights come from DRAM. Above M = 64 the work bounds the time, not
the L2, and the chain has its first `--prefill_layers` (8) layers.
`level=logits` calls copies of one `ReversibleEmbedding(reverse=True)`
table on the same inputs: tables of at least 4x `--l2_mb` (at least 1,
at most `--max_tables`). The first layer gets the random values of
bench.py; the others copy its tensors on the device (the same values in
distinct memory, for one random fill). The float chain of a (dtype,
shape) and its builds serve every mode.

Timing methods (`--timing`):
- `stream`: `--calls` calls back to back and one sync, per sample;
  `--samples` samples, the candidates in turn (interleaved, with a
  rotating order). The time of a call = a sample / `--calls`. torch and
  TensorFlow run eagerly (the host dispatch overlaps the GPU; a
  host-bound candidate shows its dispatch rate), JAX runs the `jit` of
  the chain. Every candidate can run it: the reference path of torch does
  not capture in a CUDA graph (per-call host copies in packing.py), so the
  rows of torch compare stream times.
- `sync`: one call, then a sync, `--calls` times (bench.py's timing).
- `graph` (torch on CUDA only): `--calls` replays of a CUDA graph of the
  chain and one sync, per sample. A candidate that does not capture gets
  a record with the reason. It gives the GPU time of the route; no rule
  reads it. The capture is `cuda_graph.capture`: one warm-up stream and
  one capture stream per process, the current stream restored after a
  failed capture, the cuBLAS workspaces cleared before each capture.
  After the capture, the first replay must give the bits of an eager
  call (`graph_same_as_eager` in the records of the candidate). A graph
  that does not is not timed: its graph record holds the reason, and a
  route or N5 record fails the check `graph`.
`--gap_ms` > 0 (for nsys): after the warm-up, `--calls` sync calls with a
sleep of `--gap_ms` before each, and a `rowmeta` record with `layers`,
so that `chain_split.py report --meta=<out>` splits the trace per layer.

The checks of each route and of N5 in each cell (plan 7.2), against the
float64 oracle of bench.py on the first 64 rows and 4096 columns of the
first layer: G1 (err <= 1.25 x err(F0) + floor), G2 (err <= bound), G4
(the whole first output is finite) and G5 (two calls give the same
bits); with `--timing=graph`, also `graph` (the first replay of the CUDA
graph gives the bits of an eager call). Floors and bounds by dtype as
tune.py; float32 takes the TF32 values when err(F0) > 1e-5 (a backend
whose float32 matmul runs TF32).

The rule of a row (plan 8.3), on the `stream` medians of each measured
(mode, shape, M) of its (backend, device class, dtype, M bucket), the
checks included:
- `triton`: at most 0.95 x the best non-kernel path (F0, N5, F3), in
  every bucket.
- `mosaic_gpu`: at most 0.90 x the best XLA form (F0, F1, F3) in the
  bucket 1-16, 0.95 x in the other buckets.
- `n5`: at most 0.90 x the reference path (F0).
- a logits route (`level=logits`): at most 0.90 x F0 in the bucket 1-16,
  0.95 x in the bucket 17-64.
A cell where the gate of the route refuses the shape is not routed: the
call takes the fallback, and the rule skips it. `gptq_act` cells report
the Triton general path against F0 and do not enter the rule.

Output, in $KSTAGE_OUT (else ./rows-out): `rows.jsonl` (a `rowplan`
record, then one `rowcell` record per candidate and timing, written per
cell, and one `rowverdict` record per row at the end) and
`rows_summary.json`. One `KPROF` line per row verdict and one with the
copy bandwidth. A verdict holds `pass`, `complete` (every planned cell
measured), the cells, the worst cells and the rule. A row is decided
only when it is complete.

The script stops starting cells 20 s (plus the time of the last group)
before the stage limit (KSTAGE_DEADLINE of run_pod.sh), and on SIGTERM it
writes the verdicts of what it measured. Exit code: 0; 1 when a check of
a route or of N5 failed (a wrong result, not a slow one), or when the
first graph replay of any candidate differs from its eager call; 2 when
no cell was measured.

In interpret mode (the Triton interpreter, or Mosaic GPU on a JAX CPU,
which runs `mosaic_gpu_interpret`), the chain has at most 2 layers (1
table) and 2 samples: the checks hold, the times do not. The Triton
interpreter computes a wrong bfloat16 `tl.dot` (plan 7.5), so a Triton
record in bfloat16 there carries `checks_invalid` and no check.
"""

import argparse
import collections
import functools
import gc
import json
import math
import os
import signal
import statistics
import sys
import time

import numpy as np

import keras
from benchmarks.quantized_matmul import bench
from keras.src import backend
from keras.src import ops
from keras.src.quantizers import kernels

if backend.backend() == "torch":
    import torch

    from benchmarks.quantized_matmul.pod import cuda_graph
    from keras.src.backend.torch.ops.core import get_device
elif backend.backend() == "jax":
    import jax
elif backend.backend() == "tensorflow":
    import tensorflow as tf

OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("rows-out")
MARGIN_S = 20
# Route of a row -> (the baseline candidates, the threshold in the bucket
# 1-16, the threshold in the other buckets). Plan 8.3.
RULES = {
    "triton": (("F0", "n5", "F3"), 0.95, 0.95),
    "mosaic_gpu": (("F0", "F1", "F3"), 0.90, 0.95),
    "n5": (("F0",), 0.90, 0.90),
}
LOGITS_RULE = (("F0",), 0.90, 0.95)
# The interpret routes of the dry run take the rule of their kernel.
KERNEL_OF = {"mosaic_gpu_interpret": "mosaic_gpu"}
# Dot precision -> (G1 floor, G2 bound), plan 7.2 (as tune.py).
TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "tf32": (2e-4, 2**-9),
    "float16": (2e-4, 2**-9),
    "bfloat16": (1e-3, 2**-7),
}
FORMS = ("F1", "F3")
# The reason of a graph whose first replay differs from the eager call.
GRAPH_DIFFERS = "the first CUDA graph replay differs from the eager call"


class Stop(Exception):
    """SIGTERM, or the stage deadline."""


# The reason of the stop: Keras adds text to an exception that a layer call
# raises.
STOPPED = []


def _on_term(signum, frame):
    STOPPED.append("SIGTERM")
    raise Stop("SIGTERM")


def _bucket(m):
    return kernels._m_bucket(m)


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summarize"])
    p.add_argument("files", nargs="*")
    p.add_argument("--route", default="n5")
    p.add_argument("--level", choices=["dense", "logits"], default="dense")
    p.add_argument("--modes", default="int4_g128,int4_pc,gptq,awq")
    p.add_argument("--dtypes", default="bfloat16")
    p.add_argument("--shapes", default="4096x4096")
    p.add_argument("--m", default="1,16")
    p.add_argument("--logits_shapes", default="gemma2-2b")
    p.add_argument("--logits_m", default="1,16")
    p.add_argument("--candidates", default="")
    p.add_argument("--timing", default="stream")
    p.add_argument("--l2_mb", type=float, default=-1)
    p.add_argument("--max_chain_layers", type=int, default=64)
    p.add_argument("--max_tables", type=int, default=8)
    p.add_argument("--prefill_layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--copy_mb", type=int, default=1024)
    p.add_argument("--gap_ms", type=float, default=0)
    p.add_argument("--budget_min", type=float, default=0)
    p.add_argument("--out", default="")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print("ignored flags:", " ".join(unknown), flush=True)
    return args


def _csv(text):
    return [part for part in text.split(",") if part]


# --- The platform. ---


def _cuda():
    return backend.backend() == "torch" and torch.cuda.is_available()


def _device_class():
    device = get_device() if backend.backend() == "torch" else None
    return kernels._device_class(device)


def _sync(outputs):
    if backend.backend() == "jax":
        jax.block_until_ready(outputs)
    elif _cuda():
        torch.cuda.synchronize()
    elif backend.backend() == "tensorflow" and bench._gpus():
        tf.test.experimental.sync_devices()


def _interpreted(route):
    if route == "triton":
        return os.environ.get("TRITON_INTERPRET") == "1"
    if route in ("mosaic_gpu", "mosaic_gpu_interpret"):
        return jax.devices()[0].platform == "cpu"
    return False


# --- The workload. ---


def _reverse_all(parts, x):
    return [part(x, reverse=True) for part in parts]


def _l2_bytes(args):
    if args.l2_mb >= 0:
        return args.l2_mb * 2**20
    if _cuda():
        return torch.cuda.get_device_properties(0).L2_cache_size
    return 0


def _dims(level, shape):
    """`(K, N)` of a projection, `(D, V)` of a table."""
    if level == "dense":
        return tuple(int(d) for d in shape.split("x"))
    d, v, _ = bench.LOGITS[shape]
    return d, v


def _layer(level, mode, shape, dtype, seed=None):
    """A layer of a cell, with the random values of bench.py (`seed`), or
    built and not filled (`seed=None`)."""
    policy = bench._policy(mode, dtype)
    if level == "dense":
        k, n = _dims(level, shape)
        if seed is not None:
            return bench._dense(mode, n, k, dtype, seed)
        layer = keras.layers.Dense(n, use_bias=False, dtype=policy)
        layer.build((None, k))
        return layer
    d, v, tied = bench.LOGITS[shape]
    if seed is not None:
        return bench._table(mode, d, v, tied, dtype, seed)
    layer = keras.layers.ReversibleEmbedding(
        v, d, tie_weights=tied, dtype=policy
    )
    layer.build((None,))
    return layer


class Chain:
    """Distinct layers of one (level, mode, dtype, shape).

    The first layer holds the random values of bench.py. The others copy
    its tensors on the device: the same values in distinct memory, so a
    call reads every layer from DRAM, and the build costs one random fill.
    The checks read the first layer only.
    """

    def __init__(self, args, mode, dtype, shape, seed, interpreted):
        start = time.perf_counter()
        level = args.level
        self.first = _layer(level, mode, shape, dtype, seed)
        self.fn = bench._parallel if level == "dense" else _reverse_all
        one = bench.Step([self.first], self.fn)
        per = bench._nbytes(bench._read_variables(one))
        l2 = _l2_bytes(args)
        if level == "dense":
            count = min(max(8, math.ceil(4 * l2 / per)), args.max_chain_layers)
        else:
            count = min(max(1, math.ceil(4 * l2 / per)), args.max_tables)
        if interpreted:
            count = min(count, 2 if level == "dense" else 1)
        self.layers = [self.first]
        for _ in range(count - 1):
            layer = _layer(level, mode, shape, dtype)
            for target, source in zip(layer.variables, self.first.variables):
                if target.path.split("/")[-1] != source.path.split("/")[-1]:
                    raise ValueError(f"{target.path} != {source.path}")
                target.assign(source.value)
            self.layers.append(layer)
        self.models = {}
        self.build_s = time.perf_counter() - start

    def model(self, m, prefill_layers):
        """`(Step, layers)` at `M`. Above M = 64 the work bounds the time,
        not the L2, so the chain has its first `prefill_layers` layers."""
        count = len(self.layers)
        if m > 64:
            count = min(prefill_layers, count)
        if count not in self.models:
            self.models[count] = bench.Step(self.layers[:count], self.fn)
        return self.models[count], count


def _oracles(args, mode, first, first_float, x_np):
    """The float64 outputs of the float layer and of the quantized layer."""
    if args.level == "dense":
        oracle = bench._dense_oracle
    else:
        oracle = bench._logits_oracle
    floats = oracle(first_float, "float", x_np) if first_float else None
    return floats, oracle(first, mode, x_np)


def _candidates(args, route, mode):
    if args.candidates:
        names = _csv(args.candidates)
    elif args.level == "logits":
        names = ["float", "F0"] + ([route] if route != "n5" else [])
    else:
        names = ["float", "F0"]
        if mode != "gptq_act":
            if backend.backend() == "jax":
                names += list(FORMS)
            names.append("n5")
        if route != "n5":
            names.append(route)
    if mode == "gptq_act":
        names = [n for n in names if n not in FORMS + ("n5",)]
        if route != "triton":
            names = [n for n in names if n != route]
    if args.level == "logits":
        names = [n for n in names if n not in FORMS + ("n5",)]
    return names


def _context(name):
    """`(ctx, expect)` of a candidate."""
    if name in ("float", "F0"):
        return functools.partial(kernels.policy, "off"), None
    route = bench.HARNESS.get(name, name)
    return functools.partial(kernels.force_route, route), route


def _first(outputs):
    y = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return np.asarray(ops.convert_to_numpy(ops.cast(y, "float32")))


def _build(name, model, x, want_graph):
    """`(call, graph, facts)`. Raises `ValueError` with a skip reason."""
    ctx, expect = _context(name)
    before = collections.Counter(kernels.route_counts)
    facts = {}
    if backend.backend() == "jax":
        fn, fargs = bench._workload(model, x)
        start = time.perf_counter()
        with ctx():
            compiled = jax.jit(fn).lower(*fargs).compile()
        text = compiled.as_text()
        memory = compiled.memory_analysis()
        facts.update(
            compile_s=time.perf_counter() - start,
            temp_bytes=getattr(memory, "temp_size_in_bytes", None),
            conditionals=text.count(" conditional("),
            mosaic_calls=text.count('custom_call_target="mosaic_gpu_v2"'),
        )

        def call():
            return compiled(*fargs)

    elif backend.backend() == "torch":

        def call():
            with ctx(), torch.no_grad():
                return model(x)

    else:

        def call():
            with ctx():
                return model(x)

    _sync(call())
    counts = dict(kernels.route_counts - before)
    facts["route_counts"] = counts
    if expect and not counts.get(expect):
        raise ValueError(f"the route did not run: {counts}")
    graph = None
    if want_graph and _cuda():
        graph = _capture(model, x, ctx)
        if callable(graph):
            facts["graph_same_as_eager"] = True
        elif graph == GRAPH_DIFFERS:
            facts["graph_same_as_eager"] = False
    return call, graph, facts


def _built(name, model, layers, x, want_graph, nbytes):
    """The candidate `name` on `model`: a dict, or a skip reason."""
    route = _context(name)[1]
    if route and route not in kernels._ROUTES:
        return f"unknown route {name}"
    try:
        call, graph, facts = _build(name, model, x, want_graph)
    except ValueError as error:
        return str(error)[:400]
    except Exception as error:  # A build failure is a result of the cell.
        return f"failed: {error}"[:400]
    return dict(
        call=call, graph=graph, facts=facts, layers=layers, bytes=nbytes
    )


def _capture(model, x, ctx):
    """A CUDA-graph replay of the chain, or the reason of a failure
    (`GRAPH_DIFFERS` when the first replay differs from the eager call)."""

    def call():
        with ctx(), torch.no_grad():
            return model(x)

    try:
        graph, outputs, same = cuda_graph.capture(call)
    except RuntimeError as error:
        torch.cuda.synchronize()
        return f"the CUDA graph capture failed: {str(error)[:300]}"
    if not same:
        return GRAPH_DIFFERS
    return lambda: (graph.replay(), outputs)[1]


def _samples(runs, args, deadline):
    """Stream samples of `runs` (name -> call), interleaved."""
    names = list(runs)
    times = {name: [] for name in names}
    for s in range(args.samples):
        _check_deadline(deadline, 0)
        shift = s % len(names)
        for name in names[shift:] + names[:shift]:
            call = runs[name]
            _sync(call())
            start = time.perf_counter()
            out = None
            for _ in range(args.calls):
                out = call()
            _sync(out)
            times[name].append((time.perf_counter() - start) / args.calls * 1e6)
    return times


def _sync_times(runs, args):
    times = {name: [] for name in runs}
    for _ in range(args.calls):
        for name, call in runs.items():
            start = time.perf_counter()
            _sync(call())
            times[name].append((time.perf_counter() - start) * 1e6)
    return times


def _check_deadline(deadline, reserve):
    if deadline and time.time() + reserve > deadline:
        STOPPED.append("the stage deadline")
        raise Stop("the stage deadline")


# --- One cell. ---


def _cell(args, route, base, built, oracles, copy_gbps, deadline):
    """The records of one cell. `built`: name -> a dict of `_built`, or a
    skip reason."""
    records = []
    calls = {}
    for name, item in built.items():
        record = dict(base, candidate=name, kind=_kind(name))
        if isinstance(item, str):
            records.append(dict(record, timing="stream", skipped=item))
            continue
        record.update(item["facts"], layers=item["layers"], bytes=item["bytes"])
        call = item["call"]
        oracle = oracles[0] if name == "float" else oracles[1]
        try:
            for _ in range(args.warmup):
                _sync(call())
            first = call()
            second = call()
            _sync(second)
        except Exception as error:  # A failed call is a result of the cell.
            records.append(
                dict(record, timing="stream", skipped=f"failed: {error}"[:400])
            )
            continue
        y1, y2 = _first(first), _first(second)
        record.update(
            err=bench._err(first, oracle),
            finite=bool(np.isfinite(y1).all()),
            deterministic=y1.tobytes() == y2.tobytes(),
        )
        if name == "triton" and base["dtype"] == "bfloat16":
            if os.environ.get("TRITON_INTERPRET") == "1":
                # The interpreter computes a wrong bfloat16 `tl.dot`
                # (plan 7.5); the interpreter tests use `UPCAST_DOT`.
                record["checks_invalid"] = "Triton interpreter, bfloat16"
        calls[name] = (call, item["graph"], record)
    if args.gap_ms:
        for name, (call, _, record) in calls.items():
            times = []
            for _ in range(args.calls):
                time.sleep(args.gap_ms / 1e3)
                start = time.perf_counter()
                _sync(call())
                times.append((time.perf_counter() - start) * 1e6)
            records.append(_timed(record, "gap", times, args, copy_gbps))
        return records
    timings = _csv(args.timing)
    runs = {name: c[0] for name, c in calls.items()}
    if "stream" in timings and runs:
        times = _samples(runs, args, deadline)
        for name, (_, _, record) in calls.items():
            records.append(
                _timed(record, "stream", times[name], args, copy_gbps)
            )
    if "sync" in timings and runs:
        times = _sync_times(runs, args)
        for name, (_, _, record) in calls.items():
            records.append(_timed(record, "sync", times[name], args, copy_gbps))
    if "graph" in timings and _cuda():
        graphs = {}
        for name, (_, graph, record) in calls.items():
            if isinstance(graph, str) or graph is None:
                reason = graph or "no graph"
                records.append(dict(record, timing="graph", skipped=reason))
            else:
                graphs[name] = graph
        if graphs:
            times = _samples(graphs, args, deadline)
            for name in graphs:
                record = calls[name][2]
                records.append(
                    _timed(record, "graph", times[name], args, copy_gbps)
                )
    return records


def _kind(name):
    if name == "float":
        return "float"
    if name in ("F0",) + FORMS:
        return "form"
    if name == "n5":
        return "n5"
    return "route"


def _timed(record, timing, times, args, copy_gbps):
    median = statistics.median(times)
    record = dict(record, timing=timing)
    record.update(
        min_us=min(times),
        median_us=median,
        per_layer_us=median / record["layers"],
        samples=len(times),
        calls=args.calls,
        gbps=record["bytes"] / median / 1e3,
    )
    if copy_gbps:
        record["pct_copy"] = 100 * record["gbps"] / copy_gbps
    return record


# --- The verdicts. ---


def _tolerance(dtype, err_f0):
    if dtype == "float32" and err_f0 is not None and err_f0 > 1e-5:
        return TOLERANCES["tf32"]
    return TOLERANCES.get(dtype, TOLERANCES["float32"])


def _checks(route_record, f0_record):
    """The failed checks of a route or N5 record, by name."""
    if route_record.get("checks_invalid"):
        return []
    failed = []
    err, err_f0 = route_record.get("err"), (f0_record or {}).get("err")
    floor, bound = _tolerance(route_record["dtype"], err_f0)
    if err is None or not math.isfinite(err):
        failed.append("G1/G2: no error")
    else:
        if err_f0 is not None and err > 1.25 * err_f0 + floor:
            failed.append(
                f"G1: err {err:.3g} > 1.25 x {err_f0:.3g} + {floor:g}"
            )
        if err > bound:
            failed.append(f"G2: err {err:.3g} > {bound:.3g}")
    if not route_record.get("finite", False):
        failed.append("G4: not finite")
    if not route_record.get("deterministic", False):
        failed.append("G5: two calls differ")
    if route_record.get("graph_same_as_eager") is False:
        failed.append(f"graph: {GRAPH_DIFFERS}")
    return failed


def verdicts(records):
    """The verdict of each row in `records` (one stage)."""
    plans = [r for r in records if r.get("level") == "rowplan"]
    cells = collections.defaultdict(dict)
    for r in records:
        if r.get("level") != "rowcell" or r.get("timing") != "stream":
            continue
        key = (r["probe_level"], r["mode"], r["dtype"], r["shape"], r["m"])
        cells[key][r["candidate"]] = r
    rows = collections.OrderedDict()
    for key, found in cells.items():
        level, mode, dtype, shape, m = key
        some = next(iter(found.values()))
        stage_route = some["route"]
        kernel = KERNEL_OF.get(stage_route, stage_route)
        evaluated = [stage_route] if stage_route != "n5" else []
        if level == "dense" and "n5" in found:
            evaluated.append("n5")
        for route in evaluated:
            if route not in found:
                continue
            rule_route = "n5" if route == "n5" else kernel
            if level == "logits":
                baselines, decode, other = LOGITS_RULE
            else:
                baselines, decode, other = RULES[rule_route]
            table = "_LOOKUP_ALLOW_LIST" if level == "logits" else "_ALLOW_LIST"
            row_key = (
                table,
                some["backend"],
                some["device_class"],
                dtype,
                _bucket(m),
                rule_route,
            )
            row = rows.setdefault(
                row_key,
                dict(cells=[], gptq_act=[], not_routed=[], failed_checks=[]),
            )
            row["rule"] = (
                f"stream median <= {decode} x min({', '.join(baselines)}) in "
                f"1-16, <= {other} x elsewhere; G1, G2, G4, G5, graph"
            )
            r = found[route]
            cell = dict(mode=mode, shape=shape, m=m)
            if r.get("skipped"):
                # A call that raised is a failure of the route. A route
                # that the gates refuse, or that does not import, is not
                # routed: the call takes the fallback.
                if r["skipped"].startswith("failed"):
                    row["failed_checks"].append(dict(cell, reason=r["skipped"]))
                else:
                    row["not_routed"].append(dict(cell, reason=r["skipped"]))
                continue
            base = [
                found[b]
                for b in baselines
                if b in found and "median_us" in found[b] and b != route
            ]
            if not base:
                continue
            best = min(base, key=lambda b: b["median_us"])
            ratio = r["median_us"] / best["median_us"]
            threshold = decode if _bucket(m) == "1-16" else other
            cell.update(
                route_us=r["median_us"],
                best=best["candidate"],
                best_us=best["median_us"],
                ratio=ratio,
                threshold=threshold,
                pct_copy=r.get("pct_copy"),
            )
            if "float" in found and "median_us" in found["float"]:
                cell["vs_float"] = (
                    r["per_layer_us"] / found["float"]["per_layer_us"]
                )
            checks = _checks(r, found.get("F0"))
            cell["checks"] = checks
            cell["pass"] = not checks and ratio <= threshold
            if mode == "gptq_act":
                row["gptq_act"].append(cell)
            else:
                row["cells"].append(cell)
                if checks:
                    row["failed_checks"].append(cell)
    out = []
    for row_key, row in rows.items():
        table, back, device_class, dtype, bucket, route = row_key
        planned = _planned(plans, route, dtype, bucket)
        measured = {
            (c["mode"], c["shape"], c["m"])
            for c in row["cells"] + row["not_routed"]
        }
        ratios = [c["ratio"] for c in row["cells"]]
        verdict = dict(
            level="rowverdict",
            table=table,
            key=[back, device_class, dtype, bucket],
            route=route,
            pass_=bool(row["cells"]) and all(c["pass"] for c in row["cells"]),
            complete=planned <= measured if planned else None,
            cells=len(row["cells"]),
            planned=len(planned) if planned else None,
            worst_ratio=max(ratios, default=None),
            geomean_ratio=_geomean(ratios),
            failing=sorted(
                (c for c in row["cells"] if not c["pass"]),
                key=lambda c: -c["ratio"],
            )[:6],
            failed_checks=row["failed_checks"][:6],
            not_routed=row["not_routed"][:6],
            gptq_act=row["gptq_act"][:6],
            rule=row["rule"],
            method="stream",
        )
        verdict["pass"] = verdict.pop("pass_")
        out.append(verdict)
    return out


def _planned(plans, route, dtype, bucket):
    planned = set()
    for plan in plans:
        for mode, shape, m in plan["cells"].get(dtype, []):
            if _bucket(m) != bucket or mode == "gptq_act":
                continue
            if route == "n5" and plan["probe_level"] == "logits":
                continue
            planned.add((mode, shape, m))
    return planned


def _geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


# --- run. ---


def _copy_gbps(args):
    size = args.copy_mb * 2**18  # float32 elements.
    zeros, one = ops.zeros(size), ops.ones(())
    if backend.backend() == "jax":
        add = jax.jit(lambda a, b: a + b)
    else:
        add = ops.add
    run = {"copy": lambda: add(zeros, one)}
    for _ in range(3):
        _sync(run["copy"]())
    quick = argparse.Namespace(samples=3, calls=5)
    median = statistics.median(_samples(run, quick, 0)["copy"])
    return 8 * size / median / 1e3


def run(args):
    signal.signal(signal.SIGTERM, _on_term)
    route = args.route
    if backend.backend() == "jax" and route == "mosaic_gpu":
        if jax.devices()[0].platform == "cpu":
            route = "mosaic_gpu_interpret"
    interpreted = _interpreted(route)
    if interpreted:
        args.samples = min(args.samples, 2)
    kernels._ROUTES.update(bench.ROUTES)
    deadline = float(os.environ.get("KSTAGE_DEADLINE") or 0)
    if args.budget_min:
        deadline = time.time() + 60 * args.budget_min
    os.makedirs(OUT, exist_ok=True)
    path = args.out or os.path.join(OUT, "rows.jsonl")
    env = bench._env()
    bench.ENV.update(env)
    print(json.dumps(env), flush=True)
    modes, dtypes = _csv(args.modes), _csv(args.dtypes)
    if args.level == "dense":
        shapes, ms = _csv(args.shapes), [int(m) for m in _csv(args.m)]
    else:
        shapes = _csv(args.logits_shapes)
        ms = [int(m) for m in _csv(args.logits_m)]
    common = dict(
        probe="rows",
        route=route,
        backend=backend.backend(),
        device_class=_device_class(),
        device=env["device"].get("name"),
        capability=env["device"].get("capability"),
        versions=env["versions"],
        keras_sha=env.get("commit"),
        job=os.environ.get("KJOB_ID"),
        stage=os.path.basename(OUT),
    )
    plan = dict(
        common,
        level="rowplan",
        probe_level=args.level,
        cells={
            dtype: [(mode, s, m) for s in shapes for mode in modes for m in ms]
            for dtype in dtypes
        },
    )
    with open(path, "a") as f:
        f.write(json.dumps(plan) + "\n")
    records = [plan]
    copy_gbps = None
    group_seconds = 0.0
    stopped = None
    try:
        if not args.gap_ms:
            copy_gbps = _copy_gbps(args)
            line = dict(common, level="rowcopy", copy_gbps=copy_gbps)
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")
            short = ("level", "route", "backend", "device_class", "device")
            short = {key: line[key] for key in short}
            print("KPROF " + json.dumps(dict(short, copy_gbps=copy_gbps)))
        for dtype in dtypes:
            for shape in shapes:
                _check_deadline(deadline, MARGIN_S + group_seconds)
                start = time.time()
                _group(
                    args,
                    route,
                    interpreted,
                    common,
                    dtype,
                    shape,
                    modes,
                    ms,
                    copy_gbps,
                    deadline,
                    path,
                    records,
                )
                group_seconds = time.time() - start
    except Stop:
        stopped = STOPPED[0] if STOPPED else "stopped"
        print(f"stopped: {stopped}", flush=True)
    return _finish(records, path, stopped)


def _group(
    args,
    route,
    interpreted,
    common,
    dtype,
    shape,
    modes,
    ms,
    copy_gbps,
    deadline,
    path,
    out,
):
    """The cells of one (dtype, shape): every mode and every M, appended
    to `out` cell by cell. The float chain and its builds serve every
    mode."""
    k, n = _dims(args.level, shape)
    floats, float_built = None, {}
    want_graph = "graph" in _csv(args.timing)
    for mode in modes:
        _check_deadline(deadline, MARGIN_S)
        names = _candidates(args, route, mode)
        try:
            chain = Chain(args, mode, dtype, shape, 1, interpreted)
            if floats is None and "float" in names:
                floats = Chain(args, "float", dtype, shape, 1000, interpreted)
        except Exception as error:  # For example a mode with no table.
            record = dict(
                common,
                level="rowerror",
                mode=mode,
                dtype=dtype,
                shape=shape,
                error=f"{type(error).__name__}: {error}"[:400],
            )
            print(f"{mode} {dtype} {shape}: {record['error']}", flush=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
            out.append(record)
            continue
        for m in ms:
            _check_deadline(deadline, MARGIN_S)
            x_np, x = bench._inputs(m, k, dtype)
            first_float = floats.first if floats else None
            oracles = _oracles(args, mode, chain.first, first_float, x_np)
            built = {}
            for name in names:
                source = floats if name == "float" else chain
                if source is None:
                    continue
                model, layers = source.model(m, args.prefill_layers)
                activations = (m * k, dtype), (m * n * layers, dtype)
                nbytes = bench._nbytes(
                    bench._read_variables(model), *activations
                )
                if name == "float" and (m, layers) in float_built:
                    built[name] = float_built[(m, layers)]
                    continue
                built[name] = _built(name, model, layers, x, want_graph, nbytes)
                if name == "float":
                    float_built[(m, layers)] = built[name]
            base = dict(
                common,
                level="rowcell",
                probe_level=args.level,
                mode=mode,
                dtype=dtype,
                shape=shape,
                k=k,
                n=n,
                m=m,
                bucket=_bucket(m),
                build_s=chain.build_s,
            )
            records = _cell(
                args, route, base, built, oracles, copy_gbps, deadline
            )
            if args.gap_ms:
                meta = dict(base, level="rowmeta", candidates=list(built))
                meta["layers"] = next(
                    (r["layers"] for r in records if "layers" in r), None
                )
                records.append(meta)
            with open(path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            for r in records:
                if r.get("level") != "rowcell":
                    continue
                result = r.get("skipped") or (
                    f"{r['per_layer_us']:.2f} us/layer x {r['layers']} "
                    f"({r.get('pct_copy', 0):.0f}% copy) err {r.get('err')}"
                )
                print(
                    mode,
                    dtype,
                    shape,
                    f"M={m}",
                    r["candidate"],
                    r["timing"],
                    result,
                    flush=True,
                )
            out += records
            del built
            gc.collect()
        del chain
        gc.collect()
    del floats, float_built
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()


def _finish(records, path, stopped):
    found = verdicts(records)
    with open(path, "a") as f:
        for v in found:
            f.write(json.dumps(v) + "\n")
    for v in found:
        line = {
            k: v[k]
            for k in (
                "table",
                "key",
                "route",
                "pass",
                "complete",
                "cells",
                "planned",
                "worst_ratio",
                "geomean_ratio",
            )
        }
        if v["failing"]:
            worst = v["failing"][0]
            line["worst"] = {
                k: worst.get(k)
                for k in ("mode", "shape", "m", "ratio", "best", "checks")
            }
        if v["failed_checks"]:
            line["failed_checks"] = len(v["failed_checks"])
        print("KPROF " + json.dumps(line), flush=True)
    cells = [r for r in records if r.get("level") == "rowcell"]
    differs = [
        {k: r.get(k) for k in ("candidate", "mode", "dtype", "shape", "m")}
        for r in cells
        if r.get("timing") == "graph" and r.get("graph_same_as_eager") is False
    ]
    summary = dict(stopped=stopped, verdicts=found, graph_differs=differs)
    with open(
        os.path.join(os.path.dirname(path), "rows_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=1)
    if not any("median_us" in r for r in cells):
        print("no cell was measured", flush=True)
        return 2
    if differs:
        print(f"{GRAPH_DIFFERS}: {len(differs)} candidates", flush=True)
    if any(v["failed_checks"] for v in found):
        print("a check of a route or of N5 failed", flush=True)
        return 1
    if differs:
        return 1
    return 0


# --- summarize. ---


def summarize(files):
    """The verdicts of rows.jsonl files or collect.py session files, per
    stage (a session file holds the records of every stage)."""
    by_stage = collections.defaultdict(list)
    for name in files:
        for line in open(name):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("probe") != "rows" and record.get("level") not in (
                "rowcell",
                "rowplan",
            ):
                continue
            stage = record.get("stage", "-"), record.get("job")
            by_stage[stage].append(record)
    for (stage, job), records in by_stage.items():
        for v in verdicts(records):
            v["stage"], v["job"] = stage, job
            print(json.dumps(v))
    return 0


def main(argv=None):
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.cmd == "summarize":
        return summarize(args.files)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
