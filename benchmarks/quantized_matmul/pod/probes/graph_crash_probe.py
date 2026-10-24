"""The CUDA-graph crash probe of S1x. Fork only; not part of a PR.

S1r job 1 (run s1r-20261023-2, commit d65f349fe) stopped in the CUDA-graph
replays of the rows.py cell float16 gptq 14336x4096 M=256 (the graphs of
float, F0 and triton; the n5 capture failed) with "CUDA error:
unspecified launch failure" (CUDA_ERROR_LAUNCH_FAILED 719 from
cuCtxSynchronize_v2). The same cell passed in bfloat16. This probe finds
the candidate and the kernel, and tells if the crash occurs outside the
harness of rows.py.

Run it from the Keras tree with the torch backend (place this file at
benchmarks/quantized_matmul/pod/probes/graph_crash_probe.py):

    python benchmarks/quantized_matmul/pod/probes/graph_crash_probe.py env
    python benchmarks/quantized_matmul/pod/probes/graph_crash_probe.py \
        drive --part=a|b|c|d|e
    python benchmarks/quantized_matmul/pod/probes/graph_crash_probe.py \
        child '<spec as JSON>'

Each part runs its cases one after the other, each case in a child
process. A crash ends only that child. The parts:

- a: the S1r cell. Each candidate (float, F0, n5, triton) alone in its
  own process, with CUDA_LAUNCH_BLOCKING=1: the eager calls and their
  check against the float64 oracle, then the CUDA-graph capture, 20
  replays and the same check. Then one process with single-layer graphs
  of the parts of each candidate (the dequantize and the matmul of F0,
  the Triton kernel alone, each layer call), to name the kernel. After a
  crash, the parts that did not run go to new processes, and a crashed
  chain candidate gets one process with a graph per layer.
- b: the candidates in one process in the S1r order (the flow of rows.py
  `_cell`: build and capture float, F0, n5 and triton; the stream
  samples; the graph samples), at M 64 and then 256. Then rows.py itself
  on the modes int4_g128 and gptq of float16 14336x4096 at M 64 and 256.
  When one of them crashes: the flow again with CUDA_LAUNCH_BLOCKING=1,
  and with no n5, no F0 graph, no triton graph and no float graph.
- c: part a for triton at M 65, 128 and 256, for the shapes 4096x14336
  and 14336x4096, in float16 and bfloat16, for GPTQ, GPTQ with an
  activation order (`gptq_act`) and AWQ. One process per shape and
  dtype; after a crash, the cases that did not run go to new processes,
  one each.
- d: compute-sanitizer --tool memcheck on the first case that crashed in
  a, b or c (else on the S1r cell, triton and F0 alone), when the image
  has compute-sanitizer. Else it says so.
- e: when no part crashed: rows.py with the command of the S1r stage
  gt-triton-prefill, until the stage deadline. In S1r the crash came
  after 40 cells of that stage.

The check of a candidate: `err` is the largest absolute error of the
first layer's output over the first 64 rows and 4096 columns, divided by
the largest absolute value of the float64 oracle there (bench.py
`_err`); `err_all` takes every row. G1: `err <= 1.25 x err(F0) +
floor`; G2: `err <= bound`; float16 floor 2e-4 and bound 2**-9,
bfloat16 1e-3 and 2**-7 (rows.py). The float candidate has its own
float64 oracle and G2 only. The layers of a chain are copies of the
first layer, so each layer's output must have the bits of the first
(`layers_equal`). The graph outputs must have the bits of the eager
outputs (`same_as_eager`), after each replay.

The capture is `cuda_graph.capture` of the pod folder (one warm-up stream
and one capture stream per process, the current stream restored after a
failed capture, the cuBLAS workspaces cleared before each capture). Each
capture also emits a `check` event of phase `first-replay`: whether the
first replay gives the bits of an eager call of the same function.

Output, in $KSTAGE_OUT (else ./crash-probe-out): `probe.jsonl` (each
event of each child, with `part` and `case`), `summary-<part>.json` (one
summary per case), `err/<case>.txt` (the end of the stderr of each
child), `dot-<case>.json` (the CUDA graph of each capture of part a, as
cudaGraphDebugDotPrint gives it), `rows-<case>.txt` (the output of a
rows.py run) and `rows-<case>/` (its rows.jsonl and rows_summary.json),
`sanitizer-<case>.txt`. One `KPROF` line per case.

Exit code: 0; 1 when a child crashed with a CUDA error; 2 on an error of
the probe itself. On a host with no CUDA (the dry run), the probe runs
small cases on the Triton interpreter, with no graphs.
"""

import argparse
import functools
import gc
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.abspath(__file__)
OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("crash-probe-out")
# Dot precision -> (G1 floor, G2 bound), as rows.py.
TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "float16": (2e-4, 2**-9),
    "bfloat16": (1e-3, 2**-7),
}
CELL = dict(mode="gptq", dtype="float16", shape="14336x4096", m=256)
CANDIDATES = ("float", "F0", "n5", "triton")
# The flags of the S1r stage gt-triton-prefill.
S1R_PREFILL = [
    "--route=triton",
    "--timing=stream,graph",
    "--modes=int4_g128,int4_pc,gptq,awq",
    "--dtypes=bfloat16,float16",
    "--shapes=4096x4096,14336x4096,4096x14336",
    "--m=64,256,2048",
    "--l2_mb=72",
]
CUDA_ERROR = re.compile(
    r"CUDA error|AcceleratorError|CUDA_ERROR|illegal memory|launch failure"
    r"|cudaError|device-side assert|misaligned address"
)


def _gpu():
    if os.environ.get("KMODE") == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# --- The child: one case in this process. ---


def emit(**fields):
    print("EVENT " + json.dumps(fields, default=str), flush=True)


class Dead(Exception):
    """A CUDA error: the context of this process cannot run more work."""


def child(spec):
    import ml_dtypes
    import torch

    import keras
    from benchmarks.quantized_matmul import bench
    from keras.src import ops
    from keras.src.quantizers import kernels
    from keras.src.quantizers.quantization_test_utils import oracle_contract

    gpu = torch.cuda.is_available()
    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    launches = {}
    if gpu and not interpreted:
        _hook_triton(launches, torch)
    state = dict(dead=False)

    def sync():
        if gpu:
            torch.cuda.synchronize()

    def memory():
        if not gpu:
            return None
        return dict(
            allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
            reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 1),
        )

    # A test switch of the driver logic on a host with no CUDA:
    # KPROBE_FAKE_CRASH=<case prefix>:<step> raises a CUDA error at that
    # step. No session file sets it.
    fake = os.environ.get("KPROBE_FAKE_CRASH", "").partition(":")

    def guard(where, fn, **fields):
        """Runs `fn`; an error is an event. A CUDA error raises `Dead`."""
        emit(ev="begin", where=where, **fields)
        try:
            if fake[0] and spec["case"].startswith(fake[0]) and where == fake[2]:
                raise RuntimeError(
                    "CUDA error: unspecified launch failure (KPROBE_FAKE_CRASH)"
                )
            return fn()
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            cuda = bool(CUDA_ERROR.search(text))
            emit(
                ev="error",
                where=where,
                cuda=cuda,
                error=text[:1500],
                trace=traceback.format_exc()[-2500:],
                **fields,
            )
            if cuda:
                state["dead"] = True
                raise Dead(text) from error
            raise

    def context(name):
        if name in ("float", "F0"):
            return functools.partial(kernels.policy, "off")
        return functools.partial(kernels.force_route, name)

    def numpy(tensor):
        return bench._numpy(tensor)

    def oracle_all(layer, mode, x_np):
        """bench.py `_dense_oracle` over every row of `x_np`."""
        cols = bench.ERR_COLS
        rounding = layer.compute_dtype
        cast = ml_dtypes.bfloat16 if rounding == "bfloat16" else rounding
        x = x_np.astype(cast).astype(np.float64)
        if mode == "float":
            return x @ numpy(layer.kernel)[:, :cols].astype(np.float64)
        view = layer._quantized_weight()
        zero = numpy(view.zero_point)
        columns = min(cols, view.layout.orig_len)
        return oracle_contract(
            x,
            numpy(view.codes)[:, : cols // 2],
            numpy(view.scale)[..., :cols],
            columns=columns,
            kernel_shape=(x.shape[1], columns),
            zero_point=None if zero is None else zero[:, :cols],
            g_idx=numpy(view.g_idx),
            group_size=view.scheme.group_size,
            input_scales=numpy(view.input_scales),
        )

    def errors(outputs, oracle):
        """`(err, err_all)` of the first layer's output."""
        y = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        y = numpy(ops.cast(y, "float32"))[:, : oracle.shape[1]]
        y = y[: oracle.shape[0]]
        top = oracle[: bench.ERR_ROWS]
        err = float(
            np.max(np.abs(y[: bench.ERR_ROWS] - top)) / np.max(np.abs(top))
        )
        err_all = float(np.max(np.abs(y - oracle)) / np.max(np.abs(oracle)))
        return err, err_all, bool(np.isfinite(y).all())

    def as_list(outputs):
        return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

    def same(a, b):
        return all(
            torch.equal(x, y) for x, y in zip(as_list(a), as_list(b))
        ) and len(as_list(a)) == len(as_list(b))

    def layers_equal(outputs):
        outputs = as_list(outputs)
        return all(torch.equal(outputs[0], o) for o in outputs[1:])

    def clone(outputs):
        return [o.detach().clone() for o in as_list(outputs)]

    # --- The workload of rows.py: chains of copies of one layer. ---

    def chain(mode, dtype, shape, seed, count=None):
        k, n = (int(d) for d in shape.split("x"))
        first = bench._dense(mode, n, k, dtype, seed)
        if count is None:
            one = bench.Step([first], bench._parallel)
            per = bench._nbytes(bench._read_variables(one))
            l2 = spec.get("l2_mb", 72) * 2**20
            count = min(max(8, math.ceil(4 * l2 / per)), 64)
        if not gpu:
            count = min(count, 2)
        layers = [first]
        policy = bench._policy(mode, dtype)
        for _ in range(count - 1):
            layer = keras.layers.Dense(n, use_bias=False, dtype=policy)
            layer.build((None, k))
            for target, source in zip(layer.variables, first.variables):
                target.assign(source.value)
            layers.append(layer)
        return layers

    def model(layers, m):
        count = len(layers)
        if m > 64:
            count = min(spec.get("prefill_layers", 8), count)
        return bench.Step(layers[:count], bench._parallel), count

    def capture(fn, dump=None, **where):
        """rows.py `_capture` (`cuda_graph.capture`): a CUDA graph of `fn`
        and its outputs, or the reason. `where` names the candidate in the
        `first-replay` check event."""
        if not gpu:
            return "no CUDA"
        from benchmarks.quantized_matmul.pod import cuda_graph

        graph = torch.cuda.CUDAGraph()
        if dump:
            graph.enable_debug_mode()
        try:
            graph, outputs, first_same = cuda_graph.capture(fn, graph=graph)
        except RuntimeError as error:
            # As rows.py: a failed capture is a result. A sticky CUDA error
            # shows at the next call.
            torch.cuda.synchronize()
            return f"the CUDA graph capture failed: {error}"[:600]
        emit(
            ev="check", phase="first-replay", same_as_eager=first_same, **where
        )
        if dump:
            try:
                graph.debug_dump(dump)
            except Exception as error:  # The dump is optional.
                emit(ev="note", note=f"debug_dump failed: {error}"[:300])
        return graph, outputs

    tolerance = TOLERANCES[spec["dtype"]]

    def check(name, outputs, oracle, phase, extra=None):
        err, err_all, finite = errors(outputs, oracle)
        fields = dict(
            ev="check",
            cand=name,
            phase=phase,
            err=err,
            err_all=err_all,
            finite=finite,
            layers_equal=layers_equal(outputs),
            floor=tolerance[0],
            bound=tolerance[1],
        )
        if name == "triton" and spec["dtype"] == "bfloat16" and interpreted:
            fields["checks_invalid"] = "Triton interpreter, bfloat16"
        fields.update(extra or {})
        emit(**fields)
        return err

    def route_of(name, fn):
        before = dict(kernels.route_counts)
        out = fn()
        sync()
        counts = {
            k: v - before.get(k, 0)
            for k, v in kernels.route_counts.items()
            if v != before.get(k, 0)
        }
        return out, counts

    def err_f0(layers, x, oracle):
        """The eager error of F0 on the first layer, for the G1 bound."""
        layer = layers[0]

        def run():
            with kernels.policy("off"), torch.no_grad():
                return layer(x)

        out = guard("F0-eager-single", run)
        sync()
        err, err_all, _ = errors([out], oracle)
        emit(ev="err_f0", err=err, err_all=err_all)

    # --- Flows. ---

    def alone(name, layers, m, x, oracle, dump_tag):
        """Eager calls and the check, then a graph, replays and checks."""
        step, count = model(layers, m)
        ctx = context(name)

        def call():
            with ctx(), torch.no_grad():
                return step(x)

        emit(ev="cand", cand=name, m=m, layers=count, memory=memory())
        first, counts = guard(
            "eager", lambda: route_of(name, call), cand=name, m=m
        )
        for _ in range(spec.get("warmup", 5)):
            guard("eager", call, cand=name, m=m)
        first = guard("eager", call, cand=name, m=m)
        second = guard("eager", call, cand=name, m=m)
        guard("eager-sync", sync, cand=name, m=m)
        eager = clone(first)
        check(
            name,
            first,
            oracle,
            "eager",
            dict(m=m, route_counts=counts, deterministic=same(first, second)),
        )
        if spec.get("no_graph") or not gpu:
            emit(ev="graph", cand=name, m=m, skipped="no CUDA" if not gpu else "")
            return
        path = None
        if spec.get("dump"):
            path = os.path.join(spec["out"], f"dot-{dump_tag}-{name}-m{m}.dot")
        built = guard(
            "capture",
            lambda: capture(call, path, cand=name, m=m),
            cand=name,
            m=m,
        )
        if isinstance(built, str):
            emit(ev="graph", cand=name, m=m, skipped=built, memory=memory())
            return
        graph, outputs = built
        emit(ev="graph", cand=name, m=m, captured=True, memory=memory())
        replays = spec.get("replays", 20)
        differ = []
        for i in range(replays):
            guard("replay", graph.replay, cand=name, m=m, i=i)
            guard("replay-sync", sync, cand=name, m=m, i=i)
            if not same(outputs, eager):
                differ.append(i)
        check(
            name,
            outputs,
            oracle,
            "graph",
            dict(
                m=m,
                replays=replays,
                same_as_eager=not differ,
                differ_at=differ[:10],
            ),
        )
        del graph, outputs
        sync()
        gc.collect()
        torch.cuda.empty_cache()

    def subops(layers, m, x, oracle, floats, float_oracle, only):
        """Single-layer graphs of the parts of each candidate."""
        layer = layers[0]
        geometry = layer._quantization_geometry()
        view = layer._quantized_weight()
        view = view.with_tensors(view.read_tensors())
        dtype = layer.compute_dtype

        def f0_dequant():
            with torch.no_grad():
                return view.dequantize(dtype)

        w = guard("eager", f0_dequant, cand="F0-dequant")
        sync()
        w = w.detach().clone()

        def f0_matmul():
            with torch.no_grad():
                return geometry.contract(x, w)

        def f0_layer():
            with kernels.policy("off"), torch.no_grad():
                return layer(x)

        def tr_kernel():
            with torch.no_grad():
                return kernels._triton(layer, geometry, view, x)

        def tr_layer():
            with kernels.force_route("triton"), torch.no_grad():
                return layer(x)

        def float_layer():
            with kernels.policy("off"), torch.no_grad():
                return floats[0](x)

        items = [
            ("float-layer", float_layer, float_oracle),
            ("F0-dequant", f0_dequant, None),
            ("F0-matmul", f0_matmul, oracle),
            ("F0-layer", f0_layer, oracle),
            ("triton-kernel", tr_kernel, oracle),
            ("triton-layer", tr_layer, oracle),
        ]
        for name, fn, item_oracle in items:
            if only and name not in only:
                continue
            emit(ev="cand", cand=name, m=m, memory=memory())
            first = guard("eager", fn, cand=name, m=m)
            guard("eager-sync", sync, cand=name, m=m)
            eager = clone(first)
            if item_oracle is not None:
                check(name, [first], item_oracle, "eager", dict(m=m))
            if not gpu:
                emit(ev="graph", cand=name, m=m, skipped="no CUDA")
                emit(ev="subop_done", cand=name)
                continue
            path = None
            if spec.get("dump"):
                path = os.path.join(
                    spec["out"], f"dot-{spec['case']}-{name}-m{m}.dot"
                )
            built = guard(
                "capture",
                lambda: capture(fn, path, cand=name, m=m),
                cand=name,
                m=m,
            )
            if isinstance(built, str):
                emit(ev="graph", cand=name, m=m, skipped=built)
                emit(ev="subop_done", cand=name)
                continue
            graph, outputs = built
            emit(ev="graph", cand=name, m=m, captured=True, memory=memory())
            differ = []
            for i in range(spec.get("replays", 20)):
                guard("replay", graph.replay, cand=name, m=m, i=i)
                guard("replay-sync", sync, cand=name, m=m, i=i)
                if not same(outputs, eager):
                    differ.append(i)
            extra = dict(m=m, same_as_eager=not differ, differ_at=differ[:10])
            if item_oracle is not None:
                check(name, [outputs], item_oracle, "graph", extra)
            else:
                emit(ev="check", cand=name, phase="graph", **extra)
            del graph, outputs
            sync()
            gc.collect()
            torch.cuda.empty_cache()
            emit(ev="subop_done", cand=name)

    def per_layer(name, layers, m, x, oracle):
        """One graph per layer of the chain, replayed in order."""
        _, count = model(layers, m)
        ctx = context(name)
        graphs = []
        for index, layer in enumerate(layers[:count]):

            def call(layer=layer):
                with ctx(), torch.no_grad():
                    return layer(x)

            guard("eager", call, cand=name, layer=index)
            built = guard(
                "capture",
                lambda: capture(call, cand=name, layer=index),
                cand=name,
                layer=index,
            )
            if isinstance(built, str):
                emit(ev="graph", cand=name, layer=index, skipped=built)
                return
            graphs.append(built)
        for i in range(spec.get("replays", 20)):
            for index, (graph, _) in enumerate(graphs):
                guard("replay", graph.replay, cand=name, layer=index, i=i)
                guard("replay-sync", sync, cand=name, layer=index, i=i)
        check(name, [graphs[0][1]], oracle, "graph-per-layer", dict(m=m))

    def s1r(layers, floats, ms, mode, dtype):
        """The flow of rows.py `_group` and `_cell` for one mode."""
        names = [n for n in spec["candidates"]]
        skip_capture = set(spec.get("skip_capture", []))
        float_built = {}
        k = int(spec["shape"].split("x")[0])
        for m in ms:
            x_np, x = bench._inputs(m, k, dtype)
            oracle = oracle_all(layers[0], mode, x_np[:256])
            float_oracle = oracle_all(floats[0], "float", x_np[:256])
            built = {}
            for name in names:
                source = floats if name == "float" else layers
                step, count = model(source, m)
                if name == "float" and (m, count) in float_built:
                    built[name] = float_built[(m, count)]
                    continue
                ctx = context(name)

                def call(step=step, ctx=ctx):
                    with ctx(), torch.no_grad():
                        return step(x)

                out, counts = guard(
                    "build-eager", lambda: route_of(name, call), cand=name, m=m
                )
                emit(ev="cand", cand=name, m=m, layers=count, route_counts=counts)
                graph = "not captured (variant)"
                if name not in skip_capture:
                    graph = guard(
                        "capture",
                        lambda: capture(call, cand=name, m=m),
                        cand=name,
                        m=m,
                    )
                if isinstance(graph, str):
                    emit(ev="graph", cand=name, m=m, skipped=graph)
                else:
                    emit(ev="graph", cand=name, m=m, captured=True, memory=memory())
                built[name] = dict(call=call, graph=graph)
                if name == "float":
                    float_built[(m, count)] = built[name]
            calls = {}
            eager = {}
            for name, item in built.items():
                call = item["call"]
                for _ in range(spec.get("warmup", 5)):
                    guard("warmup", lambda: (call(), sync()), cand=name, m=m)
                first = guard("first", call, cand=name, m=m)
                second = guard("second", call, cand=name, m=m)
                guard("first-sync", sync, cand=name, m=m)
                eager[name] = clone(first)
                check(
                    name,
                    first,
                    float_oracle if name == "float" else oracle,
                    "eager",
                    dict(m=m, deterministic=same(first, second)),
                )
                calls[name] = item
            runs = {n: c["call"] for n, c in calls.items()}
            samples(runs, m, "stream")
            graphs = {
                n: c["graph"]
                for n, c in calls.items()
                if not isinstance(c["graph"], str)
            }
            if graphs and gpu:
                samples(
                    {n: (lambda g=g: g[0].replay()) for n, g in graphs.items()},
                    m,
                    "graph",
                )
                for name, (graph, outputs) in graphs.items():
                    check(
                        name,
                        outputs,
                        float_oracle if name == "float" else oracle,
                        "graph",
                        dict(m=m, same_as_eager=same(outputs, eager[name])),
                    )
            emit(ev="m_done", m=m, memory=memory())
            del built, calls, graphs, eager
            gc.collect()

    def samples(runs, m, kind):
        """rows.py `_samples`: interleaved samples, a rotating order."""
        names = list(runs)
        count = spec.get("samples", 5) if gpu else 1
        calls = spec.get("calls", 10) if gpu else 1
        for s in range(count):
            shift = s % len(names)
            for name in names[shift:] + names[:shift]:
                run = runs[name]
                emit(ev="begin", where=f"{kind}-sample", cand=name, m=m, s=s)

                def sample(run=run):
                    sync()
                    run()
                    sync()
                    for _ in range(calls):
                        run()
                    sync()

                try:
                    sample()
                except Exception as error:
                    text = f"{type(error).__name__}: {error}"
                    emit(
                        ev="error",
                        where=f"{kind}-sample",
                        cand=name,
                        m=m,
                        s=s,
                        cuda=bool(CUDA_ERROR.search(text)),
                        error=text[:1500],
                        trace=traceback.format_exc()[-2500:],
                    )
                    if CUDA_ERROR.search(text):
                        state["dead"] = True
                        raise Dead(text) from error
                    raise
        emit(ev="samples_done", kind=kind, m=m)

    # --- Run the spec. ---
    emit(
        ev="start",
        case=spec["case"],
        flow=spec["flow"],
        launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING", ""),
        device=torch.cuda.get_device_name() if gpu else "cpu",
        torch=torch.__version__,
        pid=os.getpid(),
    )
    status = "ok"
    try:
        flow = spec["flow"]
        mode, dtype, shape = spec["mode"], spec["dtype"], spec["shape"]
        k = int(shape.split("x")[0])
        ms = spec["ms"]
        if flow == "s1r":
            layers = chain(mode, dtype, shape, 1)
            floats = chain("float", dtype, shape, 1000)
            s1r(layers, floats, ms, mode, dtype)
        else:
            names = spec.get("candidates", [])
            layers = chain(mode, dtype, shape, 1, spec.get("layers"))
            floats = None
            if "float" in names or flow == "subops":
                count = 1 if flow == "subops" else spec.get("layers")
                floats = chain("float", dtype, shape, 1000, count)
            inputs = {m: bench._inputs(m, k, dtype) for m in ms}
            rows = np.concatenate([inputs[m][0] for m in ms])
            stacked = oracle_all(layers[0], mode, rows)
            float_stacked = None
            if floats is not None:
                float_stacked = oracle_all(floats[0], "float", rows)
            oracles, float_oracles, start = {}, {}, 0
            for m in ms:
                oracles[m] = stacked[start : start + m]
                if float_stacked is not None:
                    float_oracles[m] = float_stacked[start : start + m]
                start += m
            for m in ms:
                x = inputs[m][1]
                if flow == "alone":
                    for name in names:
                        source = floats if name == "float" else layers
                        oracle = float_oracles[m] if name == "float" else oracles[m]
                        alone(name, source, m, x, oracle, spec["case"])
                elif flow == "subops":
                    subops(
                        layers,
                        m,
                        x,
                        oracles[m],
                        floats,
                        float_oracles[m],
                        spec.get("only"),
                    )
                elif flow == "per_layer":
                    for name in names:
                        source = floats if name == "float" else layers
                        oracle = float_oracles[m] if name == "float" else oracles[m]
                        per_layer(name, source, m, x, oracle)
                if spec.get("err_f0"):
                    err_f0(layers, x, oracles[m])
                emit(ev="m_done", m=m, memory=memory())
    except Dead:
        status = "cuda_error"
    except Exception as error:
        status = "error"
        emit(
            ev="error",
            where="child",
            cuda=False,
            error=f"{type(error).__name__}: {error}"[:1500],
            trace=traceback.format_exc()[-2500:],
        )
    if launches:
        for entry in launches.values():
            emit(ev="triton", **entry)
    if status == "ok":
        try:
            sync()
        except Exception as error:
            status = "cuda_error"
            emit(ev="error", where="final-sync", cuda=True, error=str(error)[:1500])
    emit(ev="end", status=status)
    sys.stdout.flush()
    # A dead context can hang at exit; leave at once.
    os._exit(0 if status == "ok" else (3 if status == "cuda_error" else 4))


def _hook_triton(record, torch):
    """Records each compiled Triton kernel and grid of the launches."""
    try:
        from triton.runtime import jit
    except ImportError:
        return
    original = jit.JITFunction.run

    def run(fn, *args, **kwargs):
        kernel = original(fn, *args, **kwargs)
        try:
            if kernel is None or kwargs.get("warmup"):
                return kernel
            grid = kwargs.get("grid")
            grid = list(grid) if isinstance(grid, (tuple, list)) else str(grid)
            key = (str(getattr(kernel, "hash", "")), str(grid))
            entry = record.get(key)
            if entry is None:
                capturing = torch.cuda.is_current_stream_capturing()
                if getattr(kernel, "n_regs", None) is None and not capturing:
                    if hasattr(kernel, "_init_handles"):
                        kernel._init_handles()
                meta = kernel.metadata
                skip = ("grid", "warmup", "debug", "instrumentation_mode")
                entry = record[key] = dict(
                    kernel=kernel.name,
                    hash=key[0][:12],
                    grid=grid,
                    programs=(
                        math.prod(grid) if isinstance(grid, list) else None
                    ),
                    n_regs=getattr(kernel, "n_regs", None),
                    n_spills=getattr(kernel, "n_spills", None),
                    shared=getattr(meta, "shared", None),
                    num_warps=getattr(meta, "num_warps", None),
                    num_stages=getattr(meta, "num_stages", None),
                    constexprs={
                        k: v
                        for k, v in kwargs.items()
                        if k not in skip and isinstance(v, (bool, int, str))
                    },
                    ints=[
                        a
                        for a in args
                        if isinstance(a, int) and not isinstance(a, bool)
                    ],
                    tensors=[
                        [list(a.shape), str(a.dtype), a.data_ptr() % 256]
                        for a in args
                        if hasattr(a, "data_ptr")
                    ],
                    launches=0,
                )
            entry["launches"] += 1
        except Exception as error:  # The record is optional.
            record.setdefault(("error", ""), dict(error=str(error)[:300]))
        return kernel

    jit.JITFunction.run = run


# --- The driver. ---


class Driver:
    def __init__(self, part, deadline):
        self.part = part
        self.deadline = deadline
        self.gpu = _gpu()
        self.summaries = []
        os.makedirs(os.path.join(OUT, "err"), exist_ok=True)
        self.events_path = os.path.join(OUT, "probe.jsonl")
        self.current = None
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, signum, frame):
        """The stage timeout: end the child too (it has its own session)."""
        if self.current is not None:
            try:
                os.killpg(self.current.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        print("SIGTERM: the probe stops", flush=True)
        sys.exit(2)

    def left(self):
        return self.deadline - time.time() if self.deadline else 1e9

    def run(self, spec, env=None, timeout=240):
        """Runs one child; returns its summary."""
        case = spec["case"]
        spec = dict(spec, out=OUT)
        budget = min(timeout, self.left() - 10)
        if budget < 20:
            summary = dict(case=case, status="not run", reason="stage deadline")
            self._finish(summary)
            return summary
        child_env = dict(os.environ)
        child_env.update(env or {})
        err_path = os.path.join(OUT, "err", f"{case}.txt")
        start = time.time()
        events = []
        with open(err_path, "wb") as err:
            proc = subprocess.Popen(
                [sys.executable, HERE, "child", json.dumps(spec)],
                stdout=subprocess.PIPE,
                stderr=err,
                env=child_env,
                text=True,
                start_new_session=True,
            )
            self.current = proc
            timer = _Timer(proc, budget)
            other = []
            for line in proc.stdout:
                if not line.startswith("EVENT "):
                    other = (other + [line.rstrip()])[-30:]
                if line.startswith("EVENT "):
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    events.append(event)
                    with open(self.events_path, "a") as f:
                        f.write(
                            json.dumps(dict(event, part=self.part, case=case))
                            + "\n"
                        )
            proc.wait()
            timer.cancel()
            self.current = None
        _trim(err_path, 16384)
        rc = proc.returncode
        summary = _summarize(case, spec, events, rc, time.time() - start)
        summary["timed_out"] = timer.fired
        summary["launch_blocking"] = (env or {}).get("CUDA_LAUNCH_BLOCKING", "")
        if summary["status"] != "ok":
            summary["stderr_tail"] = _error_lines(err_path)
            summary["stdout_tail"] = other[-12:]
        self._pack_dots(case)
        self._finish(summary)
        return summary

    def run_rows(self, case, flags, env=None, timeout=600):
        """rows.py in a child; returns its summary."""
        budget = min(timeout, self.left() - 10)
        if budget < 30:
            summary = dict(case=case, status="not run", reason="stage deadline")
            self._finish(summary)
            return summary
        folder = os.path.join(OUT, f"rows-{case}")
        os.makedirs(folder, exist_ok=True)
        child_env = dict(os.environ)
        child_env.update(env or {})
        child_env["KSTAGE_OUT"] = folder
        child_env["KSTAGE_DEADLINE"] = str(int(time.time() + budget - 15))
        log = os.path.join(OUT, f"rows-{case}.txt")
        cmd = [sys.executable, "-m", "benchmarks.quantized_matmul.pod.rows", "run"]
        cmd += flags + [f"--out={os.path.join(folder, 'rows.jsonl')}"]
        start = time.time()
        with open(log, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
            self.current = proc
            timer = _Timer(proc, budget)
            proc.wait()
            timer.cancel()
            self.current = None
        text = open(log, errors="replace").read()
        cells = [
            line
            for line in text.splitlines()
            if re.match(r"^(int4_g128|int4_pc|gptq|gptq_act|awq) ", line)
        ]
        crashed = bool(CUDA_ERROR.search(text)) and "Traceback" in text
        summary = dict(
            case=case,
            flow="rows.py",
            flags=flags,
            rc=proc.returncode,
            seconds=round(time.time() - start, 1),
            status="cuda_error" if crashed else ("ok" if proc.returncode in (0, 1) else "error"),
            crashed=crashed,
            cells_printed=len(cells),
            last_cells=cells[-4:],
            timed_out=timer.fired,
            stopped=[l for l in text.splitlines() if l.startswith("stopped:")][-1:],
            launch_blocking=(env or {}).get("CUDA_LAUNCH_BLOCKING", ""),
        )
        if crashed or summary["status"] == "error":
            summary["error_lines"] = _error_lines(log)
        _trim(log, 60000)
        # rows_summary.json and the rest of the folder stay; the jsonl
        # holds the records.
        self._finish(summary)
        return summary

    def _pack_dots(self, case):
        paths = sorted(glob.glob(os.path.join(OUT, f"dot-{case}-*.dot")))
        if not paths:
            return
        graphs = {}
        for path in paths:
            text = open(path, errors="replace").read()
            name = os.path.basename(path)[:-4]
            graphs[name] = dict(
                bytes=len(text),
                kernels=_dot_kernels(text),
                text=text[:200000],
            )
            os.remove(path)
        with open(os.path.join(OUT, f"dot-{case}.json"), "w") as f:
            json.dump(graphs, f)

    def _finish(self, summary):
        summary["part"] = self.part
        self.summaries.append(summary)
        with open(os.path.join(OUT, f"summary-{self.part}.json"), "w") as f:
            json.dump(self.summaries, f, indent=1, default=str)
        print("KPROF " + json.dumps(_short(summary), default=str), flush=True)

    def crashed(self):
        return [
            s
            for s in self.summaries
            if s.get("status") in ("cuda_error", "crashed")
        ]


class _Timer:
    """Kills the process group of `proc` after `seconds`."""

    def __init__(self, proc, seconds):
        import threading

        self.fired = False
        self.timer = threading.Timer(seconds, self._kill, args=(proc,))
        self.timer.daemon = True
        self.timer.start()

    def _kill(self, proc):
        self.fired = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(10)
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def cancel(self):
        self.timer.cancel()


def _trim(path, size):
    try:
        data = open(path, "rb").read()
    except OSError:
        return
    if len(data) > size:
        with open(path, "wb") as f:
            f.write(b"[...]\n" + data[-size:])


def _error_lines(path):
    """The lines of a log that name an error, and the last lines."""
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return []
    keep = [
        line
        for line in lines
        if CUDA_ERROR.search(line)
        or "Error" in line
        or "Returning" in line
        or "Xid" in line
    ]
    return (keep[:12] + ["--- last lines:"] + lines[-12:])[:40]


def _dot_kernels(text):
    """The kernel names of a cudaGraphDebugDotPrint file, with counts."""
    counts = {}
    for name in re.findall(
        r"(_Z[A-Za-z0-9_]{6,}|_w4a16_kernel|_splitk_sum|"
        r"[A-Za-z0-9_]*(?:gemm|Kernel|kernel|cutlass|xmma)[A-Za-z0-9_]*)",
        text,
    ):
        counts[name[:120]] = counts.get(name[:120], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:40])


def _summarize(case, spec, events, rc, seconds):
    """One summary per child: the status, the checks, the error."""
    end = [e for e in events if e.get("ev") == "end"]
    status = end[-1]["status"] if end else "crashed"
    errors = [e for e in events if e.get("ev") == "error"]
    if not end:
        # No end event: the process died (a signal, or a hang that the
        # timer killed).
        status = "cuda_error" if rc == 3 else "crashed"
    if errors and errors[-1].get("cuda"):
        status = "cuda_error"
    begins = [e for e in events if e.get("ev") == "begin"]
    checks = [
        {
            k: e.get(k)
            for k in (
                "cand",
                "phase",
                "m",
                "err",
                "err_all",
                "finite",
                "layers_equal",
                "same_as_eager",
                "differ_at",
                "deterministic",
                "bound",
                "floor",
                "checks_invalid",
                "replays",
                "route_counts",
            )
            if e.get(k) is not None
        }
        for e in events
        if e.get("ev") == "check"
    ]
    graphs = [
        {k: e.get(k) for k in ("cand", "m", "captured", "skipped") if e.get(k)}
        for e in events
        if e.get("ev") == "graph"
    ]
    for g in graphs:
        if "skipped" in g:
            g["skipped"] = g["skipped"].replace("\x1b[1m", "").replace("\x1b[0m", "")
            g["skipped"] = re.sub(r"\s+", " ", g["skipped"])[:200]
    f0 = [e for e in events if e.get("ev") == "err_f0"]
    tritons = [e for e in events if e.get("ev") == "triton"]
    summary = dict(
        case=case,
        flow=spec.get("flow"),
        mode=spec.get("mode"),
        dtype=spec.get("dtype"),
        shape=spec.get("shape"),
        ms=spec.get("ms"),
        candidates=spec.get("candidates"),
        skip_capture=spec.get("skip_capture"),
        rc=rc,
        seconds=round(seconds, 1),
        status=status,
        checks=checks,
        graphs=graphs,
        err_f0=[dict(err=e["err"], err_all=e["err_all"]) for e in f0],
        triton=tritons,
        done_ms=[e.get("m") for e in events if e.get("ev") == "m_done"],
        done_subops=[e.get("cand") for e in events if e.get("ev") == "subop_done"],
    )
    if errors:
        last = errors[-1]
        summary["error"] = {
            k: last.get(k)
            for k in ("where", "cand", "m", "i", "s", "layer", "error", "cuda")
            if last.get(k) is not None
        }
    if status != "ok" and begins:
        summary["last_begin"] = {
            k: begins[-1].get(k)
            for k in ("where", "cand", "m", "i", "s", "layer")
            if begins[-1].get(k) is not None
        }
    _gates(summary)
    return summary


def _gates(summary):
    """G1 and G2 of each check, against err(F0) of the same M."""
    f0 = {}
    for c in summary["checks"]:
        if c.get("cand") in ("F0", "F0-layer") and c.get("phase") == "eager":
            f0.setdefault(c.get("m"), c.get("err"))
    for e, m in zip(summary["err_f0"], summary["done_ms"] or [None]):
        f0.setdefault(m, e["err"])
    for c in summary["checks"]:
        err, bound, floor = c.get("err"), c.get("bound"), c.get("floor")
        if err is None or bound is None or c.get("checks_invalid"):
            continue
        c["G2"] = err <= bound
        name = str(c.get("cand"))
        if name.startswith("float"):
            continue
        ref = f0.get(c.get("m"))
        if ref is not None:
            c["G1_bound"] = 1.25 * ref + floor
            c["G1"] = err <= c["G1_bound"]


def _short(summary):
    """The KPROF line of a summary."""
    keep = (
        "part",
        "case",
        "status",
        "rc",
        "seconds",
        "timed_out",
        "launch_blocking",
        "error",
        "last_begin",
        "crashed",
        "cells_printed",
        "last_cells",
        "reason",
    )
    line = {k: summary[k] for k in keep if summary.get(k) not in (None, [], "")}
    if "error" in line:
        line["error"] = dict(line["error"])
        line["error"]["error"] = _first_line(line["error"].get("error", ""))
    line["checks"] = [
        "{cand} {phase} m={m} err={err:.3g} G1={G1} G2={G2}{same}".format(
            cand=c.get("cand"),
            phase=c.get("phase"),
            m=c.get("m"),
            err=c.get("err") if c.get("err") is not None else float("nan"),
            G1=c.get("G1", "-"),
            G2=c.get("G2", "-"),
            same=(
                ""
                if c.get("same_as_eager") is None
                else f" same={c.get('same_as_eager')}"
            ),
        )
        for c in summary.get("checks", [])
    ][:16]
    skipped = [
        f"{g.get('cand')} m={g.get('m')}: {g['skipped'][:80]}"
        for g in summary.get("graphs", [])
        if g.get("skipped")
    ]
    if skipped:
        line["graph_skipped"] = skipped[:4]
    for t in summary.get("triton", [])[:2]:
        line.setdefault("triton", []).append(
            dict(
                grid=t.get("grid"),
                c={
                    k: t.get("constexprs", {}).get(k)
                    for k in ("BM", "BNH", "BK", "SPLIT_K", "CONTIGUOUS")
                },
                warps=t.get("num_warps"),
                stages=t.get("num_stages"),
                regs=t.get("n_regs"),
                spills=t.get("n_spills"),
                shared=t.get("shared"),
            )
        )
    return line


def _first_line(text):
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""


def _cell_spec(case, flow, candidates, **kwargs):
    spec = dict(
        case=case,
        flow=flow,
        mode=CELL["mode"],
        dtype=CELL["dtype"],
        shape=CELL["shape"],
        ms=[CELL["m"]],
        candidates=list(candidates),
        replays=20,
        l2_mb=72,
    )
    spec.update(kwargs)
    return spec


def _small(spec):
    """The case on a host with no CUDA: a small shape, M <= 65."""
    spec = dict(spec)
    k, n = (int(d) for d in spec["shape"].split("x"))
    spec["shape"] = "512x256" if k > n else "256x512"
    spec["ms"] = sorted({min(m, 65) for m in spec["ms"]})
    spec["layers"] = 2
    spec["replays"] = 2
    return spec


BLOCKING = {"CUDA_LAUNCH_BLOCKING": "1"}


def part_a(d):
    small = (lambda s: s) if d.gpu else _small
    for name in CANDIDATES:
        spec = small(
            _cell_spec(
                f"a-{name}",
                "alone",
                [name],
                dump=True,
                err_f0=name in ("n5", "triton"),
            )
        )
        d.run(spec, BLOCKING, timeout=180)
    subops = small(_cell_spec("a-subops", "subops", [], dump=True))
    result = d.run(subops, BLOCKING, timeout=180)
    if result["status"] != "ok":
        names = [
            "float-layer",
            "F0-dequant",
            "F0-matmul",
            "F0-layer",
            "triton-kernel",
            "triton-layer",
        ]
        # The part that crashed runs again alone: does it crash alone?
        done = set(result.get("done_subops") or [])
        for name in names:
            if name in done:
                continue
            spec = dict(subops, case=f"a-subops-{name}", only=[name])
            d.run(spec, BLOCKING, timeout=120)
    for name in ("float", "F0", "triton"):
        found = [s for s in d.crashed() if s["case"] == f"a-{name}"]
        if found:
            spec = small(_cell_spec(f"a-perlayer-{name}", "per_layer", [name]))
            d.run(spec, BLOCKING, timeout=180)


def part_b(d):
    small = (lambda s: s) if d.gpu else _small
    s1r_spec = small(
        _cell_spec("b-s1r", "s1r", CANDIDATES, ms=[64, CELL["m"]])
    )
    first = d.run(s1r_spec, timeout=300)
    rows_flags = [
        "--route=triton",
        "--timing=stream,graph",
        "--modes=int4_g128,gptq",
        "--dtypes=float16",
        "--shapes=14336x4096" if d.gpu else "--shapes=256x512",
        "--m=64,256" if d.gpu else "--m=65",
        "--l2_mb=72",
    ]
    if not d.gpu:
        rows_flags += ["--warmup=2", "--calls=5", "--copy_mb=64"]
    harness = d.run_rows("b-rows", rows_flags, timeout=420)
    if first["status"] in ("cuda_error", "crashed") or harness.get("crashed"):
        variants = [
            ("b-s1r-blocking", {}, BLOCKING),
            ("b-s1r-no-F0-graph", dict(skip_capture=["F0"]), {}),
            ("b-s1r-no-n5", dict(candidates=["float", "F0", "triton"]), {}),
            ("b-s1r-no-triton-graph", dict(skip_capture=["triton"]), {}),
            ("b-s1r-no-float-graph", dict(skip_capture=["float"]), {}),
        ]
        for case, change, env in variants:
            spec = dict(s1r_spec, case=case, **change)
            d.run(spec, env, timeout=300)


def part_c(d):
    small = (lambda s: s) if d.gpu else _small
    modes = ("gptq", "gptq_act", "awq")
    for shape in ("4096x14336", "14336x4096"):
        for dtype in ("float16", "bfloat16"):
            for mode in modes:
                spec = small(
                    dict(
                        case=f"c-{mode}-{dtype}-{shape}",
                        flow="alone",
                        mode=mode,
                        dtype=dtype,
                        shape=shape,
                        ms=[65, 128, 256],
                        candidates=["triton"],
                        replays=20,
                        err_f0=True,
                        l2_mb=72,
                    )
                )
                result = d.run(spec, BLOCKING, timeout=150)
                if result["status"] == "ok":
                    continue
                done = set(result.get("done_ms") or [])
                for m in spec["ms"]:
                    if m in done:
                        continue
                    retry = dict(spec, case=f"{spec['case']}-m{m}", ms=[m])
                    d.run(retry, BLOCKING, timeout=120)
            if not d.gpu:
                return  # The dry run: one shape and dtype.


def _sanitizer():
    """The path of compute-sanitizer, and the places searched."""
    places = ["PATH"]
    found = shutil.which("compute-sanitizer")
    patterns = [
        "/usr/local/cuda/bin/compute-sanitizer",
        "/usr/local/cuda*/bin/compute-sanitizer",
        "/usr/local/cuda*/compute-sanitizer/compute-sanitizer",
        "/opt/nvidia/*/compute-sanitizer",
        "/usr/bin/compute-sanitizer",
    ]
    site = os.path.dirname(os.path.dirname(np.__file__))
    patterns.append(
        os.path.join(site, "nvidia", "*", "compute-sanitizer", "compute-sanitizer")
    )
    patterns.append(os.path.join(site, "nvidia", "*", "bin", "compute-sanitizer"))
    for pattern in patterns:
        places.append(pattern)
        if found:
            break
        matches = sorted(glob.glob(pattern))
        if matches:
            found = matches[0]
    return found, places


def part_d(d):
    tool, places = _sanitizer()
    if not tool:
        summary = dict(
            case="d-sanitizer",
            status="not run",
            reason="compute-sanitizer is not in the image",
            searched=places,
        )
        d._finish(summary)
        return
    targets = _earlier_crashes()
    if not targets:
        targets = [
            _cell_spec("d-triton", "alone", ["triton"], replays=3),
            _cell_spec("d-F0", "alone", ["F0"], replays=3),
        ]
    for spec in targets[:2]:
        spec = dict(
            spec,
            case="d-" + spec["case"],
            replays=3,
            samples=1,
            calls=3,
            dump=False,
            out=OUT,
        )
        log = os.path.join(OUT, f"sanitizer-{spec['case']}.txt")
        budget = min(420, d.left() - 15)
        if budget < 60:
            d._finish(dict(case=spec["case"], status="not run", reason="stage deadline"))
            continue
        cmd = [
            tool,
            "--tool",
            "memcheck",
            "--target-processes",
            "all",
            "--print-limit",
            "100",
            "--leak-check",
            "no",
            sys.executable,
            HERE,
            "child",
            json.dumps(spec),
        ]
        start = time.time()
        with open(log, "w") as f:
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
            )
            d.current = proc
            timer = _Timer(proc, budget)
            proc.wait()
            timer.cancel()
            d.current = None
        text = open(log, errors="replace").read()
        lines = [
            l
            for l in text.splitlines()
            if l.startswith("=========") or "EVENT" in l and '"error"' in l
        ]
        summary = dict(
            case=spec["case"],
            tool=tool,
            rc=proc.returncode,
            seconds=round(time.time() - start, 1),
            timed_out=timer.fired,
            status="ok" if proc.returncode == 0 else "sanitizer_report",
            sanitizer_lines=[l[:300] for l in lines[:60]],
            summary_line=[l for l in text.splitlines() if "ERROR SUMMARY" in l][-1:],
        )
        _trim(log, 60000)
        d._finish(summary)


def _earlier_crashes():
    """The specs of the cases that crashed in the stages before."""
    found = []
    root = os.environ.get("KOUT")
    paths = []
    if root:
        paths = sorted(glob.glob(os.path.join(root, "stages", "*", "summary-*.json")))
    paths += sorted(glob.glob(os.path.join(OUT, "summary-*.json")))
    for path in paths:
        try:
            for s in json.load(open(path)):
                if s.get("status") in ("cuda_error", "crashed") and s.get("flow") in (
                    "alone",
                    "subops",
                    "per_layer",
                    "s1r",
                ):
                    found.append(
                        {
                            k: s[k]
                            for k in ("case", "flow", "mode", "dtype", "shape", "ms")
                        }
                        | dict(candidates=s.get("candidates") or [], l2_mb=72)
                    )
        except (OSError, ValueError, KeyError):
            continue
    return found


def part_e(d):
    crashes = _earlier_crashes()
    if crashes:
        d._finish(
            dict(
                case="e-s1r-stage",
                status="not run",
                reason=f"the crash occurred in {crashes[0]['case']}",
            )
        )
        return
    if not d.gpu:
        d._finish(dict(case="e-s1r-stage", status="not run", reason="no CUDA"))
        return
    d.run_rows("e-s1r-stage", S1R_PREFILL, timeout=3600)


def env_report():
    """The host, the device and the tools."""
    import platform

    print("python", sys.version.split()[0], platform.platform())
    print("KMODE", os.environ.get("KMODE"), "TRITON_INTERPRET", os.environ.get("TRITON_INTERPRET"))
    try:
        import torch

        print("torch", torch.__version__, "cuda", torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(
                "device",
                p.name,
                f"cc={p.major}.{p.minor}",
                f"sms={p.multi_processor_count}",
                f"memory_mb={p.total_memory // 2**20}",
                f"l2_mb={getattr(p, 'L2_cache_size', 0) / 2**20:.0f}",
            )
    except ImportError as error:
        print("torch: not importable:", error)
    try:
        import triton

        print("triton", triton.__version__)
    except ImportError:
        print("triton: not importable")
    tool, places = _sanitizer()
    print("compute-sanitizer:", tool or "not found", "(searched: " + ", ".join(places) + ")")
    for name in ("nvcc", "cuda-gdb", "nsys", "ncu"):
        print(f"{name}:", shutil.which(name) or "not found")
    if shutil.which("nvidia-smi"):
        query = (
            "name,driver_version,pstate,clocks.sm,clocks.max.sm,"
            "temperature.gpu,power.draw,ecc.mode.current,"
            "pcie.link.gen.current,pcie.link.width.current"
        )
        for args in (
            ["--query-gpu=" + query, "--format=csv"],
            ["-q", "-d", "PAGE_RETIREMENT,ROW_REMAPPER,ECC"],
        ):
            try:
                out = subprocess.run(
                    ["nvidia-smi"] + args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
            except (OSError, subprocess.TimeoutExpired) as error:
                out = f"nvidia-smi failed: {error}"
            lines = [l for l in out.splitlines() if l.strip()]
            print("\n".join(lines[:40]))
    else:
        print("nvidia-smi: not found")
    print("probe env: done")
    return 0


PARTS = dict(a=part_a, b=part_b, c=part_c, d=part_d, e=part_e)


def main(argv):
    if argv and argv[0] == "child":
        child(json.loads(argv[1]))
        return 0
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["drive", "env"])
    p.add_argument("--part", default="a")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print("ignored flags:", " ".join(unknown), flush=True)
    if args.cmd == "env":
        return env_report()
    os.makedirs(OUT, exist_ok=True)
    deadline = float(os.environ.get("KSTAGE_DEADLINE") or 0)
    if deadline:
        deadline -= 15
    d = Driver(args.part, deadline)
    print(f"part {args.part}: gpu={d.gpu} deadline in {d.left():.0f} s", flush=True)
    try:
        PARTS[args.part](d)
    except Exception:
        traceback.print_exc()
        return 2
    crashed = d.crashed()
    print(
        f"part {args.part}: {len(d.summaries)} cases, "
        f"{len(crashed)} with a CUDA error: "
        + ", ".join(s["case"] for s in crashed),
        flush=True,
    )
    for s in d.summaries:
        print(json.dumps(_short(s), default=str)[:3000], flush=True)
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
