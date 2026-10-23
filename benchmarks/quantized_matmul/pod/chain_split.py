"""Chain A/B of the Mosaic GPU K1 frontend, and the per-kernel split of an
nsys trace. Probe for the review of 2026-10-22 (not part of the PR).

Run from the Keras tree (place this file at
benchmarks/quantized_matmul/pod/chain_split.py), one backend per process:

    KERAS_BACKEND=jax python -m benchmarks.quantized_matmul.pod.chain_split \
        run --mode=int4_g128 --shape=4096x4096 --m=1 \
        --candidates=float,shipped,noguard,pre [--structure=parallel] \
        [--l2_mb=50] [--layers=0] [--calls=30] [--warmup=5] [--gap_ms=0] \
        [--interpret] --out=FILE.jsonl

    python benchmarks/quantized_matmul/pod/chain_split.py report \
        TRACE.sqlite --meta=FILE.jsonl [--gap_us=1000] > split.json

Candidates (each one jit of the whole chain, as bench.py `--levels=chain`):
- `float`: bfloat16 `Dense` layers of the same shape.
- `shipped`: the `mosaic_gpu` route as it is (group guard `lax.cond`, rows
  built per call).
- `noguard`: the same route without the group guard (`contiguous_groups`
  False). Valid here: the bench weights have `g_idx = k // G`.
- `pre`: `noguard`, and the `rows` input precomputed once per layer, as a
  non-trainable variable that the jit gets as an argument.

`--structure=parallel` is bench.py's chain (every layer reads the same x).
`sequential` feeds each layer the output of the previous one: square
shapes chain directly, `KxN` with K != N alternates `KxN` and `NxK`.

Each record holds two timings: `sync_us` (one call, then
`block_until_ready`, as bench.py: min and median of `--calls`) and
`stream_us` (`--calls` calls back to back, one sync at the end, per call).
`sync - stream` is the host dispatch and sync latency of one call.
`--gap_ms` sleeps between the timed calls so that an nsys trace separates
them (use it only under nsys: it lets the GPU clocks drop).
"""

import argparse
import dataclasses
import json
import math
import sqlite3
import statistics
import sys
import time


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from benchmarks.quantized_matmul import bench
    from keras.src import ops
    from keras.src.backend.jax.ops import quantized
    from keras.src.quantizers import kernels

    k, n = (int(v) for v in args.shape.split("x"))
    dtype = "bfloat16"
    base = "mosaic_gpu_interpret" if args.interpret else "mosaic_gpu"
    capability = (
        "8.9" if args.interpret else jax.devices()[0].compute_capability
    )
    config = quantized.MOSAIC_GPU_CONFIGS[
        (capability, dtype, kernels._m_bucket(args.m))
    ]

    def layer_shapes(count):
        if args.structure == "parallel" or k == n:
            return [(k, n)] * count
        return [(k, n) if i % 2 == 0 else (n, k) for i in range(count)]

    first = bench._dense(args.mode, n, k, dtype, 1)
    count = args.layers or min(
        max(8, math.ceil(4 * args.l2_mb * 2**20 / bench._nbytes(first.variables))),
        64,
    )
    if args.structure == "sequential" and k != n:
        count += count % 2

    def frontend_pre(x, codes, rows, *, signed, group_size, input_scales):
        """`mosaic_gpu_w4a16_matmul` with `rows` given (quantized.py:94-146
        without lines 107-124)."""
        block_m, block_n, stages, min_blocks = config
        if input_scales is not None:
            x = x / input_scales.astype(x.dtype)
        m, kk = x.shape
        half = codes.shape[1]
        while half % block_n:
            block_n //= 2
        padded = -(-m // block_m) * block_m
        blocks = padded // block_m * (half // block_n)
        split = quantized._split_k(kk // quantized._BK, blocks, min_blocks)
        xor = 8 if signed else 0
        planes = quantized._w4a16_kernel(
            jnp.pad(x, ((0, padded - m), (0, 0))),
            codes,
            rows,
            xor=xor,
            per_group=group_size // quantized._BK,
            block_m=block_m,
            block_n=block_n,
            stages=stages,
            split=split,
            interpret=args.interpret,
        )
        sums = []
        for plane in planes:
            total = plane[0, :m]
            for i in range(1, split):
                total = total + plane[i, :m]
            sums.append(total)
        y = jnp.stack(sums, axis=-1).reshape(m, 2 * half)
        return y.astype(x.dtype)

    def build_rows(view):
        """The `rows` of quantized.py:112-124, built once."""
        block_n = config[1]
        half = view.codes.shape[1]
        while half % block_n:
            block_n //= 2
        xor = 8 if str(view.codes.dtype) == "int8" else 0
        scale = jnp.asarray(view.scale).astype("float32")
        zero = jnp.asarray(view.zero_point).astype("float32") + xor
        shape = (scale.shape[0], half // block_n, block_n)
        return jnp.stack(
            [r[:, i::2].reshape(shape) for r in (scale, zero) for i in (0, 1)],
            axis=2,
        )

    def pre_fn(layer, geometry, view, x, _):
        rows = layer._probe_rows.value
        return frontend_pre(
            x,
            view.codes,
            rows,
            signed=str(view.codes.dtype) == "int8",
            group_size=min(view.scheme.group_size, x.shape[1]),
            input_scales=view.input_scales,
        )

    source = kernels._ROUTES[base]
    with_config = dataclasses.replace(
        source,
        fn=lambda layer, geometry, view, x, _: source.fn(
            layer, geometry, view, x, config
        ),
        version=None,
    )
    kernels._ROUTES["probe_shipped"] = with_config
    kernels._ROUTES["probe_noguard"] = dataclasses.replace(
        with_config, contiguous_groups=False
    )
    kernels._ROUTES["probe_pre"] = dataclasses.replace(
        with_config, fn=pre_fn, contiguous_groups=False
    )

    def chain_fn(parts, x):
        if args.structure == "parallel":
            return [part(x) for part in parts]
        y = x
        for part in parts:
            y = part(y)
        return y

    rng = np.random.default_rng(args.m)
    x = ops.cast(
        ops.convert_to_tensor(rng.standard_normal((args.m, k)).astype("f4")),
        dtype,
    )
    records = []
    # The first route candidate's output; the others report their largest
    # difference from it (`max_diff_vs_first_route`, 0 when exact).
    outputs = {}
    for name in args.candidates:
        mode = "float" if name == "float" else args.mode
        layers = [
            bench._dense(mode, nn, kk, dtype, s + 1)
            for s, (kk, nn) in enumerate(layer_shapes(count))
        ]
        if name == "pre":
            if layers[0]._quantized_weight().zero_point is None:
                print(f"skip pre: {args.mode} has no rows", flush=True)
                continue
            for layer in layers:
                rows = build_rows(layer._quantized_weight())
                # A built layer locks its state; the probe adds one buffer.
                layer._tracker.unlock()
                layer._probe_rows = layer.add_weight(
                    name="probe_rows",
                    shape=rows.shape,
                    initializer="zeros",
                    dtype="float32",
                    trainable=False,
                )
                layer._probe_rows.assign(rows)
                layer._tracker.lock()
        model = bench.Step(layers, chain_fn)
        fn, fargs = bench._workload(model, x)
        ctx = (
            kernels.policy("off")
            if name == "float"
            else kernels.force_route("probe_" + name)
        )
        before = dict(kernels.route_counts)
        with ctx:
            compiled = jax.jit(fn).lower(*fargs).compile()
        counts = {
            key: value - before.get(key, 0)
            for key, value in kernels.route_counts.items()
            if value != before.get(key, 0)
        }
        text = compiled.as_text()
        for _ in range(args.warmup):
            jax.block_until_ready(compiled(*fargs))
        y = compiled(*fargs)
        y = np.asarray(
            jnp.concatenate([jnp.ravel(t) for t in jax.tree.leaves(y)]),
            dtype="float32",
        )
        if name != "float":
            outputs.setdefault("reference", y)
        diff = (
            None
            if name == "float"
            else float(np.max(np.abs(y - outputs["reference"])))
        )
        sync = []
        for _ in range(args.calls):
            if args.gap_ms:
                time.sleep(args.gap_ms / 1e3)
            start = time.perf_counter()
            jax.block_until_ready(compiled(*fargs))
            sync.append((time.perf_counter() - start) * 1e6)
        start = time.perf_counter()
        out = None
        for _ in range(args.calls):
            out = compiled(*fargs)
        jax.block_until_ready(out)
        stream = (time.perf_counter() - start) * 1e6 / args.calls
        weight_bytes = sum(
            bench._nbytes(layer.variables) for layer in layers
        )
        record = dict(
            candidate=name,
            mode=mode,
            shape=args.shape,
            structure=args.structure,
            m=args.m,
            layers=count,
            config=list(config),
            route_counts=counts,
            max_diff_vs_first_route=diff,
            sync_min_us=min(sync),
            sync_median_us=statistics.median(sync),
            stream_us=stream,
            per_layer_sync_min_us=min(sync) / count,
            per_layer_stream_us=stream / count,
            weight_bytes=weight_bytes,
            stream_gbps=weight_bytes / stream / 1e3,
            conditionals=text.count(" conditional("),
            custom_calls=text.count('custom_call_target="mosaic_gpu_v2"'),
            temp_bytes=getattr(
                compiled.memory_analysis(), "temp_size_in_bytes", None
            ),
            device=str(jax.devices()[0]),
            jax=jax.__version__,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        del layers, model, compiled
    if args.out:
        with open(args.out, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


KIND_DTOH = 2


def classify(name):
    low = name.lower()
    if "mosaic" in low:
        return "mosaic"
    if any(s in low for s in ("gemm", "gemv", "cutlass", "xmma", "sm90_", "cublas", "triton")):
        return "gemm"
    return "other"


def report(args):
    """Per timed call of one candidate process: the span of its kernels,
    their busy time (union), the idle gaps, the time per kernel class, and
    the host API calls that sync or launch."""
    db = sqlite3.connect(args.trace)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = [
        (s, e, strings.get(name, str(name)))
        for s, e, name in db.execute(
            "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "ORDER BY start"
        )
    ]
    memcpy = list(
        db.execute(
            "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    )
    try:
        api = [
            (s, e, strings.get(name, str(name)))
            for s, e, name in db.execute(
                "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        ]
    except sqlite3.OperationalError:
        api = []
    bursts, current = [], []
    for item in kernels:
        if current and item[0] - current[-1][1] > args.gap_us * 1e3:
            bursts.append(current)
            current = []
        current.append(item)
    if current:
        bursts.append(current)
    meta = None
    if args.meta:
        with open(args.meta) as f:
            meta = [json.loads(line) for line in f][-1]
    calls = args.calls or (meta and 0) or 0
    timed = bursts[-calls:] if calls else bursts[-10:]
    layers = meta["layers"] if meta else 1
    rows = []
    for burst in timed:
        lo, hi = burst[0][0], max(e for _, e, _ in burst)
        busy, edge = 0, lo
        for s, e, _ in burst:
            if e > edge:
                busy += e - max(s, edge)
                edge = e
        per_class = {}
        names = {}
        for s, e, name in burst:
            c = classify(name)
            per_class[c] = per_class.get(c, 0) + (e - s)
            names[name] = names.get(name, 0) + 1
        # A host-blocking `conditional` copies its predicate (a few bytes)
        # to the host once per layer.
        small_dtoh = sum(
            1 for s, e, b, kind in memcpy
            if kind == KIND_DTOH and b <= 16 and lo <= s <= hi + 1e6
        )
        calls_in = {}
        for s, e, name in api:
            if not lo - 2e6 <= s <= hi + 1e6:
                continue
            low = name.lower()
            for key in ("GraphLaunch", "Launch", "Synchronize", "MemcpyDtoH",
                        "TensorMap", "EventRecord"):
                if key.lower() in low:
                    calls_in[key] = calls_in.get(key, 0) + 1
                    break
        rows.append(dict(
            span_us=(hi - lo) / 1e3,
            busy_us=busy / 1e3,
            idle_us=(hi - lo - busy) / 1e3,
            kernels=len(burst),
            per_class_us={c: v / 1e3 for c, v in per_class.items()},
            small_dtoh=small_dtoh,
            api=calls_in,
            names=dict(sorted(names.items(), key=lambda i: -i[1])[:12]),
        ))
    best = min(rows, key=lambda r: r["span_us"]) if rows else None
    out = dict(
        trace=args.trace,
        meta=meta,
        bursts=len(bursts),
        timed=len(rows),
        best=best,
        per_layer_best=best and dict(
            span_us=best["span_us"] / layers,
            busy_us=best["busy_us"] / layers,
            idle_us=best["idle_us"] / layers,
            kernels=best["kernels"] / layers,
            per_class_us={c: v / layers for c, v in best["per_class_us"].items()},
            small_dtoh=best["small_dtoh"] / layers,
        ),
        median_span_us=statistics.median(r["span_us"] for r in rows) if rows else None,
    )
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", default="int4_g128")
    r.add_argument("--shape", default="4096x4096")
    r.add_argument("--m", type=int, default=1)
    r.add_argument("--candidates", type=lambda s: s.split(","),
                   default=["float", "shipped", "noguard", "pre"])
    r.add_argument("--structure", choices=["parallel", "sequential"],
                   default="parallel")
    r.add_argument("--l2_mb", type=float, default=50)
    r.add_argument("--layers", type=int, default=0)
    r.add_argument("--calls", type=int, default=30)
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--gap_ms", type=float, default=0)
    r.add_argument("--interpret", action="store_true")
    r.add_argument("--out")
    q = sub.add_parser("report")
    q.add_argument("trace")
    q.add_argument("--meta")
    q.add_argument("--calls", type=int, default=0)
    q.add_argument("--gap_us", type=float, default=1000)
    args = p.parse_args()
    (run if args.cmd == "run" else report)(args)


if __name__ == "__main__":
    sys.exit(main())
