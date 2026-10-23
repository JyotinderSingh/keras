"""The temp memory and time of an F0 chain under `jax.jit`, with and
without an optimization barrier in the reference path. Probe for V2
(question Q6; not part of the PR).

F0 is the reference path (`kernels._reference`: `dequantize`, then the
float contraction), the path of JAX `predict` when no route applies. The
red team of 2026-10-23 found that, under `jit`, XLA unpacks the codes of
every layer at the start of the program and keeps them live: on CPU, an
8-layer int4 chain of 1024 x 1024 reserves 32 MiB of temp memory and
takes 2.0 ms. With the barrier of `kernels._wait_for_inputs` (stack
branch 39) inside `_reference`, it reserves 4 MiB and takes 1.4 ms. This
probe measures the same two variants on a GPU.

Run from the Keras tree (this file is
benchmarks/quantized_matmul/pod/refbarrier.py), KERAS_BACKEND=jax:

    python -m benchmarks.quantized_matmul.pod.refbarrier \
        [--modes=int4_g128,int4_pc] [--shapes=1024x1024,4096x4096] \
        [--layers=8] [--m=1,16] [--dtype=bfloat16] [--calls=200] \
        [--warmup=10] [--interpret] [--out=FILE.jsonl]

Variants, each one `jit` of the whole chain with the variables as
arguments (as `bench.py` and `predict` pass them), under the policy
"off":
- `plain`: the reference path as it is.
- `barrier`: the reference path after one `jax.lax.optimization_barrier`
  over the inputs and the tensors of the weight (the code of
  `_wait_for_inputs`, written here so that the probe runs on every
  stack branch).

The chain is sequential: square shapes chain directly, `KxN` with
K != N alternates `KxN` and `NxK`. Each record holds `temp_bytes`
(`memory_analysis()`), the number of `optimization_barrier` ops in the
lowered program, `sync_min_us` and `sync_median_us` (one call, then
`block_until_ready`), `stream_us` (calls back to back, one sync, per
call), and `max_diff_vs_plain` (0.0 when the outputs are the same bits).
`--interpret` (CPU dry runs) only marks the records; the reference path
needs no kernel.
"""

import argparse
import json
import os
import statistics
import sys
import time


def run(args):
    import jax
    import numpy as np

    from benchmarks.quantized_matmul import bench
    from keras.src import ops
    from keras.src.quantizers import kernels

    original = kernels._reference

    def barrier_reference(
        layer, geometry, quantized_weight, inputs, config=None
    ):
        if isinstance(inputs, jax.core.Tracer):
            inputs, tensors = jax.lax.optimization_barrier(
                (inputs, quantized_weight.read_tensors())
            )
            quantized_weight = quantized_weight.with_tensors(tensors)
        return original(layer, geometry, quantized_weight, inputs, config)

    def chain_fn(parts, x):
        for part in parts:
            x = part(x)
        return x

    records = []
    cells = [
        (mode, shape, m)
        for mode in args.modes
        for shape in args.shapes
        for m in args.m
    ]
    for mode, shape, m in cells:
        k, n = (int(v) for v in shape.split("x"))
        shapes = [(k, n) if i % 2 == 0 else (n, k) for i in range(args.layers)]
        layers = [
            bench._dense(mode, nn, kk, args.dtype, s + 1)
            for s, (kk, nn) in enumerate(shapes)
        ]
        model = bench.Step(layers, chain_fn)
        rng = np.random.default_rng(m)
        x = ops.cast(
            ops.convert_to_tensor(rng.standard_normal((m, k)).astype("f4")),
            args.dtype,
        )
        outputs = {}
        for variant in ("plain", "barrier"):
            # A new function per variant: `jax.jit` of the same function
            # would reuse the trace of the first variant.
            fn, fargs = bench._workload(model, x)
            kernels._reference = (
                barrier_reference if variant == "barrier" else original
            )
            try:
                before = dict(kernels.route_counts)
                with kernels.policy("off"):
                    lowered = jax.jit(fn).lower(*fargs)
                counts = {
                    key: value - before.get(key, 0)
                    for key, value in kernels.route_counts.items()
                    if value != before.get(key, 0)
                }
            finally:
                kernels._reference = original
            barriers = lowered.as_text().count("optimization_barrier")
            compiled = lowered.compile()
            memory = compiled.memory_analysis()
            for _ in range(args.warmup):
                jax.block_until_ready(compiled(*fargs))
            y = np.asarray(compiled(*fargs), dtype="float32")
            outputs[variant] = y
            sync = []
            for _ in range(args.calls):
                start = time.perf_counter()
                jax.block_until_ready(compiled(*fargs))
                sync.append((time.perf_counter() - start) * 1e6)
            start = time.perf_counter()
            out = None
            for _ in range(args.calls):
                out = compiled(*fargs)
            jax.block_until_ready(out)
            stream = (time.perf_counter() - start) * 1e6 / args.calls
            record = dict(
                probe="refbarrier",
                variant=variant,
                mode=mode,
                shape=shape,
                m=m,
                dtype=args.dtype,
                layers=args.layers,
                route_counts=counts,
                barriers=barriers,
                temp_bytes=getattr(memory, "temp_size_in_bytes", None),
                argument_bytes=getattr(memory, "argument_size_in_bytes", None),
                sync_min_us=min(sync),
                sync_median_us=statistics.median(sync),
                stream_us=stream,
                per_layer_stream_us=stream / args.layers,
                max_diff_vs_plain=float(np.max(np.abs(y - outputs["plain"]))),
                finite=bool(np.all(np.isfinite(y))),
                interpret=args.interpret,
                device=str(jax.devices()[0]),
                jax=jax.__version__,
                keras_sha=os.environ.get("KERAS_SHA"),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
            del compiled, lowered
        del layers, model
    if args.out:
        with open(args.out, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modes", type=lambda s: s.split(","), default=["int4_g128", "int4_pc"]
    )
    p.add_argument(
        "--shapes",
        type=lambda s: s.split(","),
        default=["1024x1024", "4096x4096"],
    )
    p.add_argument("--layers", type=int, default=8)
    p.add_argument(
        "--m", type=lambda s: [int(v) for v in s.split(",")], default=[1]
    )
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--calls", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--out")
    # The dry run appends the grid flags of bench.py; ignore them.
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"refbarrier.py: ignored flags: {' '.join(unknown)}", flush=True)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
