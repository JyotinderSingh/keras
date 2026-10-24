"""The N5 CUDA-graph check of S1x job 1. Fork only; not part of a PR.

In S1r (commit d65f349fe), the N5 route of GPTQ and AWQ did not capture
in a CUDA graph: "Cannot copy between CPU and CUDA tensors during CUDA
graph capture". The cause was two Python number operands of branch 31
(`kernels.py`: the uint8 nibble mask and the `k // G` of the group
guard), which torch copies from the host on each call. The new 31 has no
such operand. The guard of GPTQ with several groups still reads its
predicate on the host (`ops.cond` -> a Python `if`), so the N5 call of
GPTQ G128 is expected to fail the capture still.

Run it from the Keras tree with the torch backend:

    PYTHONPATH=. python benchmarks/quantized_matmul/pod/probes/n5_graph_probe.py

Each case runs in its own child process (a failed capture cannot then
touch the next case): a `Dense` layer of 4096x4096 (K x N), M = 1,
bfloat16 compute, random stored tensors (bench.py `_fill`), the route
forced to `n5`:

- `awq`: `awq/4/128` (no `g_idx` in the view, no guard);
- `gptq_one_group`: `gptq/4/-1` (one group, no `g_idx`, no guard);
- `gptq_g128`: `gptq/4/128` (a `g_idx`, so the group guard runs);
- `int4_g128`: `int4/128`, the control (its N5 captured in S1r).

Per case: the route counts of one call (N5 must run, not the reference
path); the torch ops of one eager call that make a tensor from host data
(`aten.lift_fresh`), that read a device value on the host
(`aten._local_scalar_dense`, `aten.item`), and, on CUDA, the ops with a
CPU tensor operand; `err` against the float64 oracle (bench.py
`_dense_oracle` and `_err`) and the bound of rows.py (bfloat16 2**-7);
then the capture as rows.py `_capture` (`cuda_graph.capture` of the pod
folder: 3 calls on the warm-up stream, the cuBLAS workspaces cleared,
`torch.cuda.graph` on the capture stream, the current stream restored;
then one replay against an eager call, `first_replay_same`), 20 replays,
and whether each replay gives the bits of the eager call. On a host with
no CUDA (the dry run), the shape is 512x512 and there is no capture.

Output: one `N5GRAPH {json}` line per case, and `n5_graph.json` (all
cases) in $KSTAGE_OUT (else ./n5-graph-out). Exit code 0 when `awq` and
`gptq_one_group` run N5, pass the bound, make no host tensor and no host
read (`lift_fresh`, `host_reads`; the CPU operands are recorded only),
and (on CUDA) capture and replay with the bits of the eager call; else
1. The result of `gptq_g128` and `int4_g128` is recorded and gates
nothing.
"""

import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.abspath(__file__)
OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("n5-graph-out")
# Case -> (the mode of the dtype policy, the bench.py mode of the oracle).
CASES = {
    "awq": ("awq/4/128", "awq"),
    "gptq_one_group": ("gptq/4/-1", "gptq"),
    "gptq_g128": ("gptq/4/128", "gptq"),
    "int4_g128": ("int4/128", "int4_g128"),
}
GATED = ("awq", "gptq_one_group")
BOUND = 2**-7  # rows.py G2, bfloat16.
DTYPE = "bfloat16"
M = 1


def child(case):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    import keras
    from benchmarks.quantized_matmul import bench
    from keras.src.quantizers import kernels

    gpu = torch.cuda.is_available() and os.environ.get("KMODE") != "cpu"
    k, n = (4096, 4096) if gpu else (512, 512)
    spec, oracle_mode = CASES[case]
    result = dict(case=case, policy=f"{spec}_from_{DTYPE}", k=k, n=n, m=M)
    result["device"] = torch.cuda.get_device_name() if gpu else "cpu"
    result["torch"] = torch.__version__

    layer = keras.layers.Dense(n, use_bias=False, dtype=result["policy"])
    layer.build((None, k))
    view = layer._quantized_weight()
    group = view.scheme.group_size
    result["group_size"] = group
    result["view_has_g_idx"] = view.g_idx is not None
    # bench.py `_fill` writes `g_idx = k // 128`; a mode that fixes the
    # groups refuses another value, so give it the group of the scheme.
    saved = bench.GROUP
    bench.GROUP = min(group, k) if group else saved
    try:
        bench._fill(layer, oracle_mode, 0)
    finally:
        bench.GROUP = saved
    x_np, x = bench._inputs(M, k, DTYPE)
    oracle = bench._dense_oracle(layer, oracle_mode, x_np)

    def call():
        with kernels.force_route("n5"), torch.no_grad():
            return layer(x)

    def sync():
        if gpu:
            torch.cuda.synchronize()

    before = dict(kernels.route_counts)
    y = call()
    sync()
    result["route_counts"] = {
        name: count - before.get(name, 0)
        for name, count in kernels.route_counts.items()
        if count != before.get(name, 0)
    }

    counts = dict(lift_fresh=0, host_reads=0, cpu_operand_ops=0)
    names = {}

    def cpu_operand(args):
        for a in args:
            if isinstance(a, (list, tuple)):
                if cpu_operand(a):
                    return True
            elif isinstance(a, torch.Tensor) and a.device.type == "cpu":
                return True
        return False

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            key = None
            if func is torch.ops.aten.lift_fresh.default:
                key = "lift_fresh"
            elif func in (
                torch.ops.aten._local_scalar_dense.default,
                torch.ops.aten.item.default,
            ):
                key = "host_reads"
            elif gpu and cpu_operand(list(args) + list(kwargs.values())):
                key = "cpu_operand_ops"
            if key:
                counts[key] += 1
                label = f"{key}:{func}"
                names[label] = names.get(label, 0) + 1
            return func(*args, **kwargs)

    with Record():
        call()
    sync()
    result.update(counts)
    result["host_ops"] = names
    eager = y.detach().clone()
    result["err"] = bench._err([eager], oracle)
    result["bound"] = BOUND
    result["finite"] = bool(torch.isfinite(eager.float()).all())
    result["deterministic"] = bool(torch.equal(call(), eager))

    if not gpu:
        result["captured"] = None
        result["reason"] = "no CUDA"
    else:
        from benchmarks.quantized_matmul.pod import cuda_graph

        try:
            graph, outputs, first_same = cuda_graph.capture(call)
            result["captured"] = True
            result["first_replay_same"] = first_same
        except RuntimeError as error:
            torch.cuda.synchronize()
            result["captured"] = False
            result["reason"] = f"{type(error).__name__}: {error}"[:600]
        if result["captured"]:
            differ = []
            for i in range(20):
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(outputs, eager):
                    differ.append(i)
            result["replays"] = 20
            result["same_as_eager"] = not differ
            result["differ_at"] = differ[:10]
            result["graph_err"] = bench._err([outputs], oracle)
            # A capture can hide a host read; time one replay as a sanity
            # figure (one layer, no L2 rotation: not a row timing).
            start = time.perf_counter()
            for _ in range(50):
                graph.replay()
            torch.cuda.synchronize()
            result["replay_us"] = round(
                (time.perf_counter() - start) / 50 * 1e6, 2
            )
    return result


def gate(result):
    """The problems of a gated case; empty when it passes."""
    problems = []
    if result.get("route_counts") != {"n5": 1}:
        problems.append(f"route_counts {result.get('route_counts')}")
    if result.get("err") is None or not result["err"] <= BOUND:
        problems.append(f"err {result.get('err')} > bound {BOUND}")
    if not result.get("finite"):
        problems.append("not finite")
    # `cpu_operand_ops` is recorded only: on CUDA the capture decides.
    for key in ("lift_fresh", "host_reads"):
        if result.get(key):
            problems.append(f"{key} {result[key]}")
    if result.get("captured") is False:
        problems.append("capture failed")
    if result.get("captured") and not result.get("same_as_eager"):
        problems.append(f"replays differ at {result.get('differ_at')}")
    if result.get("first_replay_same") is False:
        problems.append("the first replay differs from an eager call")
    return problems


def main(argv):
    if len(argv) == 2 and argv[0] == "child":
        try:
            result = child(argv[1])
        except Exception as error:
            result = dict(
                case=argv[1],
                error=f"{type(error).__name__}: {error}"[:1500],
                trace=traceback.format_exc()[-2500:],
            )
        print("RESULT " + json.dumps(result, default=str), flush=True)
        sys.stdout.flush()
        os._exit(0)
    os.makedirs(OUT, exist_ok=True)
    results = []
    for case in CASES:
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, HERE, "child", case],
                capture_output=True,
                text=True,
                timeout=300,
            )
            lines = [
                line
                for line in proc.stdout.splitlines()
                if line.startswith("RESULT ")
            ]
            if lines:
                result = json.loads(lines[-1][len("RESULT ") :])
            else:
                result = dict(
                    case=case, error=f"no result, rc={proc.returncode}"
                )
            result["rc"] = proc.returncode
            if proc.returncode or "error" in result:
                result["stderr_tail"] = proc.stderr[-2000:]
        except subprocess.TimeoutExpired:
            result = dict(case=case, error="timeout 300 s")
        result["seconds"] = round(time.time() - start, 1)
        result["gated"] = case in GATED
        result["problems"] = gate(result) if case in GATED else []
        if "error" in result and case in GATED:
            result["problems"].append(result["error"][:300])
        results.append(result)
        short = {
            k: result.get(k)
            for k in (
                "case",
                "policy",
                "k",
                "n",
                "group_size",
                "view_has_g_idx",
                "route_counts",
                "lift_fresh",
                "host_reads",
                "cpu_operand_ops",
                "host_ops",
                "err",
                "bound",
                "captured",
                "reason",
                "same_as_eager",
                "graph_err",
                "replay_us",
                "error",
                "gated",
                "problems",
                "seconds",
            )
            if k in result
        }
        print("N5GRAPH " + json.dumps(short, default=str), flush=True)
        if "error" in result:
            print(result.get("trace", "") or result.get("stderr_tail", ""))
    with open(os.path.join(OUT, "n5_graph.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    failed = [r["case"] for r in results if r["problems"]]
    print(f"n5 graph check: gated cases {list(GATED)}; failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
