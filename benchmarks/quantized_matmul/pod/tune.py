"""The configuration sweep of Triton K1 and Mosaic GPU K1 (plan_v3.md 4.1,
4.3 and 8.9; SESSION.md sections 3.4 and 8: the sessions T1 and T2).

Run it from the Keras tree, one backend per process: torch for Triton K1,
jax for Mosaic GPU K1. In a stage line (kind `prof`, route `triton` or
`mosaic`):

    python -m benchmarks.quantized_matmul.pod.tune [--kernel triton|mosaic]
        [--modes int4_g128,gptq,awq,int4_pc] [--dtypes bfloat16,float16]
        [--shapes 4096x4096,...] [--m 1,4,16] [--grid small|full]
        [--max_configs N] [--compile_workers N] [--l2_mb X] [--copy_mb N]
        [--warmup N] [--calls N] [--top N] [--confirm_samples N]
        [--watchdog_s S] [--budget_min X]

Offline, on the `tune.jsonl` of a stage, or on a `collect.py` file that
holds `tune.jsonl` records (one summary per stage):

    python -m benchmarks.quantized_matmul.pod.tune --summarize <file>

The values of `--m` make slices: one per (dtype, M bucket of `kernels`).
Each slice has its own deadline: the time that is left, divided by the
number of slices that are left. In each slice:

1. One quantized `Dense` per (mode, shape) with the random stored tensors
   of bench.py (`_dense`), and the inputs of the largest M (the inputs of a
   smaller M are its first rows). The float64 oracle of plan 7.1
   (`oracle_contract`) gives the rows that the checks compare: every row
   up to M = 256, else about 128 rows with an odd step, and the last row.
   An odd step puts a checked row at each residue mod 16, 32 and 64.
2. The baselines at each (mode, shape, M): the reference path F0
   (`kernels._reference`), N5 (`kernels._sum_then_scale`, contiguous groups
   only) and the shipped default row of the kernel.
3. The grid of the slice (`GRIDS`), shuffled with a fixed seed. The sweep
   stops at the slice deadline minus the time of the confirmation pass
   (step 4; an estimate from the default row, at most half of the slice).
   So a slice that stops early has measured an even sample of the grid.
   The `slice` line gives the share of the grid that the slice measured.
   For each configuration, at each (mode, shape, M):
   - The checks, before any timing: G1 (err <= 1.25 x the error of F0 +
     the floor of the dtype), G2 (err <= the bound of the dtype), G4 (the
     whole output is finite) and G5 (a second call gives the same bits),
     with the floors and bounds of plan 7.2. `err` = max |y - oracle| /
     max |oracle| over the checked rows. A configuration that fails is not
     timed: its record has the status `wrong`.
   - The time of the kernel alone. At M <= 64 the calls rotate over R
     copies of the stored tensors, with R x the weight bytes >= 4 x the L2
     cache (plan 8.1), so the weights come from DRAM. At a larger M, R = 1.
     torch: two eager calls give the outputs of the checks. After the
     checks, a CUDA graph of the R calls; CUDA events time each replay.
     The capture is `cuda_graph.capture` (one warm-up stream and one
     capture stream per process, the current stream restored after a
     failed capture, the cuBLAS workspaces cleared before each capture).
     When the first replay does not give the bits of an eager call of
     the R calls, the record has the status `wrong` (check `graph`).
     JAX: one inner `jit` of the call per candidate and cell, and one
     outer `jit` that calls it once per copy. So the kernel lowers once,
     not R times. The first two runs of the outer `jit` give the outputs
     of the checks (a separate check call would lower the kernel again).
     A sample is a run of back-to-back runs of the outer `jit` that lasts
     at least 20 ms, timed with `block_until_ready`.
     At least `--warmup` warm-up calls, `--calls` timed calls and 10
     samples. The time of one call is a sample / its calls. The record
     holds the min and the median, `method` (`graph`, `eager`, `jit` or
     `wall`), and the build and timing seconds.
   - `gbps` = the bytes that a call must move / the median time: the
     codes, the scale, the zero point, the AWQ scales, the activations and
     the output (not `g_idx`). `pct_copy` divides it by the copy bandwidth,
     which the script measures first, with the same timing (`--copy_mb`
     MiB, read and written once). A value above 100 means that the
     rotation did not defeat the L2 cache.
   A configuration that does not compile, exceeds the shared memory or
   does not tile the shape gets the status `skip` and the reason.
4. The confirmation pass. For each key of the configuration tables, the
   `--top` (5) fastest rows of the sweep and the default row again, at
   each cell of the key. Per cell: after about 1 s of GPU load, the rows
   take turns sample by sample (interleaved), `--confirm_samples` (30)
   samples each. One NVML reading (nvidia-smi) of the SM and memory
   clocks per cell, during the load. The records have `cand` `confirm`.

Output, in $KSTAGE_OUT (else ./tune-out):
- One `KRES <json>` line per record on stdout, and the same record in
  `tune.jsonl`. Every record holds the card name (`card`) and the compute
  capability (`cc`). The rows of the tables key on the compute capability
  only, so tune on the card that the rows target.
- `KPROF <json>` summary lines, and the same in `tune_best.json`. The best
  row by the geometric mean of the median times, on the cells of the
  group:
  - `level` `shape`: per (kernel, mode, dtype, shape, M bucket), over the
    Ms of the bucket: `best` (a grid configuration), with the times per M;
  - `level` `mode`: per (kernel, mode, dtype, M bucket), over all shapes;
  - `level` `table`: per key of the configuration table, over its modes
    and all shapes. Triton: (dtype, bucket, `G128`: int4 g128, GPTQ and
    AWQ; `per_channel`: int4 per channel). Mosaic GPU: (dtype, bucket),
    all modes.
  Mosaic GPU at the `mode` and `table` levels: `table_row` (BM, BNH,
  stages, min_blocks) in place of `best`. Its `min_blocks` gives the split
  at each cell by the rule of the kernel, and the grid measured that
  split. `table_row` is the result for `MOSAIC_GPU_CONFIGS`. (One grid
  configuration has one split, and the splits 4 and 8 do not divide the
  K steps of every shape, so a grid configuration is not a table row.)
  Only rows that pass at every cell of the group compete.
  `vs_default`, `vs_F0` and `vs_N5`: the geometric mean of the baseline /
  the one of the best row, on the same cells, when both have one timing
  method. Otherwise the ratio is under `mixed`, with the two methods. On
  CUDA, a `warning` line names each default row that has no CUDA graph.
  At the `table` level, `confirm` holds the rows of the confirmation pass
  with their `vs_default`, and the clocks. When the pass covered every
  cell, its fastest row is the result (`source` `confirm`), `vs_default`
  comes from the pass, and `sweep_vs_default` is the ratio of the sweep.
  `KPROF` lines also give the plan and the measured share of each slice,
  the copy bandwidth, and the first 20 wrong measurements.

The script stops 60 s before the stage limit (KSTAGE_DEADLINE of
run_pod.sh, or `--budget_min`) or on SIGTERM, and writes the summary of
what it measured. It exits 1 when a measurement was wrong or none passed.
A watchdog thread writes the summary and ends the script with exit code 2
when one measurement takes more than `--watchdog_s` (120 s), or still runs
30 s before the stage limit: a kernel that hangs on the GPU blocks the
main thread in a call that a signal cannot stop.

Triton compiles about one kernel per configuration, dtype, code dtype and
specialization of the integer arguments (M = 1, M % 16 == 0, other), each
in 0.2-3 s. On a CUDA device, `--compile_workers` processes first compile
the grid of a slice into the Triton cache, for at most half of the time
that is left and at most to the slice deadline; the main process then
loads each kernel from the cache. The default count is the CPUs of the
process (the affinity mask, cut by the CPU quota of the cgroup,
`cpu.max`) minus 2, at most 8. Mosaic GPU compiles each (configuration,
shape, M) in the main process.

Without a GPU (TRITON_INTERPRET=1, or a JAX CPU), the kernels run in the
Triton interpreter or in Mosaic GPU interpret mode: the checks hold, the
times do not. Then `--max_configs` defaults to 3, R is 1, and the timing
and confirmation take 2 samples. The script ignores flags that it does
not know (the dry run appends the grid flags of bench.py; `--shapes`,
`--m`, `--warmup`, `--calls`, `--copy_mb` and `--l2_mb` replace its own).
"""

import argparse
import collections
import contextlib
import gc
import itertools
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from benchmarks.quantized_matmul import bench
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import AutocastScope
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import kernels
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.utils.dtype_utils import dtype_size

if backend.backend() == "torch":
    import torch

    from benchmarks.quantized_matmul.pod import cuda_graph
    from keras.src.backend.torch.ops import quantized
elif backend.backend() == "jax":
    import jax
    import jax.numpy as jnp

    from keras.src.backend.jax.ops import quantized

OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("tune-out")
STAGE = os.path.basename(OUT)
JOB = os.environ.get("KJOB_ID", "")
MARGIN_S = 60
# The least duration of one JAX sample, and the most runs in it.
SAMPLE_S = 0.02
MAX_INNER = 10000
# The GPU load before the confirmation samples of a cell.
WARM_S = 1.0
# Kernel -> backend.
KERNELS = {"triton": "torch", "mosaic": "jax"}
# The modes that each kernel serves (plan 4.1 and 4.3). Mosaic GPU K1 has
# no general path: GPTQ's activation order takes the reference branch.
MODES = {
    "triton": ("int4_g128", "int4_pc", "gptq", "gptq_act", "awq"),
    "mosaic": ("int4_g128", "int4_pc", "gptq", "awq"),
}
DTYPES = {
    "triton": ("bfloat16", "float16", "float32"),
    "mosaic": ("bfloat16", "float16"),
}
# Dot precision -> (G1 floor, G2 bound), plan 7.2.
TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "tf32": (2e-4, 2**-9),
    "float16": (2e-4, 2**-9),
    "bfloat16": (1e-3, 2**-7),
}
# Kernel -> grid kind -> size -> the values of each axis. Kinds: `decode`
# (M <= 16), `mid` (17-64) and `prefill` (65 and more). Triton: the fields
# of `KernelConfig` (BK divides G = 128, so every K tile of a grouped
# weight takes the fast path). Mosaic GPU: BM, BNH, stages and the split
# of the K steps; the table row holds `min_blocks`, and the summary maps
# the splits back to it.
TRITON_AXES = ("BM", "BNH", "BK", "warps", "stages", "split_k")
MOSAIC_AXES = ("BM", "BNH", "stages", "split_k")
GRIDS = {
    "triton": {
        "decode": {
            "small": ([16], [32, 64], [64, 128], [4], [3, 4], [1, 2, 4]),
            "full": ([16], [16, 32, 64, 128], [32, 64, 128], [2, 4, 8],
                     [2, 3, 4, 5], [1, 2, 4, 8]),
            # The loop jobs of O1 (opt/triton-k1/PLAN.md): 48 rows around
            # the T1a best rows, with the stage axis and split 16.
            "k1": ([16], [32, 64], [64, 128], [4], [2, 3, 4], [2, 4, 8, 16]),
        },
        "mid": {
            "small": ([32, 64], [32, 64], [64, 128], [4], [3], [1, 2]),
            "k1": ([32, 64], [32, 64], [64, 128], [4], [3], [1, 2]),
            "full": ([16, 32, 64], [16, 32, 64, 128], [32, 64, 128],
                     [4, 8], [2, 3, 4], [1, 2, 4]),
        },
        "prefill": {
            "small": ([64, 128], [32, 64], [64, 128], [4, 8], [2, 3], [1]),
            "k1": ([32, 64, 128, 256], [32, 64], [32, 64, 128], [4], [3, 4],
                   [1, 2]),
            "full": ([32, 64, 128, 256], [32, 64, 128], [32, 64, 128],
                     [4, 8], [2, 3, 4], [1, 2]),
        },
    },
    "mosaic": {
        "decode": {
            "small": ([16], [32, 64], [3, 4], [1, 2, 4]),
            "full": ([16], [32, 64, 128], [2, 3, 4, 5], [1, 2, 4, 8]),
        },
        "mid": {
            "small": ([16, 32], [64], [2, 3], [1, 2]),
            "full": ([16, 32, 64], [32, 64, 128], [2, 3, 4], [1, 2, 4, 8]),
        },
        "prefill": {
            "small": ([64], [64, 128], [2, 3], [1]),
            "full": ([16, 32, 64], [32, 64, 128], [2, 3, 4], [1, 2]),
        },
    },
}
# Compute capability -> the shared memory of one block in bytes (the
# opt-in limit), for the SMEM estimate of a Mosaic GPU configuration.
SMEM = {"8.0": 166912, "8.6": 101376, "8.9": 101376, "9.0": 232448}
# The `min_blocks` values that the Mosaic GPU summary tries, as multiples
# of the SM count.
MIN_BLOCKS = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8)
# The `cand` values of the sweep records. Other records: `device` (the
# first line), `slice`, `confirm` and `end`.
MEASURED = ("F0", "N5", "default", "grid")
CLOCKS = ("sm_mhz", "mem_mhz", "max_sm_mhz", "max_mem_mhz", "reasons")
CLOCKS_QUERY = "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_event_reasons.active"

# A timed run: `run()` makes `count` kernel calls; a JAX sample holds
# `inner` runs. `method`: `graph`, `eager`, `jit` or `wall`; `note`: why a
# CUDA graph capture failed.
Timer = collections.namedtuple("Timer", "run count inner method note")
# The note of a CUDA graph whose first replay differs from the eager call.
GRAPH_DIFFERS = "graph: the first CUDA graph replay differs from the eager call"

DEV = {}  # The device facts.
RECORDS = []  # Every record of this process.
WORKERS = []  # The running compile workers.
WRONG = []  # The wrong measurements.
DEADLINE = math.inf
RESULTS = None  # tune.jsonl
STOP = []  # The reason of a stop request (SIGTERM).
# `(start, key)` of the measurement that runs, for the watchdog.
CURRENT = [None]
# The watchdog writes records and the summary from its own thread.
LOCK = threading.RLock()


class Stop(Exception):
    """Ends the measurements; the summary still runs."""


def on_term(*_):
    # A library can wrap the exception (the Triton interpreter raises
    # InterpreterError), so `check_time` and `prepare` also read the flag.
    STOP.append("SIGTERM")
    raise Stop("SIGTERM")


def kprof(**fields):
    line = {"tool": "tune", "stage": STAGE}
    if JOB:
        line["job"] = JOB
    line.update(fields)
    print("KPROF " + json.dumps(line, separators=(",", ":"), default=str), flush=True)


def emit(record):
    record = dict(record, card=DEV.get("name"), cc=DEV.get("cc"))
    text = json.dumps(record, separators=(",", ":"), default=str)
    head = {"stage": STAGE, "job": JOB} if JOB else {"stage": STAGE}
    with LOCK:
        RECORDS.append(record)
        RESULTS.write(text + "\n")
        RESULTS.flush()
        print("KRES " + json.dumps(dict(head, **record), separators=(",", ":"),
                                   default=str), flush=True)


def watch(key):
    """Marks the start (or with `None`, the end) of a measurement."""
    CURRENT[0] = None if key is None else (time.time(), key)


def check_time():
    if STOP:
        raise Stop(STOP[0])
    if time.time() > DEADLINE - MARGIN_S:
        raise Stop("the stage limit is near")


def csv(text):
    return [part for part in text.split(",") if part]


def named(kernel, config, row=False):
    if config is None:
        return None
    axes = TRITON_AXES if kernel == "triton" else MOSAIC_AXES
    if row:
        axes = ("BM", "BNH", "stages", "min_blocks")
    return dict(zip(axes, config))


def nbytes(tensor):
    if tensor is None:
        return 0
    return math.prod(tensor.shape) * dtype_size(standardize_dtype(tensor.dtype)) // 8


def bucket_kind(m):
    return "decode" if m <= 16 else "mid" if m <= 64 else "prefill"


def geomean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values))


# --- Device, clocks, copy bandwidth and timing ---


def cpu_count():
    """The CPUs of this process: the affinity mask, cut by the CPU quota of
    the cgroup (`cpu.max`; cgroup v1: `cpu.cfs_quota_us`). A container can
    see all the CPUs of the host and have a small quota."""
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    quota = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            limit, period = f.read().split()[:2]
        if limit != "max":
            quota = int(limit) / int(period)
    except (OSError, ValueError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                limit = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if limit > 0:
                quota = limit / period
        except (OSError, ValueError):
            pass
    if quota:
        cpus = min(cpus, max(1, int(quota)))
    return cpus


def setup_device(args):
    """The facts of the device in `DEV`."""
    if backend.backend() == "torch":
        interpreted = quantized.INTERPRETED
        cuda = torch.cuda.is_available() and not interpreted
        if quantized.triton is None:
            sys.exit("tune.py: Triton does not import.")
        if not (cuda or interpreted):
            sys.exit("tune.py: no CUDA device, and TRITON_INTERPRET is not 1.")
        DEV.update(interpreted=interpreted, cuda=cuda, name="cpu", cc=None,
                   capability=None, sms=128, l2=0)
        if cuda:
            props = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)
            DEV.update(name=props.name, capability=capability,
                       cc="%d.%d" % capability, sms=props.multi_processor_count,
                       l2=props.L2_cache_size)
    else:
        device = jax.devices()[0]
        cc = getattr(device, "compute_capability", None)
        gpu = device.platform == "gpu" and cc in kernels._MOSAIC_GPU_CAPABILITIES
        if not quantized.plgpu.available:
            sys.exit(f"tune.py: JAX {jax.__version__} has no `plgpu.mma`.")
        sms = getattr(device, "core_count", None) or quantized._SMS.get(cc, 128)
        DEV.update(interpreted=not gpu, cuda=gpu, name=device.device_kind,
                   cc=cc if gpu else "8.9", sms=sms, l2=0)
    if args.l2_mb >= 0:
        DEV["l2"] = int(args.l2_mb * 2**20)
    elif not DEV["l2"]:
        # JAX has no L2 size: KL2_MB of the session, else 128 MiB (more
        # copies than an sm_89 card needs).
        DEV["l2"] = int(float(os.environ.get("KL2_MB", 128)) * 2**20)
    DEV["smem"] = SMEM.get(DEV["cc"], 101376)
    DEV["cpus"] = cpu_count()


def device_facts():
    return {k: v for k, v in DEV.items() if k != "capability"}


def clocks_start():
    """Starts one NVML reading of the clocks (nvidia-smi), or returns
    `None` without a GPU."""
    if not DEV["cuda"]:
        return None
    try:
        return subprocess.Popen(
            ["nvidia-smi", "--format=csv,noheader,nounits", f"--query-gpu={CLOCKS_QUERY}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError as error:
        return str(error)


def clocks_read(probe):
    """The clocks of a reading of `clocks_start`: the SM and memory clocks
    and their maximums in MHz, and the active clock event reasons."""
    if probe is None:
        return None
    if isinstance(probe, str):
        return dict(error=probe[:200])
    try:
        text = probe.communicate(timeout=30)[0].strip()
    except subprocess.TimeoutExpired:
        probe.kill()
        probe.communicate()
        return dict(error="nvidia-smi: no answer in 30 s")
    values = [v.strip() for v in text.splitlines()[0].split(",")] if text else []
    if len(values) != len(CLOCKS):
        return dict(error=text[:200])
    clock = dict(zip(CLOCKS, values))
    for name in CLOCKS[:4]:
        with contextlib.suppress(ValueError):
            clock[name] = int(float(clock[name]))
    return clock


def sync(value=None):
    if backend.backend() == "jax":
        jax.block_until_ready(value)
    elif DEV["cuda"]:
        torch.cuda.synchronize()


def times(timers, samples, warmup):
    """The seconds of one call of each timer: `samples` samples each, after
    `warmup` samples that are not kept. The timers take turns, sample by
    sample. torch on CUDA: CUDA events around each run, one synchronize at
    the end, so the GPU runs the samples back to back. Else the wall clock
    around `inner` runs and a sync."""
    seconds = [[] for _ in timers]
    if backend.backend() == "torch" and DEV["cuda"]:
        events = []
        for turn in range(warmup + samples):
            for i, timer in enumerate(timers):
                if turn < warmup:
                    timer.run()
                    continue
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                timer.run()
                end.record()
                events.append((i, start, end))
        torch.cuda.synchronize()
        for i, start, end in events:
            seconds[i].append(start.elapsed_time(end) / 1e3 / timers[i].count)
        return seconds
    for turn in range(warmup + samples):
        for i, timer in enumerate(timers):
            start = time.perf_counter()
            for _ in range(timer.inner):
                value = timer.run()
            sync(value)
            if turn >= warmup:
                calls = timer.inner * timer.count
                seconds[i].append((time.perf_counter() - start) / calls)
    return seconds


def warm(timers, seconds):
    """About `seconds` of GPU load: turns of the timers that are not kept.
    Without a GPU, one turn."""
    end = time.time() + seconds
    while True:
        times(timers, 0, 1)
        if not DEV["cuda"] or time.time() >= end:
            return


def jax_timer(run, count):
    """The timer of `run`, a JAX call that makes `count` kernel calls: a
    sample is `inner` runs back to back, at least `SAMPLE_S` in total. The
    host dispatch of a run overlaps the device work of the run before it."""
    jax.block_until_ready(run())
    one = math.inf
    for _ in range(2):
        start = time.perf_counter()
        jax.block_until_ready(run())
        one = min(one, time.perf_counter() - start)
    inner = max(1, min(MAX_INNER, math.ceil(SAMPLE_S / max(one, 1e-6))))
    return Timer(run, count, inner, "jit", None)


def torch_graph(run):
    """A CUDA graph of `run`, or `None` and the reason (`GRAPH_DIFFERS`
    when the first replay differs from the eager call)."""
    try:
        graph, _, same = cuda_graph.capture(run, warmup=1)
    except RuntimeError as error:
        return None, f"capture failed: {str(error)[:200]}"
    if not same:
        return None, GRAPH_DIFFERS
    return graph, None


def torch_timer(fn, x, copies):
    """The timer of `fn` over the copies: a CUDA graph of the calls on a
    GPU (eager calls if the capture fails), else the wall clock."""

    def run():
        return [fn(x, c) for c in copies]

    if not DEV["cuda"]:
        return Timer(run, len(copies), 1, "wall", None)
    graph, why = torch_graph(run)
    if graph is None:
        # A `GRAPH_DIFFERS` note makes the record `wrong` (`prepare`).
        return Timer(run, len(copies), 1, "eager", why)
    return Timer(graph.replay, len(copies), 1, "graph", None)


def jax_timed(fn):
    """The outer `jit` of `fn` over the copies. The inner `jit` traces and
    lowers `fn` (and so the Mosaic GPU kernel) once; the outer `jit` calls
    it once per copy."""
    inner = jax.jit(fn)
    return jax.jit(lambda x, copies: [inner(x, c) for c in copies])


def outputs(fn, x, copies):
    """`(y, again, build)`: the output of the first copy, a second output
    to compare it with (G5), and `build()`, which returns the timer."""
    if backend.backend() == "torch":
        y = fn(x, copies[0])
        again = fn(x, copies[-1])
        sync()
        return y, again, lambda: torch_timer(fn, x, copies)
    timed = jax_timed(fn)
    y = timed(x, copies)[0]
    again = timed(x, copies)[-1]
    return y, again, lambda: jax_timer(lambda: timed(x, copies), len(copies))


def copy_gbps(mb):
    """The bandwidth of a device copy of `mb` MiB, read and written once,
    timed as the kernels are."""
    count = max(1, mb) * 2**18
    if backend.backend() == "torch":
        device = "cuda" if DEV["cuda"] else "cpu"
        source = torch.ones(count, device=device)
        target = torch.empty_like(source)
        timer = Timer(lambda: target.copy_(source), 1, 1, "eager", None)
    else:
        source = jnp.ones(count, "float32")
        add = jax.jit(lambda a: a + 1.0)
        timer = jax_timer(lambda: add(source), 1)
    seconds = times([timer], 10, 3)[0]
    return 8 * count / min(seconds) / 1e9


# --- Cells ---


def check_rows(m):
    """The rows that the checks compare with the oracle. Above M = 256, an
    odd step: it is coprime with 16, 32 and 64, so the rows reach every
    residue of these tile sizes."""
    if m <= 256:
        return list(range(m))
    return sorted(set(range(0, m, (m // 128) | 1)) | {m - 1})


def copy_tensor(tensor):
    return tensor.clone() if backend.backend() == "torch" else jnp.copy(tensor)


def numpy32(tensor):
    if tensor is None:
        return None
    return np.asarray(ops.convert_to_numpy(ops.cast(tensor, "float32")))


def build_cell(mode, dtype, k, n, ms, light=False):
    """One quantized `Dense` of `mode` with its stored tensors, the inputs
    of each M, the copies, the bytes per call and the oracle rows.
    `light` (compile workers): the layer and the inputs only."""
    layer = bench._dense(mode, n, k, dtype, 1)
    with AutocastScope(layer.compute_dtype):
        view = layer._quantized_weight()
        tensors = view.read_tensors()
    view = view.with_tensors(tensors)
    x_np = np.random.default_rng(7).standard_normal((max(ms), k))
    x_np = x_np.astype("float32")
    cell = dict(
        mode=mode, k=k, n=n, shape=f"{k}x{n}", layer=layer, view=view,
        geometry=layer._quantization_geometry(), copies=[tensors],
        x={m: ops.cast(ops.convert_to_tensor(x_np[:m]), dtype) for m in ms},
    )
    if light:
        return cell
    weight = sum(nbytes(t) for t in tensors)
    copies = 1
    if max(ms) <= 64 and not DEV["interpreted"]:
        copies = max(1, min(256, math.ceil(4 * DEV["l2"] / weight)))
    cell["copies"] += [tuple(copy_tensor(t) for t in tensors)
                       for _ in range(copies - 1)]
    # The bytes that a call must move, beside the activations and the
    # output. `g_idx` is not one of them.
    stored = sum(nbytes(t) for t in (view.codes, view.scale, view.zero_point,
                                     view.input_scales))
    size = dtype_size(dtype) // 8
    cell["bytes"] = {m: stored + m * k * size + m * n * size for m in ms}
    rows = {m: check_rows(m) for m in ms}
    union = sorted(set().union(*rows.values()))
    position = {row: i for i, row in enumerate(union)}
    zero = view.zero_point
    groups = numpy32(view.g_idx)
    if groups is None and grouped(view):
        # A fixed group index (stack branch 40 on): the view has no
        # `g_idx`, and row `r` is in group `r // G`.
        group = min(view.scheme.group_size, k)
        groups = (np.arange(k) // group).astype("float32")
    oracle = oracle_contract(
        x_np[union],
        ops.convert_to_numpy(view.codes),
        numpy32(view.scale),
        columns=n,
        kernel_shape=(k, n),
        zero_point=None if zero is None else ops.convert_to_numpy(zero),
        g_idx=groups,
        input_scales=numpy32(view.input_scales),
        dtype=dtype,
    )
    cell["want"] = {m: oracle[[position[r] for r in rows[m]]] for m in ms}
    cell["rows"] = {
        m: ops.convert_to_tensor(np.asarray(rows[m], "int32")) for m in ms
    }
    return cell


# --- Candidates ---


def grouped(view):
    """Whether `view` is a grouped weight. The scheme decides it: from
    stack branch 40 on, a grouped view with a fixed group index has no
    `g_idx`. On the branches before, a grouped view always has one."""
    return view.scheme.group_size is not None


def triton_fn(cell, config):
    view = cell["view"]
    kernel_config = quantized.KernelConfig(*config)

    def fn(x, tensors):
        v = view.with_tensors(tensors)
        return quantized.w4a16_matmul(
            x, v.codes, v.scale, v.zero_point, v.g_idx,
            group_size=v.scheme.group_size,
            signed=standardize_dtype(v.codes.dtype) == "int8",
            per_channel=not grouped(v),
            config=kernel_config,
            awq_scales=v.input_scales,
            upcast_dot=DEV["interpreted"],
        )

    return fn


def mosaic_fn(cell, row):
    view, k = cell["view"], cell["k"]

    def fn(x, tensors):
        v = view.with_tensors(tensors)
        is_grouped = grouped(v)
        return quantized.mosaic_gpu_w4a16_matmul(
            x, v.codes, v.scale, v.zero_point,
            group_size=min(v.scheme.group_size, k) if is_grouped else None,
            signed=standardize_dtype(v.codes.dtype) == "int8",
            per_channel=not is_grouped,
            config=row,
            input_scales=v.input_scales,
            interpret=DEV["interpreted"],
        )

    return fn


def baseline_fn(cell, name):
    layer, geometry, view = cell["layer"], cell["geometry"], cell["view"]
    source = kernels._reference if name == "F0" else kernels._sum_then_scale

    def fn(x, tensors):
        return source(layer, geometry, view.with_tensors(tensors), x)

    return fn


def mosaic_split(cell, m, block_m, block_n, min_blocks):
    """The `BNH` and the split of the K steps that the kernel takes."""
    half = cell["n"] // 2
    while half % block_n:
        block_n //= 2
    blocks = -(-m // block_m) * (half // block_n)
    split = quantized._split_k(cell["k"] // quantized._BK, blocks, min_blocks)
    return block_n, split, blocks


def default_row(kernel, cell, m, dtype):
    """The shipped row at this cell: `(row, effective configuration)`."""
    view = cell["view"]
    if kernel == "triton":
        group = view.scheme.group_size
        c = quantized.kernel_config(DEV["capability"], dtype, m, group)
        per_channel = not grouped(view)
        block_k = c.block_k or quantized._block_k(
            None if per_channel else group, cell["k"]
        )
        config = (c.block_m, c.block_nh, block_k, c.num_warps, c.num_stages,
                  c.split_k)
        return config, config
    row = quantized.MOSAIC_GPU_CONFIGS[(DEV["cc"], dtype, kernels._m_bucket(m))]
    block_n, split, _ = mosaic_split(cell, m, row[0], row[1], row[3])
    return row, (row[0], block_n, row[2], split)


def grid_row(kernel, cell, m, config):
    """`(row, effective configuration, reason)` of a grid configuration
    at this cell. `reason` names a configuration that cannot run here."""
    if kernel == "triton":
        block_m, block_n, block_k, _, _, split = config
        tiles = -(-cell["k"] // block_k)
        if split > tiles:
            return None, None, f"split_k {split} > {tiles} K tiles"
        return config, config, None
    block_m, block_n, stages, split = config
    smem = mosaic_smem(config)
    if smem > DEV["smem"]:
        return None, None, f"SMEM estimate {smem} > {DEV['smem']} bytes"
    effective, actual, blocks = mosaic_split(cell, m, block_m, block_n, 1)
    min_blocks = blocks * split
    _, actual, _ = mosaic_split(cell, m, block_m, block_n, min_blocks)
    if actual != split:
        steps = cell["k"] // quantized._BK
        return None, None, f"split_k {split} does not divide K / 128 = {steps}"
    return ((block_m, block_n, stages, min_blocks),
            (block_m, effective, stages, split), None)


def kernel_fn(kernel, cell, row):
    return triton_fn(cell, row) if kernel == "triton" else mosaic_fn(cell, row)


def limits(dtype):
    """(G1 floor, G2 bound) of the dtype and the device."""
    key = dtype
    if backend.backend() == "torch" and dtype == "float32":
        if quantized.float32_input_precision() == "tf32":
            key = "tf32"
    floor, bound = TOLERANCES[key]
    if DEV["interpreted"] and backend.backend() == "torch" and dtype == "bfloat16":
        # The Triton interpreter rounds float32 to bfloat16 toward zero:
        # up to one ulp (2^-7) of the output (quantized_matmul_test.py).
        floor = 2**-7
    return floor, bound


def failure(error):
    """The `skip` result of a configuration that cannot run."""
    if STOP:
        raise Stop(STOP[0]) from error
    name = type(error).__name__
    text = str(error).replace("\n", " ")[:300]
    resource = name == "OutOfResources" or "shared memory" in text.lower()
    return dict(status="skip", reason=f"{name}: {text}", resource=resource,
                failed=True)


def checks(y, again, cell, m, limit):
    """G1, G2, G4 and G5 of the outputs. `limit` is `None` for F0."""
    finite = bool(ops.convert_to_numpy(ops.all(ops.isfinite(y))))
    same = bool(ops.convert_to_numpy(ops.all(ops.equal(y, again))))
    got = numpy32(ops.take(y, cell["rows"][m], axis=0)).astype(np.float64)
    want = cell["want"][m]
    err = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
    result = dict(err=err, limit=limit)
    problems = []
    if not finite:
        problems.append("G4: not finite")
    if not same:
        problems.append("G5: two calls differ")
    if limit is not None and not err <= limit:
        problems.append(f"G1/G2: err {err:.3g} > {limit:.3g}")
    if problems:
        return dict(result, status="wrong", reason="; ".join(problems))
    return dict(result, status="ok")


def prepare(fn, cell, m, limit, key):
    """The checks of `fn` at the cell, then its timer if they pass.
    Returns `(result, timer)`; the timer is `None` unless the status is
    `ok`. `result["build_s"]`: the seconds from the start to the timer."""
    start = time.time()
    watch(key)
    try:
        y, again, build = outputs(fn, cell["x"][m], cell["copies"])
        result = checks(y, again, cell, m, limit)
        if result["status"] != "ok":
            return result, None
        timer = build()
        if timer.note == GRAPH_DIFFERS:
            return dict(result, status="wrong", reason=GRAPH_DIFFERS), None
    except Stop:
        raise
    except Exception as error:  # A configuration that cannot run.
        return failure(error), None
    finally:
        watch(None)
    result["build_s"] = round(time.time() - start, 3)
    return result, timer


def timing(seconds, cell, m, timer):
    median = statistics.median(seconds)
    gbps = cell["bytes"][m] / median / 1e9
    result = dict(min_us=round(min(seconds) * 1e6, 3),
                  median_us=round(median * 1e6, 3), gbps=round(gbps, 2),
                  copies=timer.count, method=timer.method)
    if timer.note:
        result["capture"] = timer.note
    if DEV.get("copy_gbps"):
        result["pct_copy"] = round(100 * gbps / DEV["copy_gbps"], 2)
    return result


def evaluate(fn, cell, m, args, limit, key):
    """The checks and, if they pass, the timing of `fn` at the cell."""
    result, timer = prepare(fn, cell, m, limit, key)
    if timer is None:
        return result
    runs = timer.count * timer.inner
    samples = max(args.min_samples, math.ceil(args.calls / runs))
    start = time.time()
    watch(key)
    try:
        seconds = times([timer], samples, max(1, math.ceil(args.warmup / runs)))[0]
    except Stop:
        raise
    except Exception as error:
        return failure(error)
    finally:
        watch(None)
    result.update(timing(seconds, cell, m, timer), samples=samples,
                  time_s=round(time.time() - start, 3))
    return result


# --- Slices ---


def slices(args):
    """`(dtype, bucket, Ms)` in flag order."""
    for dtype in args.dtypes:
        buckets = {}
        for m in args.m:
            buckets.setdefault(kernels._m_bucket(m), []).append(m)
        for bucket, ms in buckets.items():
            yield dtype, bucket, ms


def mosaic_smem(config):
    """The SMEM of a Mosaic GPU block (plan 4.3): the pipeline buffers of
    the activation, code and rows tiles, and the two staging tiles."""
    block_m, block_n, stages, _ = config
    tiles = 2 * block_m * quantized._BK + quantized._BK * block_n + 16 * block_n
    return stages * tiles + 4 * quantized._BK * block_n


def grid(args, ms):
    """The configurations of the grid of `ms`, shuffled. `--max_configs`
    keeps the first ones that fit in the shared memory (Mosaic GPU: by
    the estimate); on a GPU every other one still gets a skip record."""
    axes = GRIDS[args.kernel][bucket_kind(max(ms))][args.grid]
    configs = list(itertools.product(*axes))
    random.Random(0).shuffle(configs)
    if args.max_configs is not None:
        if args.kernel == "mosaic":
            configs = [c for c in configs if mosaic_smem(c) <= DEV["smem"]]
        configs = configs[: args.max_configs]
    return configs


def build_cells(args, dtype, ms, light=False):
    cells = []
    for mode, (k, n) in itertools.product(args.modes, args.shapes):
        if not light:
            check_time()
        cells.append(build_cell(mode, dtype, k, n, ms, light))
    return cells


def start_workers(args, dtype, ms, until):
    """Starts the compile workers of a slice. They stop at half of the time
    that is left, or at the slice deadline `until` if it is earlier."""
    count = args.compile_workers
    left = DEADLINE - MARGIN_S - time.time()
    stop = min(time.time() + (left / 2 if math.isfinite(left) else 3600), until)
    folder = os.path.join(OUT, "compile")
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        cmd = [sys.executable, "-m", "benchmarks.quantized_matmul.pod.tune",
               *sys.argv[1:], f"--dtypes={dtype}",
               "--m=" + ",".join(map(str, ms)),
               f"--compile_only={i}/{count}", f"--until={stop}"]
        log = open(os.path.join(folder, f"{dtype}-{ms[0]}-{i}.txt"), "w")
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        WORKERS.append((process, log))
    return stop


def wait_workers(until):
    """Waits for the compile workers, and returns their last lines."""
    lines = []
    for process, log in WORKERS:
        try:
            process.wait(timeout=max(1, until + 30 - time.time()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        with open(log.name) as f:
            text = f.read().strip().splitlines()
        try:
            lines.append(json.loads(text[-1]))
        except (IndexError, ValueError):
            lines.append(dict(rc=process.returncode, tail=text[-1:]))
    WORKERS.clear()
    return lines


def stop_workers():
    for process, log in WORKERS:
        process.kill()
        log.close()
    WORKERS.clear()


def compile_only(args):
    """A compile worker: one call of each configuration with
    `index % count == worker`, at each cell and M. The calls compile the
    kernels into the Triton cache."""
    worker, count = (int(v) for v in args.compile_only.split("/"))
    calls = failed = 0
    start = time.time()
    for dtype, bucket, ms in slices(args):
        cells = build_cells(args, dtype, ms, light=True)
        jobs = [None] + grid(args, ms)
        for index, config in enumerate(jobs):
            if index % count != worker:
                continue
            for cell, m in itertools.product(cells, ms):
                if time.time() > args.until:
                    break
                if config is None:
                    row = default_row(args.kernel, cell, m, dtype)[0]
                else:
                    row, _, reason = grid_row(args.kernel, cell, m, config)
                    if reason:
                        continue
                try:
                    kernel_fn(args.kernel, cell, row)(cell["x"][m], cell["copies"][0])
                    calls += 1
                except Exception:
                    failed += 1
                if calls % 100 == 0:
                    sync()
        sync()
    print(json.dumps(dict(worker=worker, calls=calls, failed=failed,
                          seconds=round(time.time() - start, 1))), flush=True)
    return 0


def limit_of(error, floor, bound):
    """The G1/G2 limit of a cell from the error of F0 there."""
    return bound if error is None else min(1.25 * error + floor, bound)


def confirm_seconds(args, defaults):
    """An estimate of the seconds of the confirmation pass of a slice: per
    cell, the warm-up, and the build and the samples of each row at the
    cost of the default row there."""
    if not args.top:
        return 0.0
    total = 0.0
    for mode, r in defaults:
        if mode != "gptq_act":
            sample_s = r["time_s"] / r["samples"]
            total += WARM_S + (args.top + 1) * (
                r["build_s"] + args.confirm_samples * sample_s)
    return total


def run_slice(args, dtype, bucket, ms, until):
    """The baselines, the sweep and the confirmation pass of one slice.
    `until` is the slice deadline."""
    kernel = args.kernel
    configs = grid(args, ms)
    check_time()
    started = time.time()
    workers_until = None
    if args.compile_workers:
        workers_until = start_workers(args, dtype, ms, until)
    cells = build_cells(args, dtype, ms)
    base = dict(kernel=kernel, dtype=dtype, bucket=bucket)
    left = until - time.time()
    kprof(plan=dict(base, ms=ms, modes=args.modes,
                    shapes=[f"{k}x{n}" for k, n in args.shapes],
                    grid=args.grid, configs=len(configs),
                    copies={c["shape"]: len(c["copies"]) for c in cells},
                    deadline_s=round(left) if math.isfinite(left) else None,
                    workers=args.compile_workers, cpus=DEV["cpus"],
                    card=DEV["name"], interpreted=DEV["interpreted"]))
    if workers_until is not None:
        kprof(compile=wait_workers(workers_until))
    floor, bound = limits(dtype)
    errors = {}  # (mode, shape, m) -> the error of F0
    done = {}  # (mode, shape, m, effective configuration) -> result
    defaults = []  # (mode, result) of the default row
    for cell, m in itertools.product(cells, ms):
        check_time()
        key = dict(base, mode=cell["mode"], shape=cell["shape"], m=m)
        result = evaluate(baseline_fn(cell, "F0"), cell, m, args, None,
                          dict(key, cand="F0"))
        emit(dict(key, cand="F0", **clean(result)))
        note_wrong(key, "F0", None, result)
        ref = result.get("err")
        errors[(cell["mode"], cell["shape"], m)] = ref
        limit = limit_of(ref, floor, bound)
        view = cell["view"]
        if cell["mode"] != "gptq_act" and kernels._n5_available(
            m, cell["k"], cell["n"], view
        ):
            result = evaluate(baseline_fn(cell, "N5"), cell, m, args, limit,
                              dict(key, cand="N5"))
            emit(dict(key, cand="N5", **clean(result)))
            note_wrong(key, "N5", None, result)
        row, effective = default_row(kernel, cell, m, dtype)
        result = evaluate(kernel_fn(kernel, cell, row), cell, m, args, limit,
                          dict(key, cand="default", config=row))
        done[(cell["mode"], cell["shape"], m, effective)] = result
        emit(dict(key, cand="default", config=row, eff=effective, **clean(result)))
        note_wrong(key, "default", row, result)
        if result["status"] == "ok":
            defaults.append((cell["mode"], result))
            if DEV["cuda"] and result["method"] == "eager":
                kprof(warning=dict(key, cand="default", config=row,
                                   method=result["method"],
                                   capture=result.get("capture"),
                                   text="the default row is not timed with a CUDA graph"))
    reserve = confirm_seconds(args, defaults)
    sweep_until = until - min(reserve, max(0.0, until - time.time()) / 2)
    state = dict(measured=0, stopped=None)
    try:
        sweep(args, configs, cells, ms, base, errors, done, floor, bound,
              sweep_until, state)
    except Stop as stop:
        state["stopped"] = str(stop)
        raise
    finally:
        measured, stopped = state["measured"], state["stopped"]
        facts = dict(base, ms=ms, configs=len(configs), measured=measured,
                     share=round(measured / len(configs), 3) if configs else 1.0,
                     stopped=stopped, seconds=round(time.time() - started, 1),
                     confirm_reserve_s=round(reserve, 1))
        emit(dict(facts, cand="slice"))
        kprof(slice=facts)
    confirm(args, dtype, bucket, cells, until)
    del cells, done
    gc.collect()
    if backend.backend() == "torch" and DEV["cuda"]:
        torch.cuda.empty_cache()
    elif backend.backend() == "jax":
        jax.clear_caches()


def sweep(args, configs, cells, ms, base, errors, done, floor, bound, until,
          state):
    """The grid of a slice until `until`. `state["measured"]` counts the
    configurations that ran at every cell; `state["stopped"]` gets the stop
    reason."""
    kernel = args.kernel
    broken = {}  # (mode or "any", configuration) -> reason
    failures = 0
    for index, config in enumerate(configs):
        if index % 25 == 0:
            print(f"slice {base['dtype']} {base['bucket']}: configuration "
                  f"{index} of {len(configs)}, {len(RECORDS)} records", flush=True)
        for cell, m in itertools.product(cells, ms):
            check_time()
            if time.time() > until:
                state["stopped"] = "the slice deadline"
                return
            mode = cell["mode"]
            key = dict(base, mode=mode, shape=cell["shape"], m=m, cand="grid",
                       config=config)
            reason = broken.get(("any", config)) or broken.get((mode, config))
            if reason:
                emit(dict(key, status="skip", reason=reason))
                continue
            row, effective, reason = grid_row(kernel, cell, m, config)
            if reason:
                emit(dict(key, status="skip", reason=reason))
                continue
            limit = limit_of(errors[(mode, cell["shape"], m)], floor, bound)
            cached = done.get((mode, cell["shape"], m, effective))
            extra = {} if row == config else dict(row=row)
            if effective != config:
                extra["eff"] = effective
            if cached is not None:
                # The same kernel as an earlier configuration (or the
                # default row) at this cell.
                emit(dict(key, **extra, same_as=list(effective), **clean(cached)))
                continue
            cached = evaluate(kernel_fn(kernel, cell, row), cell, m, args, limit, key)
            done[(mode, cell["shape"], m, effective)] = cached
            emit(dict(key, **extra, **clean(cached)))
            note_wrong(key, "grid", config, cached)
            if cached.get("failed"):
                failures += 1
                where = f"{mode} {cell['shape']} M={m}"
                scope = "any" if cached.get("resource") else mode
                broken[(scope, config)] = f"failed at {where}: {cached['reason']}"
                if failures >= 50:
                    raise Stop(f"50 configurations failed in a row: {cached['reason']}")
            else:
                failures = 0
        state["measured"] = index + 1
        if backend.backend() == "jax":
            # The compiled outer `jit`s of the configuration.
            jax.clear_caches()


def confirm(args, dtype, bucket, cells, until):
    """The confirmation pass of a slice (step 4 of the module docstring),
    until the slice deadline `until`."""
    if not args.top:
        return
    kernel = args.kernel
    by_shape = {(c["mode"], c["shape"]): c for c in cells}
    mine = [r for r in RECORDS if r["cand"] in MEASURED
            and r["dtype"] == dtype and r["bucket"] == bucket]
    floor, bound = limits(dtype)
    errors = {(r["mode"], r["shape"], r["m"]): r.get("err")
              for r in mine if r["cand"] == "F0"}
    samples = args.confirm_samples
    for (_, _, group), (records, keys) in table_keys(kernel, mine).items():
        rows = [row for _, row, _ in ranking(kernel, records, keys)[: args.top]]
        if not rows:
            continue
        kprof(confirm=dict(kernel=kernel, dtype=dtype, bucket=bucket, group=group,
                           cells=len(keys), samples=samples,
                           rows=[named(kernel, r, kernel == "mosaic") for r in rows]))
        for mode, shape, m in sorted(keys):
            check_time()
            if time.time() > until:
                kprof(confirm_stopped=dict(kernel=kernel, dtype=dtype, bucket=bucket,
                                           group=group, at=[mode, shape, m],
                                           reason="the slice deadline"))
                break
            cell = by_shape[(mode, shape)]
            key = dict(kernel=kernel, dtype=dtype, bucket=bucket, mode=mode,
                       shape=shape, m=m, cand="confirm", group=group)
            limit = limit_of(errors.get((mode, shape, m)), floor, bound)
            rows_here = [("default", default_row(kernel, cell, m, dtype)[0])]
            rows_here += [("grid", row) for row in rows]
            timers, kept = [], []
            for rank, (of, row) in enumerate(rows_here):
                rkey = dict(key, of=of, rank=rank, config=row)
                result, timer = prepare(kernel_fn(kernel, cell, row), cell, m, limit, rkey)
                if timer is None:
                    emit(dict(rkey, **clean(result)))
                    note_wrong(key, "confirm", row, result)
                    continue
                timers.append(timer)
                kept.append((rkey, result))
            if not timers:
                continue
            probe = clocks_start()
            start = time.time()
            watch(dict(key, of="all"))
            try:
                warm(timers, WARM_S)
                clock = clocks_read(probe)
                seconds = times(timers, samples, 0)
            except Stop:
                raise
            except Exception as error:
                result = failure(error)
                for rkey, _ in kept:
                    emit(dict(rkey, **clean(result)))
                continue
            finally:
                watch(None)
            spent = round(time.time() - start, 3)
            for (rkey, result), timer, s in zip(kept, timers, seconds):
                emit(dict(rkey, **clean(result), **timing(s, cell, m, timer),
                          samples=samples, time_s=spent, clocks=clock))


def clean(result):
    """A result without the fields that only the sweep loop reads."""
    return {k: v for k, v in result.items() if k not in ("failed", "resource")}


def wrong_entry(key, cand, config, result):
    return dict(key, cand=cand, config=config, err=result.get("err"),
                limit=result.get("limit"), reason=result.get("reason"))


def note_wrong(key, cand, config, result):
    if result.get("status") != "wrong":
        return
    entry = wrong_entry(key, cand, config, result)
    WRONG.append(entry)
    if len(WRONG) <= 20:
        kprof(wrong=entry)


# --- Summary ---


def group_of(kernel, mode):
    """The table key part of a mode: Triton `_CONFIGS` (G 128 or per
    channel), Mosaic GPU `MOSAIC_GPU_CONFIGS` (all modes)."""
    if kernel == "triton":
        return "per_channel" if mode == "int4_pc" else "G128"
    return "all"


def ok_cells(records):
    """`{(dtype, bucket): {(mode, shape, m)}}`: the cells where F0 passed."""
    cells = {}
    for r in records:
        if r["cand"] == "F0" and r["status"] == "ok":
            cells.setdefault((r["dtype"], r["bucket"]), set()).add(
                (r["mode"], r["shape"], r["m"]))
    return cells


def table_keys(kernel, records):
    """`{(dtype, bucket, group): (records, cells)}`: the keys of the
    configuration tables, their sweep records and their cells."""
    ok = ok_cells(records)
    tables = {}
    for r in records:
        if r["mode"] != "gptq_act":
            key = (r["dtype"], r["bucket"], group_of(kernel, r["mode"]))
            tables.setdefault(key, ([], set()))[0].append(r)
    for (dtype, bucket, group), (_, cells) in tables.items():
        cells.update(c for c in ok.get((dtype, bucket), ())
                     if c[0] != "gptq_act" and group_of(kernel, c[0]) == group)
    return tables


def collect(records):
    """The median time and the timing method of each ok record, as
    `name -> cell -> value`. A grid record's name is its configuration."""
    times, methods = {}, {}
    for r in records:
        if r["status"] == "ok":
            name = tuple(r["config"]) if r["cand"] == "grid" else r["cand"]
            cell = (r["mode"], r["shape"], r["m"])
            times.setdefault(name, {})[cell] = r["median_us"]
            methods.setdefault(name, {})[cell] = r.get("method")
    return times, methods


def ranking(kernel, records, cells, table=True):
    """The rows that passed at every cell, fastest first, as `(mean us,
    row, methods)`: the geometric mean of the median times over the cells,
    and the timing method at each cell. Triton, or `table=False`: a row is a
    grid configuration. Mosaic GPU tables: a row is (BM, BNH, stages,
    min_blocks). Its `min_blocks` gives the split at each cell by the rule
    of the kernel, and the grid measured that split. Two rows that give
    the same splits at every cell count once (the smaller `min_blocks`)."""
    times, methods = collect(records)
    found = []
    if not cells:
        return found
    if kernel == "triton" or not table:
        for name, t in times.items():
            if isinstance(name, tuple) and all(c in t for c in cells):
                found.append((geomean([t[c] for c in cells]), name,
                              {c: methods[name][c] for c in cells}))
        return sorted(found, key=lambda f: (f[0], f[1]))
    tried = sorted({name[:3] for name in times if isinstance(name, tuple)})
    values = {max(1, round(f * DEV["sms"])) for f in MIN_BLOCKS} | {1}
    values |= {r["config"][3] for r in records if r["cand"] == "default"}
    seen = set()
    for (block_m, block_n, stages), min_blocks in itertools.product(
        tried, sorted(values)
    ):
        names = []
        for mode, shape, m in sorted(cells):
            k, n = (int(d) for d in shape.split("x"))
            split = mosaic_split(dict(k=k, n=n), m, block_m, block_n, min_blocks)[1]
            name = (block_m, block_n, stages, split)
            if (mode, shape, m) not in times.get(name, {}):
                break
            names.append(name)
        else:
            if tuple(names) in seen:
                continue
            seen.add(tuple(names))
            ordered = sorted(cells)
            found.append((geomean([times[nm][c] for nm, c in zip(names, ordered)]),
                          (block_m, block_n, stages, min_blocks),
                          {c: methods[nm][c] for nm, c in zip(names, ordered)}))
    return sorted(found, key=lambda f: (f[0], f[1]))


def ratios(entry, means, methods, cells, best_us, best_methods):
    """`vs_<name>`: the geometric mean of a baseline / the one of the best
    row, when both have one timing method. Else under `mixed`."""
    ours = sorted({str(v) for v in best_methods.values()})
    for name in ("default", "F0", "N5"):
        if name not in means:
            continue
        ratio = round(means[name] / best_us, 3)
        theirs = sorted({str(methods[name][c]) for c in cells})
        if theirs == ours and len(ours) == 1:
            entry[f"vs_{name}"] = ratio
        else:
            entry.setdefault("mixed", {})[f"vs_{name}"] = dict(
                ratio=ratio, methods={name: "/".join(theirs), "best": "/".join(ours)})


def counts(records):
    configs = {}
    for r in records:
        if r["cand"] == "grid":
            configs.setdefault(tuple(r["config"]), set()).add(r["status"])
    return dict(
        configs=len(configs),
        passed=sum(s == {"ok"} for s in configs.values()),
        wrong=sum(bool(s & {"wrong", "hang"}) for s in configs.values()),
        skipped=sum("skip" in s and not s & {"wrong", "hang"} for s in configs.values()),
    )


def confirmation(kernel, dtype, bucket, group, cells):
    """The confirmation pass of a table key, from its `confirm` records:
    per row, the geometric mean of the median times over the cells of the
    default row, fastest first. Returns `(summary, fastest row)`."""
    rs = [r for r in RECORDS if r["cand"] == "confirm" and r["dtype"] == dtype
          and r["bucket"] == bucket and r["group"] == group]
    if not rs:
        return None, None
    times = {}
    for r in rs:
        if r["status"] == "ok":
            name = "default" if r["of"] == "default" else tuple(r["config"])
            times.setdefault(name, {})[(r["mode"], r["shape"], r["m"])] = r["median_us"]
    base = set(times.get("default", {}))
    rows = sorted((geomean([t[c] for c in base]), name) for name, t in times.items()
                  if name != "default" and base and base <= set(t))
    summary = dict(cells=len(base), of_cells=len(cells), complete=base >= cells,
                   samples=max(r.get("samples") or 0 for r in rs))
    if base:
        default_us = geomean([times["default"][c] for c in base])
        summary["default_us"] = round(default_us, 3)
        summary["rows"] = [dict(row=named(kernel, row, kernel == "mosaic"),
                                us=round(us, 3), vs_default=round(default_us / us, 3))
                           for us, row in rows]
    dropped = [list(name) for name in times if name != "default" and not base <= set(times[name])]
    if dropped:
        summary["dropped"] = dropped
    clocks = [r["clocks"] for r in rs if isinstance(r.get("clocks"), dict)]
    for name in ("sm_mhz", "mem_mhz"):
        values = [c[name] for c in clocks if isinstance(c.get(name), int)]
        if values:
            summary[name] = [min(values), max(values)]
    reasons = sorted({c["reasons"] for c in clocks if "reasons" in c})
    errors = sorted({c["error"] for c in clocks if "error" in c})
    if reasons:
        summary["reasons"] = reasons
    if errors:
        summary["clock_errors"] = errors
    fastest = rows[0][1] if rows and summary["complete"] else None
    return summary, fastest


def entry_for(kernel, level, records, cells, **fields):
    entry = dict(kernel=kernel, level=level, **fields, card=DEV.get("name"),
                 cc=DEV.get("cc"), cells=len(cells), **counts(records))
    times, methods = collect(records)
    means = {name: geomean([t[c] for c in cells]) for name, t in times.items()
             if not isinstance(name, tuple) and cells and all(c in t for c in cells)}
    table = kernel == "mosaic" and level != "shape"
    key = "table_row" if table else "best"
    found = ranking(kernel, records, cells, table)
    if not found:
        return dict(entry, **{key: None})
    mean, row, row_methods = found[0]
    summary, confirmed = None, False
    if level == "table":
        summary, fastest = confirmation(kernel, fields["dtype"], fields["bucket"],
                                        fields["group"], cells)
        match = [f for f in found if fastest is not None and f[1] == fastest]
        if match:
            (mean, row, row_methods), confirmed = match[0], True
    entry.update({key: named(kernel, row, table), key + "_us": round(mean, 3)})
    ratios(entry, means, methods, cells, mean, row_methods)
    if level == "shape":
        pct = {(r["mode"], r["shape"], r["m"]): r.get("pct_copy") for r in records
               if r["status"] == "ok" and r["cand"] == "grid"
               and tuple(r["config"]) == row}
        entry["per_m"] = {
            str(m): dict(us=times[row][c], pct_copy=pct.get(c),
                         **{f"{n}_us": times[n][c] for n in ("default", "F0", "N5")
                            if c in times.get(n, {})})
            for c in sorted(cells) for m in [c[2]]
        }
    if level == "table":
        entry["source"] = "confirm" if confirmed else "sweep"
        if summary is not None:
            entry["confirm"] = summary
        if confirmed:
            entry["sweep_vs_default"] = entry.pop("vs_default", None)
            entry["vs_default"] = summary["rows"][0]["vs_default"]
    return entry


def summarize(kernel):
    measured = [r for r in RECORDS if r["cand"] in MEASURED]
    ok = ok_cells(measured)
    entries = []
    groups = {}
    for r in measured:
        groups.setdefault((r["mode"], r["dtype"], r["bucket"]), []).append(r)
    for (mode, dtype, bucket), records in groups.items():
        cells = {c for c in ok.get((dtype, bucket), ()) if c[0] == mode}
        for shape in sorted({c[1] for c in cells}):
            mine = [r for r in records if r["shape"] == shape]
            entries.append(entry_for(
                kernel, "shape", mine, {c for c in cells if c[1] == shape},
                mode=mode, dtype=dtype, shape=shape, bucket=bucket,
                ms=sorted({c[2] for c in cells if c[1] == shape})))
        if cells:
            entries.append(entry_for(kernel, "mode", records, cells, mode=mode,
                                     dtype=dtype, shape="all", bucket=bucket))
    for (dtype, bucket, group), (records, cells) in table_keys(kernel, measured).items():
        if cells:
            entries.append(entry_for(kernel, "table", records, cells, mode="table",
                                     group=group, dtype=dtype, shape="all",
                                     bucket=bucket))
    return entries


def finish(kernel, run_args, stopped, offline=False, name="tune_best.json"):
    """The summary: `KPROF` lines and the summary file. Returns the exit
    code: 1 when a measurement was wrong or no grid configuration passed."""
    with LOCK:
        if not offline:
            emit(dict(cand="end", kernel=kernel, stopped=stopped))
        measured = [r for r in RECORDS if r["cand"] in MEASURED]
        entries = summarize(kernel) if measured else []
        slice_facts = [{k: v for k, v in r.items() if k != "cand"}
                       for r in RECORDS if r["cand"] == "slice"]
        if offline:
            for facts in slice_facts:
                kprof(slice=facts)
        for entry in entries:
            kprof(**entry)
        passed = sum(r["status"] == "ok" and r["cand"] == "grid" for r in measured)
        summary = dict(device=device_facts(), card=DEV.get("name"), args=run_args,
                       stopped=stopped, records=len(RECORDS), grid_passed=passed,
                       slices=slice_facts, wrong=WRONG, summary=entries)
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        kprof(done=dict(records=len(RECORDS), grid_passed=passed, wrong=len(WRONG),
                        stopped=stopped, card=DEV.get("name")))
        return 1 if WRONG or not passed else 0


def summarize_file(path):
    """`--summarize`: the summary of each tune.py run in `path`, offline."""
    global STAGE
    runs = {}
    with open(path) as f:
        for text in f:
            text = text.strip()
            if not text:
                continue
            r = json.loads(text)
            if "cand" in r and "kernel" in r and r.get("file", "tune.jsonl") == "tune.jsonl":
                runs.setdefault(r.get("stage"), []).append(r)
    if not runs:
        sys.exit(f"tune.py: {path} holds no tune.jsonl records")
    os.makedirs(OUT, exist_ok=True)
    code = 0
    for stage, records in runs.items():
        device = next((r for r in records if r["cand"] == "device"), None)
        if device is None:
            print(f"tune.py: stage {stage}: no device record, no summary", flush=True)
            code = max(code, 1)
            continue
        kernel = device["kernel"]
        if kernel == "mosaic" and backend.backend() != "jax":
            sys.exit("tune.py: the summary of a Mosaic GPU sweep needs KERAS_BACKEND=jax")
        STAGE = stage or STAGE
        DEV.clear()
        DEV.update(device["device"])
        RECORDS[:] = records
        WRONG[:] = [wrong_entry({k: r.get(k) for k in ("kernel", "dtype", "bucket",
                                                         "mode", "shape", "m")},
                                r["cand"], r.get("config"), r)
                    for r in records if r.get("status") in ("wrong", "hang")]
        end = next((r for r in reversed(records) if r["cand"] == "end"), None)
        stopped = end["stopped"] if end else "no end record: the run did not finish"
        name = "tune_best.json" if stage is None else f"tune_best-{stage}.json"
        code = max(code, finish(kernel, device.get("args"), stopped, True, name))
    return code


def watchdog(args):
    """Writes the summary and ends the process (exit code 2) when one
    measurement takes more than `--watchdog_s`, or still runs 30 s before
    the stage limit."""
    while True:
        time.sleep(1)
        current = CURRENT[0]
        if current is None:
            continue
        start, key = current
        now = time.time()
        if now - start > args.watchdog_s:
            reason = f"a measurement took more than {args.watchdog_s:g} s"
        elif now > DEADLINE - MARGIN_S / 2:
            reason = "a measurement still runs 30 s before the stage limit"
        else:
            continue
        with LOCK:
            emit(dict(key, status="hang", reason=reason))
            WRONG.append(dict(key, status="hang", reason=reason))
            kprof(watchdog=dict(key, reason=reason))
            stop_workers()
            finish(args.kernel, vars(args), "watchdog: " + reason)
        os._exit(2)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     allow_abbrev=False)
    parser.add_argument("--kernel", choices=sorted(KERNELS))
    parser.add_argument("--modes", type=csv)
    parser.add_argument("--dtypes", type=csv, default=["bfloat16"])
    parser.add_argument("--shapes", type=csv,
                        default=["4096x4096", "4096x14336", "14336x4096", "2304x9216"])
    parser.add_argument("--m", type=csv, default=["1", "4", "16"])
    parser.add_argument("--grid", choices=["small", "full", "k1"],
                        default="small")
    parser.add_argument("--max_configs", type=int)
    parser.add_argument("--compile_workers", type=int)
    parser.add_argument("--l2_mb", type=float, default=-1)
    parser.add_argument("--copy_mb", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--top", type=int, default=5,
                        help="rows per table key in the confirmation pass (0: none)")
    parser.add_argument("--confirm_samples", type=int, default=30)
    parser.add_argument("--watchdog_s", type=float, default=120)
    parser.add_argument("--budget_min", type=float)
    parser.add_argument("--summarize", metavar="FILE",
                        help="print the summary of a tune.jsonl, offline")
    parser.add_argument("--compile_only", help=argparse.SUPPRESS)
    parser.add_argument("--until", type=float, help=argparse.SUPPRESS)
    args, unknown = parser.parse_known_args()
    if args.summarize:
        return args, unknown
    by_backend = {b: k for k, b in KERNELS.items()}
    args.kernel = args.kernel or by_backend.get(backend.backend())
    if KERNELS.get(args.kernel) != backend.backend():
        parser.error(f"--kernel={args.kernel} needs KERAS_BACKEND="
                     f"{KERNELS.get(args.kernel)}, not {backend.backend()}")
    served = MODES[args.kernel]
    args.modes = args.modes or [m for m in served if m != "gptq_act"]
    for name, values, allowed in (("mode", args.modes, served),
                                  ("dtype", args.dtypes, DTYPES[args.kernel])):
        for value in values:
            if value not in allowed:
                parser.error(f"{args.kernel} does not serve the {name} {value}; "
                             f"it serves {', '.join(allowed)}")
    args.shapes = [tuple(int(d) for d in s.split("x")) for s in args.shapes]
    if args.kernel == "mosaic":
        for k, n in args.shapes:
            if k % 128 or n % 64:
                parser.error(f"the Mosaic GPU alignment gate refuses {k}x{n}")
    args.m = [int(m) for m in args.m]
    return args, unknown


def main():
    global DEADLINE, RESULTS
    args, unknown = parse()
    if args.summarize:
        sys.exit(summarize_file(args.summarize))
    setup_device(args)
    # Before the worker branch: a worker must see the grid of the main
    # process.
    args.min_samples = 10
    if DEV["interpreted"]:
        args.warmup, args.calls = min(args.warmup, 1), min(args.calls, 2)
        args.confirm_samples = min(args.confirm_samples, 2)
        args.min_samples = 2
        if args.max_configs is None:
            args.max_configs = 3
    no_grad = torch.no_grad if backend.backend() == "torch" else contextlib.nullcontext
    if args.compile_only:
        with no_grad():
            sys.exit(compile_only(args))
    os.makedirs(OUT, exist_ok=True)
    if unknown:
        print(f"tune.py: ignored flags: {' '.join(unknown)}", flush=True)
    if args.kernel != "triton":
        # Only the Triton cache on disk carries a compile to the main process.
        args.compile_workers = 0
    elif args.compile_workers is None:
        args.compile_workers = min(8, max(0, DEV["cpus"] - 2)) if DEV["cuda"] else 0
    if os.environ.get("KSTAGE_DEADLINE"):
        DEADLINE = int(os.environ["KSTAGE_DEADLINE"])
    elif args.budget_min:
        DEADLINE = time.time() + 60 * args.budget_min
    signal.signal(signal.SIGTERM, on_term)
    RESULTS = open(os.path.join(OUT, "tune.jsonl"), "w")
    threading.Thread(target=watchdog, args=(args,), daemon=True).start()
    stopped = None
    try:
        with no_grad():
            watch(dict(cand="copy"))
            DEV["copy_gbps"] = round(copy_gbps(args.copy_mb), 2)
            watch(None)
            kprof(copy_gbps=DEV["copy_gbps"], copy_mb=args.copy_mb,
                  l2_mb=round(DEV["l2"] / 2**20, 1), sms=DEV["sms"], cc=DEV["cc"],
                  card=DEV["name"], cpus=DEV["cpus"])
            emit(dict(cand="device", kernel=args.kernel, device=device_facts(),
                      args=vars(args)))
            plan = list(slices(args))
            for i, (dtype, bucket, ms) in enumerate(plan):
                # Each slice gets an equal share of the time that is left.
                left = DEADLINE - MARGIN_S - time.time()
                until = time.time() + left / (len(plan) - i)
                run_slice(args, dtype, bucket, ms, until)
    except (Stop, KeyboardInterrupt) as stop:
        stopped = str(stop) or type(stop).__name__
        kprof(stopped=stopped, records=len(RECORDS))
    finally:
        watch(None)
        # A later SIGTERM must not cut the summary; run_pod.sh sends
        # SIGKILL 30 s after the first one.
        signal.signal(signal.SIGTERM, lambda *_: STOP.append("SIGTERM"))
        stop_workers()
    sys.exit(finish(args.kernel, vars(args), stopped))


if __name__ == "__main__":
    main()
