"""The profiling tools of the pod sessions (SESSION.md section 3.4).

Run it from the Keras tree, one backend per process. In a stage line it is
`$PROF` (`python -m benchmarks.quantized_matmul.pod.prof`):

    $PROF probe [--require ncu] [--require nsys] [--install]
    $PROF run [--triton] [--selftest] [--ir] [--torch-profile]
        [--jax-profile] [--mosaic] [--keep-dumps] [--xla-dump REGEX]
        [-- <bench.py flags>]
    $PROF ncu --kernel REGEX [--launch-skip N] -- <bench.py flags>

`run` runs `bench.py` in this process with the flags after `--`, under the
tools of the options. Then it writes one report per tool into $KSTAGE_OUT.
The bench records of a profiled run are not timing results, so they go to
a temporary file and not into the results, and bench.py runs without its
own torch profiler (two profilers clash). Without `--`, `run` runs no
bench (the dry run adds its grid flags at the end of the line; they are
then ignored).

Each tool prints short `KPROF <json>` lines. `run_pod.sh` copies them from
the stage log to the pod log. Each line holds `tool`, `stage` and `job`.

- `--triton`: one entry for each compiled Triton kernel (one compiled
  configuration) that the run launches: `n_regs`, `n_spills`, `shared`
  (bytes of shared memory), `num_warps`, `num_stages`, the launches, the
  constexpr arguments, and `triton.testing.do_bench` of the last launch of
  that kernel alone (Triton clears the L2 before each call). `gbps_args` is
  the bytes of all tensor arguments, once, divided by the median time.
  `--ir` also writes the TTGIR and the PTX of each kernel. `--selftest`
  launches the small `tl.dot` kernel of this file, so the tool shows its
  numbers also on a commit without a kernel route. With
  TRITON_INTERPRET=1 no kernel compiles, and the report says so.
- `--torch-profile`: the torch profiler table, by device time (by CPU time
  when there is no GPU).
- `--jax-profile`: the JAX profiler trace as a table of device time per
  kernel name (host time per event name when there is no GPU).
- `--mosaic`: sets MOSAIC_GPU_DUMP_PTXAS, MOSAIC_GPU_DUMP_RESOURCES and
  MOSAIC_GPU_DUMP_TO. The report holds the registers, the spills and the
  SMEM of each kernel from the ptxas output, the list of dump files, and
  the ptxas and resource files (`--keep-dumps`: all dump files).
- `--xla-dump REGEX`: the HLO text of the XLA modules whose name matches
  REGEX: the 20 largest modules before the XLA passes and the 20 largest
  after them.

`probe` finds ncu and nsys, and reads the driver setting
RmProfilingAdminOnly and the capabilities of the process. With `--require`,
it runs the tool on a small GPU program: ncu with counter metrics (the
error ERR_NVGPUCTRPERM means that the container cannot read the GPU
performance counters), nsys with CUDA tracing. It exits 1 when a required
tool does not work. `--install` first installs each required tool that is
missing, from the NVIDIA CUDA apt repository (KPROF_CUDA_PKG, default
13-0). It does not install ncu when the counters are blocked
(RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN): ncu cannot work then. The
install ends before the stage limit (KSTAGE_DEADLINE of run_pod.sh) and
keeps time for the tests. The KPROF lines and profilers.json record the
tools that exist, the tools that the probe installed, and the counter
state.

`ncu` runs bench.py under Nsight Compute. It measures one launch of the
kernels whose name matches REGEX (ncu syntax: `regex:<expression>`) with a
short metric set: the time, the DRAM throughput, the achieved occupancy,
the SM activity and the registers per thread. Every ncu command here uses
`--clock-control none`: the GPU keeps its own clocks, as in the timing
stages, and an ncu that a timeout kills leaves no clock lock for the later
stages. ncu flushes the caches before the launch (its default).
"""

import argparse
import collections
import contextlib
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("prof-out")
STAGE = os.path.basename(OUT)
JOB = os.environ.get("KJOB_ID", "")
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
]
PROBE_TARGETS = {
    "torch": "import torch; x = torch.ones(1 << 20, device='cuda'); "
    "print('target ok', (x * 2).sum().item())",
    "jax": "import jax.numpy as jnp; "
    "print('target ok', (jnp.ones(1 << 20) * 2).sum().block_until_ready())",
}


def kprof(tool, **fields):
    line = {"tool": tool, "stage": STAGE}
    if JOB:
        line["job"] = JOB
    line.update(fields)
    text = json.dumps(line, separators=(",", ":"), default=str)
    print("KPROF " + text, flush=True)


def write_json(name, data):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(data, f, indent=1, default=str)


def tail(text, count=40):
    return "\n".join(text.splitlines()[-count:])


# --- Triton ---


class TritonLaunches:
    """Records each launch of a Triton JIT function: the compiled kernel,
    the number of launches, and the arguments of the last launch."""

    def __init__(self):
        from triton.runtime import jit

        self.jit = jit
        self.original = jit.JITFunction.run
        self.kernels = {}
        launches = self

        def run(fn, *args, **kwargs):
            kernel = launches.original(fn, *args, **kwargs)
            if kernel is not None and not kwargs.get("warmup"):
                entry = launches.kernels.setdefault(
                    id(kernel), {"fn": fn, "kernel": kernel, "launches": 0}
                )
                entry["launches"] += 1
                entry["args"], entry["kwargs"] = args, kwargs
            return kernel

        jit.JITFunction.run = run

    def close(self):
        self.jit.JITFunction.run = self.original


def selftest():
    """A small fp16 `tl.dot` kernel with a K loop. It checks itself
    against torch.matmul."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def selftest_matmul(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
            y = tl.load(b + (k + rk)[:, None] * N + rn[None, :])
            acc += tl.dot(x, y)
        tl.store(c + rm[:, None] * N + rn[None, :], acc)

    gpu = torch.cuda.is_available()
    device, dtype = ("cuda", torch.float16) if gpu else ("cpu", torch.float32)
    size = 512 if gpu else 64
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    c = torch.empty(size, size, device=device, dtype=torch.float32)
    grid = (size // 32, size // 32)
    selftest_matmul[grid](a, b, c, size, size, size, BM=32, BN=32, BK=32,
                          num_warps=4, num_stages=3)
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-2, atol=1e-1)
    kprof("triton", selftest="pass", device=device, size=size)


def triton_report(launches, ir):
    import triton.testing

    rows = []
    for entry in launches.kernels.values():
        kernel, fn = entry["kernel"], entry["fn"]
        args, kwargs = entry["args"], entry["kwargs"]
        if getattr(kernel, "n_regs", None) is None and hasattr(kernel, "_init_handles"):
            kernel._init_handles()
        meta = kernel.metadata
        row = dict(
            kernel=kernel.name,
            hash=str(kernel.hash)[:12],
            launches=entry["launches"],
            n_regs=getattr(kernel, "n_regs", None),
            n_spills=getattr(kernel, "n_spills", None),
            shared=getattr(meta, "shared", None),
            num_warps=getattr(meta, "num_warps", None),
            num_stages=getattr(meta, "num_stages", None),
            num_ctas=getattr(meta, "num_ctas", None),
            target=str(getattr(meta, "target", "")),
        )
        skip = ("grid", "warmup", "debug", "instrumentation_mode")
        row["constexprs"] = {
            k: v for k, v in kwargs.items()
            if k not in skip and isinstance(v, (bool, int, float, str))
        }
        tensors = [a for a in list(args) + list(kwargs.values())
                   if hasattr(a, "data_ptr") and hasattr(a, "nbytes")]
        row["args"] = [f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
                       for t in tensors]
        arg_bytes = sum(int(t.nbytes) for t in tensors)
        try:
            ms = triton.testing.do_bench(
                lambda: launches.original(fn, *args, **kwargs),
                quantiles=[0.5, 0.2, 0.8],
            )
            row.update(bench_ms=ms[0], bench_p20_ms=ms[1], bench_p80_ms=ms[2])
            row["arg_mb"] = round(arg_bytes / 2**20, 3)
            row["gbps_args"] = round(arg_bytes / (ms[0] * 1e-3) / 1e9, 1)
        except Exception as error:  # the report keeps the other kernels
            row["bench_error"] = repr(error)[:300]
        if ir:
            for kind in ("ttgir", "ptx"):
                if kind in kernel.asm:
                    path = os.path.join(OUT, f"{kernel.name}-{row['hash']}.{kind}")
                    with open(path, "w") as f:
                        f.write(kernel.asm[kind])
        rows.append(row)
    write_json("triton_kernels.json", rows)
    for row in rows:
        kprof("triton", **{k: v for k, v in row.items() if k not in ("args", "target")})
    return rows


# --- torch and JAX profilers ---


class TorchProfile:
    def __enter__(self):
        import torch
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile

        self.gpu = torch.cuda.is_available()
        activities = [ProfilerActivity.CPU]
        if self.gpu:
            activities.append(ProfilerActivity.CUDA)
        self.profile = profile(activities=activities)
        self.profile.__enter__()
        return self

    def __exit__(self, *exc):
        self.profile.__exit__(*exc)

    def report(self):
        averages = self.profile.key_averages()
        keys = ["self_device_time_total", "self_cuda_time_total"] if self.gpu else []
        keys.append("self_cpu_time_total")
        key = next(k for k in keys if hasattr(averages[0], k)) if averages else keys[-1]
        with open(os.path.join(OUT, "torch_profile.txt"), "w") as f:
            f.write(averages.table(sort_by=key, row_limit=60))
        events = sorted(averages, key=lambda e: getattr(e, key, 0) or 0, reverse=True)
        rows = [dict(name=e.key[:160], count=e.count, self_us=round(getattr(e, key, 0) or 0, 1))
                for e in events[:60]]
        write_json("torch_profile.json", dict(sort_key=key, rows=rows))
        for row in rows[:10]:
            kprof("torch-profiler", key=key, **row)


class JaxProfile:
    def __init__(self, tmp):
        self.dir = os.path.join(tmp, "jax-trace")

    def __enter__(self):
        import jax

        self.trace = jax.profiler.trace(self.dir)
        self.trace.__enter__()
        return self

    def __exit__(self, *exc):
        self.trace.__exit__(*exc)

    def report(self):
        from jax.profiler import ProfileData

        paths = glob.glob(os.path.join(self.dir, "**", "*.xplane.pb"), recursive=True)
        data = ProfileData.from_file(paths[0])
        planes = [p for p in data.planes if p.name.startswith("/device:")]
        device = bool(planes)
        if not device:
            planes = [p for p in data.planes if p.name.startswith("/host:CPU")]
        totals = collections.defaultdict(lambda: [0, 0.0])
        for plane in planes:
            lines = list(plane.lines)
            # The kernels of a GPU are on its "Stream" lines. Other lines of
            # the plane repeat them (XLA Ops, XLA Modules).
            streams = [line for line in lines if line.name.startswith("Stream")]
            for line in streams or lines:
                for event in line.events:
                    total = totals[event.name]
                    total[0] += 1
                    total[1] += event.duration_ns
        rows = [dict(name=name[:160], count=c, total_us=round(ns / 1e3, 1))
                for name, (c, ns) in sorted(totals.items(), key=lambda i: -i[1][1])]
        write_json("jax_profile.json", dict(device=device, planes=[p.name for p in planes],
                                            rows=rows[:60]))
        for row in rows[:10]:
            kprof("jax-profiler", device=device, **row)


# --- Mosaic GPU and XLA dumps ---


class CaptureFds:
    """Sends file descriptors 1 and 2 to a file: ptxas output of C++ code
    does not pass sys.stdout."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = [os.dup(1), os.dup(2)]
        with open(self.path, "wb") as f:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, saved in zip((1, 2), self.saved):
            os.dup2(saved, fd)
            os.close(saved)
        with open(self.path, errors="replace") as f:
            self.text = f.read()
        print(tail(self.text, 200), flush=True)


PTXAS_KERNEL = re.compile(r"(?:Compiling entry function|Function properties for) '?([\w.$]+)'?")
PTXAS_REGS = re.compile(r"Used (\d+) registers")
PTXAS_SPILLS = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
PTXAS_SMEM = re.compile(r"(\d+) bytes smem")


def parse_ptxas(text, source):
    kernels, current = {}, None
    for line in text.splitlines():
        match = PTXAS_KERNEL.search(line)
        if match:
            name = match.group(1)
            current = kernels.setdefault(name, {"kernel": name, "source": source})
            continue
        if current is None:
            continue
        if match := PTXAS_REGS.search(line):
            current["n_regs"] = int(match.group(1))
        if match := PTXAS_SPILLS.search(line):
            current["spill_stores"], current["spill_loads"] = map(int, match.groups())
        if match := PTXAS_SMEM.search(line):
            current["smem"] = int(match.group(1))
    return list(kernels.values())


def mosaic_report(dump_dir, captured, keep):
    kernels = parse_ptxas(captured, "stdout")
    files = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        name = os.path.relpath(path, dump_dir)
        files.append(dict(file=name, bytes=os.path.getsize(path)))
        with open(path, errors="replace") as f:
            kernels += parse_ptxas(f.read(), name)
        if keep or re.search(r"ptxas|resource", name, re.I):
            target = os.path.join(OUT, "mosaic", name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(path, target)
    write_json("mosaic_ptxas.json", dict(kernels=kernels, files=files))
    for kernel in kernels:
        kprof("mosaic-ptxas", **kernel)
    if not kernels:
        kprof("mosaic-ptxas", note="no ptxas output", dump_files=len(files))


def xla_report(dump_dir):
    """Keeps the 20 largest modules after the XLA passes and the 20
    largest before them. Small modules are casts and broadcasts."""
    os.makedirs(os.path.join(OUT, "xla"), exist_ok=True)
    total = kept = 0
    for pattern in ("*after_optimizations.txt", "*before_optimizations.txt"):
        paths = sorted(glob.glob(os.path.join(dump_dir, pattern)), key=os.path.getsize)
        total += len(paths)
        for path in paths[-20:]:
            shutil.copy(path, os.path.join(OUT, "xla"))
            kept += 1
    kprof("xla-dump", modules=total, kept=kept,
          files=len(glob.glob(os.path.join(dump_dir, "*"))))


def run(args, bench_args, has_bench):
    tmp = tempfile.mkdtemp(prefix="kprof-")
    if args.mosaic:
        mosaic_dir = os.path.join(tmp, "mosaic")
        os.makedirs(mosaic_dir)
        os.environ.update(MOSAIC_GPU_DUMP_PTXAS="1", MOSAIC_GPU_DUMP_RESOURCES="1",
                          MOSAIC_GPU_DUMP_TO=mosaic_dir)
    if args.xla_dump:
        xla_dir = os.path.join(tmp, "xla")
        flags = f"--xla_dump_to={xla_dir} --xla_dump_hlo_as_text "
        flags += f"--xla_dump_hlo_module_re={args.xla_dump}"
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flags).strip()
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    launches = TritonLaunches() if args.triton else None
    if args.selftest:
        selftest()
    tools = []
    if has_bench:
        from benchmarks.quantized_matmul import bench

        bench.FLAGS([sys.argv[0], f"--out={tmp}/bench.jsonl", *bench_args])
        # bench.py runs a torch profiler for each torch candidate (the
        # `events` of a record). A second torch profiler, or ncu, cannot run
        # at the same time, and these records are not results.
        bench._torch_facts = lambda run, timing: {}
        with contextlib.ExitStack() as stack:
            if args.mosaic:
                capture = stack.enter_context(CaptureFds(os.path.join(tmp, "stdout.txt")))
            if args.torch_profile:
                tools.append(stack.enter_context(TorchProfile()))
            if args.jax_profile:
                tools.append(stack.enter_context(JaxProfile(tmp)))
            bench.main(None)
    elif bench_args:
        print(f"no `--`: the flags {bench_args} are not used", flush=True)
    for tool in tools:
        tool.report()
    if args.mosaic:
        mosaic_report(mosaic_dir, capture.text if has_bench else "", args.keep_dumps)
    if args.xla_dump:
        xla_report(xla_dir)
    if launches is not None:
        launches.close()
        if interpret:
            kprof("triton", note="TRITON_INTERPRET=1: no kernel compiles")
            write_json("triton_kernels.json", [])
        elif not triton_report(launches, args.ir):
            kprof("triton", note="the run launched no Triton kernel")
            return 3
    return 0


# --- ncu and nsys ---


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    for pattern in (f"/usr/local/cuda*/bin/{name}", f"/opt/nvidia/nsight-*/*/{name}",
                    f"/opt/nvidia/nsight-*/*/bin/{name}", f"/usr/local/cuda*/nsight-*/{name}"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return None


def run_text(cmd, timeout):
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout + done.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, repr(error)


def time_left():
    """The seconds until run_pod.sh ends the stage (KSTAGE_DEADLINE). 20
    minutes outside a stage."""
    deadline = os.environ.get("KSTAGE_DEADLINE")
    return int(deadline) - time.time() if deadline else 1200


def install(packages, seconds, log):
    """Installs apt packages from the NVIDIA CUDA repository. `timeout` ends
    its whole process group (apt-get and dpkg too) after `seconds`, so the
    install ends before the stage limit. Returns the exit code."""
    script = (
        ". /etc/os-release; arch=$(uname -m); [ \"$arch\" = aarch64 ] && arch=sbsa; "
        "base=https://developer.download.nvidia.com/compute/cuda/repos/"
        "ubuntu${VERSION_ID//./}/$arch; "
        "curl -fsSL -o /tmp/cuda-keyring.deb $base/cuda-keyring_1.1-1_all.deb && "
        "dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
        + " ".join(packages)
    )
    cmd = ["timeout", "-k", "10", str(int(seconds)), "bash", "-c", script]
    rc, text = run_text(cmd, seconds + 30)
    with open(log, "a") as f:
        f.write(text)
    return rc


def probe(args):
    pkg = os.environ.get("KPROF_CUDA_PKG", "13-0")
    packages = {"ncu": f"cuda-nsight-compute-{pkg}", "nsys": f"cuda-nsight-systems-{pkg}"}
    info = {}
    try:
        params = open("/proc/driver/nvidia/params").read()
        found = re.search(r"RmProfilingAdminOnly: (\d+)", params)
        info["RmProfilingAdminOnly"] = found.group(1) if found else "absent"
    except OSError:
        info["RmProfilingAdminOnly"] = "no /proc/driver/nvidia/params"
    status = open("/proc/self/status").read() if os.path.exists("/proc/self/status") else ""
    found = re.search(r"CapEff:\s*([0-9a-f]+)", status)
    info["cap_sys_admin"] = bool(found and int(found.group(1), 16) >> 21 & 1)
    info["uid"] = os.getuid()
    # Without CAP_SYS_ADMIN, RmProfilingAdminOnly=1 blocks the counters.
    blocked = info["RmProfilingAdminOnly"] == "1" and not info["cap_sys_admin"]
    paths = {name: find_tool(name) for name in packages}
    info["found"] = {name: path or "absent" for name, path in paths.items()}
    # Install only the required tools that are missing. ncu cannot work
    # when the counters are blocked, so its install is of no use then.
    info["installed"] = {}
    wanted = [name for name in args.require if not paths[name]]
    if args.install and "ncu" in wanted and blocked:
        wanted.remove("ncu")
        info["installed"]["ncu"] = "skipped: counters blocked"
    if args.install and wanted:
        # Keep 60 s and 90 s per required tool for the tests after it.
        seconds = int(time_left()) - 60 - 90 * len(args.require)
        if not shutil.which("apt-get"):
            result = "no apt-get"
        elif seconds < 60:
            result = f"skipped: {seconds} s left for it before the stage limit"
        else:
            rc = install([packages[n] for n in wanted], seconds,
                         os.path.join(OUT, "install.log"))
            # apt-get can exit 0 and install no tool that find_tool finds.
            result = "rc=0, tool not found" if rc == 0 else f"failed rc={rc}"
        for name in wanted:
            paths[name] = find_tool(name)
            info["installed"][name] = "ok" if paths[name] else result
    kprof("probe", **info)
    target = next(([sys.executable, "-c", PROBE_TARGETS[n]] for n in PROBE_TARGETS
                   if importlib.util.find_spec(n)), None)
    tmp = tempfile.mkdtemp(prefix="kprobe-")
    failed = []
    info["counters"] = "not tested"
    if blocked:
        info["counters"] = "blocked: RmProfilingAdminOnly=1, no CAP_SYS_ADMIN"
    for name in packages:
        path = paths[name]
        row = dict(name=name, path=path or "absent")
        if path:
            row["version"] = tail(run_text([path, "--version"], 60)[1], 1)
        if name in args.require:
            limit = min(300, max(30, int(time_left()) - 30))
            if not path or not target:
                row["test"] = "no tool" if not path else "no torch or jax for the target"
            elif name == "ncu":
                rc, text = run_text([path, "--clock-control", "none", "--target-processes",
                                     "all", "--launch-count", "1", "--metrics",
                                     "gpu__time_duration.sum,dram__bytes.sum", *target], limit)
                if "ERR_NVGPUCTRPERM" in text:
                    row["test"] = "counters blocked (ERR_NVGPUCTRPERM)"
                elif rc == 0 and "dram__bytes.sum" in text:
                    row["test"] = "ok"
                else:
                    row["test"] = f"error rc={rc}"
                info["counters"] = row["test"]
                row["output"] = tail(text, 30)
            else:
                report = os.path.join(tmp, "probe")
                rc, text = run_text([path, "profile", "--trace=cuda", "--sample=none",
                                     "--cpuctxsw=none", "--force-overwrite=true", "-o",
                                     report, *target], limit)
                ok = rc == 0 and os.path.exists(report + ".nsys-rep")
                row["test"] = "ok" if ok else f"error rc={rc}"
                row["output"] = tail(text, 30)
            if row["test"] != "ok":
                failed.append(name)
        info[name] = row
        kprof("probe", **{k: v for k, v in row.items() if k != "output"})
    kprof("probe", counters=info["counters"], found=info["found"], installed=info["installed"])
    write_json("profilers.json", info)
    return 1 if failed else 0


def ncu(args, bench_args):
    path = find_tool("ncu")
    if not path:
        kprof("ncu", error="ncu is not installed")
        return 1
    log = os.path.join(OUT, "ncu.csv")
    # The target is `run` without a tool: bench.py without its own profiler.
    # --clock-control none: see the docstring.
    cmd = [path, "--clock-control", "none", "--target-processes", "all",
           "--kernel-name", args.kernel,
           "--launch-skip", str(args.launch_skip), "--launch-count", "1",
           "--metrics", ",".join(NCU_METRICS), "--csv", "--log-file", log,
           sys.executable, "-m", "benchmarks.quantized_matmul.pod.prof", "run",
           "--", *bench_args]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    if "ERR_NVGPUCTRPERM" in text:
        kprof("ncu", error="counters blocked (ERR_NVGPUCTRPERM)")
        return 1
    rows = list(csv.DictReader(line for line in text.splitlines() if line.startswith('"')))
    keep = ("Kernel Name", "Block Size", "Grid Size", "Metric Name", "Metric Unit", "Metric Value")
    rows = [{k: r.get(k) for k in keep} for r in rows if r.get("Metric Name")]
    write_json("ncu.json", rows)
    for r in rows:
        kprof("ncu", kernel=str(r["Kernel Name"])[:120], metric=r["Metric Name"],
              unit=r["Metric Unit"], value=r["Metric Value"])
    if not rows:
        kprof("ncu", error=f"no metric rows (ncu rc={rc})", log=tail(text, 10))
    return 0 if rows else 1


def main():
    argv = sys.argv[1:]
    has_bench = "--" in argv
    own, bench_args = argv, []
    if has_bench:
        own, bench_args = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", allow_abbrev=False)
    p.add_argument("--require", action="append", default=[], choices=["ncu", "nsys"])
    p.add_argument("--install", action="store_true")
    r = sub.add_parser("run", allow_abbrev=False)
    for flag in ("--triton", "--selftest", "--ir", "--torch-profile", "--jax-profile",
                 "--mosaic", "--keep-dumps"):
        r.add_argument(flag, action="store_true")
    r.add_argument("--xla-dump", metavar="REGEX")
    n = sub.add_parser("ncu", allow_abbrev=False)
    n.add_argument("--kernel", required=True)
    n.add_argument("--launch-skip", type=int, default=0)
    args, extra = parser.parse_known_args(own)
    os.makedirs(OUT, exist_ok=True)
    if args.command == "probe":
        sys.exit(probe(args))
    if args.command == "ncu":
        sys.exit(ncu(args, bench_args))
    if args.selftest and not args.triton:
        parser.error("--selftest needs --triton")
    sys.exit(run(args, bench_args, has_bench) if has_bench else run(args, extra, False))


if __name__ == "__main__":
    main()
