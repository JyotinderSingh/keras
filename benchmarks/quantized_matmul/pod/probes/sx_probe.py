"""The CUDA launch failure probe of SX. Fork only; not part of a PR.

S1r job 1 (s1r-20261023-2) and S1x job 0 (s1x-20261024-1) ran the rows.py
stage gt-triton-prefill on commit B (the old 43 + the host-scalar fix,
kbench-s1x-b 2e0e834ca). Both stopped in cell 56, float16 gptq
14336x4096 M=256, in the CUDA-graph samples of float, F0 and triton (the
n5 capture failed), with "CUDA error: unspecified launch failure"
(CUDA_ERROR_LAUNCH_FAILED 719 from cuCtxSynchronize_v2). Each candidate
alone and the short flows passed. This probe tests three hypotheses:

- H1: a kernel reads or writes out of bounds, and the access faults only
  with a certain memory layout. compute-sanitizer finds such an access
  on the cell alone, also with no fault.
- H2: the many failed N5 captures before the cell leave torch or the
  harness in a bad state.
- H3: the F0 graph (new since the host-scalar fix) causes the fault.

Run it from the Keras tree with the torch backend (this file is at
benchmarks/quantized_matmul/pod/probes/sx_probe.py, next to
graph_crash_probe.py):

    python benchmarks/quantized_matmul/pod/probes/sx_probe.py install
    python benchmarks/quantized_matmul/pod/probes/sx_probe.py sanitize
    python benchmarks/quantized_matmul/pod/probes/sx_probe.py rows \
        [--blocking] [--coredump] [--no-graph=F0] \
        [--stop-after=float16/gptq/14336x4096/256] -- <rows.py flags>

- install: installs compute-sanitizer, from the NVIDIA CUDA apt
  repository (cuda-sanitizer-<the CUDA version of the driver>, with
  cuda-gdb of the same version for the GPU core dumps), else
  cuda-sanitizer-13-0, else the pip package nvidia-cuda-sanitizer-api.
  Then it runs memcheck on a small torch program, with the options of
  `sanitize`. It writes $KOUT/sx-sanitizer.json (the tool, the options,
  cuda-gdb) and exits 1 when no compute-sanitizer works.
- sanitize: runs cases of the S1r cell in child processes of
  graph_crash_probe.py (flow `alone`: the eager calls and their check,
  then, if not `no_graph`, the capture, 3 replays and the check), each
  under compute-sanitizer, in this order:
  1. mc-f16-precise: memcheck, triton, F0 and float at M=256, eager
     only, with PYTORCH_NO_CUDA_MEMORY_CACHING=1. Each tensor is then
     its own allocation, so memcheck knows its bounds (the caching
     allocator gives memcheck only its segments). `--padding` adds
     invalid bytes after each allocation.
  2. mc-f16-graphs: memcheck, triton, F0 and float at M=256, with the
     caching allocator: the eager calls, the capture and the replays.
  3. mc-f16-m64-precise: memcheck, triton at M=64 (the 17-64 row of the
     kernel table), eager, no caching.
  4. mc-bf16-precise: memcheck, triton in bfloat16 at M=256 and 64,
     eager, no caching.
  5. ic-f16-precise: initcheck, triton and F0 at M=256, eager, no
     caching.
  6. mc-s1r: memcheck, the flow of rows.py `_cell` at M 64 then 256 with
     the four candidates (the failed n5 capture, then the graph
     samples), caching allocator.
  A case does not start with less than 70 s to the stage limit. On a
  host with no CUDA (the dry run), the cases run small, with no
  sanitizer.
- rows: rows.py `run` with the flags after `--`, in a child process
  that patches rows.py from outside (rows.py does not change):
  `--no-graph=F0` captures no graph of F0 (F0 keeps its stream
  samples); `--stop-after=DTYPE/MODE/SHAPE/M` stops at the next
  deadline check of rows.py after that cell (the cell is complete and
  written); `--blocking` sets CUDA_LAUNCH_BLOCKING=1; `--coredump` asks
  the driver for a light GPU core dump on a GPU exception
  (CUDA_ENABLE_COREDUMP_ON_EXCEPTION), which cuda-gdb then reads: the
  kernel, the exception and the PC. The child records each cell
  (`cells.jsonl`: the memory of the caching allocator at the start, the
  candidates and their graphs) and the candidate that runs (`state.json`
  at each change, and the last calls), so a crash names the cell, the
  candidate, the timing (eager or graph) and the sample.

Output, in $KSTAGE_OUT (else ./sx-probe-out): install.log,
sanitizer.json; summary-sanitize.json, sanitizer-<case>.txt; rows.jsonl
and rows_summary.json (rows.py), rows-output.txt, cells.jsonl,
state.json, core/ and cudagdb-<n>.txt, summary-rows.json. One `KPROF`
line per case.

Exit code: install 0 or 1; sanitize 0 (a finding is a result), 2 on an
error of the probe; rows 0, 1 when rows.py ended with a CUDA error (a
result), 2 on another error of rows.py or of the probe.
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = os.path.abspath(__file__)
CRASH_PROBE = os.path.join(os.path.dirname(HERE), "graph_crash_probe.py")
OUT = os.environ.get("KSTAGE_OUT") or os.path.abspath("sx-probe-out")
INFO_NAME = "sx-sanitizer.json"
CUDA_ERROR = re.compile(
    r"CUDA error|AcceleratorError|CUDA_ERROR|illegal memory|launch failure"
    r"|cudaError|device-side assert|misaligned address"
)
# A cell line of rows.py: mode, dtype, shape, M, then the candidate.
ROWS_LINE = re.compile(r"^(int4_g128|int4_pc|gptq|gptq_act|awq) (\w+) (\S+) M=(\d+) ")
# The memcheck options, then the smaller set if this sanitizer refuses one.
MEMCHECK_FULL = [
    "--tool", "memcheck", "--target-processes", "all", "--print-limit",
    "50", "--leak-check", "no", "--padding", "65536",
    "--report-api-errors", "no", "--error-exitcode", "99",
]
MEMCHECK_MIN = [
    "--tool", "memcheck", "--target-processes", "all", "--print-limit", "50",
]
INITCHECK_FULL = [
    "--tool", "initcheck", "--target-processes", "all", "--print-limit",
    "50", "--report-api-errors", "no", "--error-exitcode", "99",
]
INITCHECK_MIN = [
    "--tool", "initcheck", "--target-processes", "all", "--print-limit", "50",
]


def kprof(**fields):
    print("KPROF " + json.dumps(fields, default=str)[:6000], flush=True)


def deadline_left():
    deadline = os.environ.get("KSTAGE_DEADLINE")
    return int(deadline) - time.time() if deadline else 1e9


def gpu():
    if os.environ.get("KMODE") == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_text(cmd, timeout, env=None):
    """Runs `cmd` in its own process group; returns `(rc, output)`. The
    timeout ends the whole group."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, _ = proc.communicate()
        return 124, (out or "") + f"\n[timeout after {timeout:.0f} s]\n"
    return proc.returncode, out or ""


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def trim(path, size):
    try:
        data = open(path, "rb").read()
    except OSError:
        return
    if len(data) > size:
        with open(path, "wb") as f:
            f.write(b"[...]\n" + data[-size:])


def find_sanitizer():
    """The path of a compute-sanitizer binary, or None."""
    found = shutil.which("compute-sanitizer")
    if found:
        return found
    patterns = [
        "/usr/local/cuda/bin/compute-sanitizer",
        "/usr/local/cuda-*/bin/compute-sanitizer",
        "/usr/local/cuda*/compute-sanitizer/compute-sanitizer",
        "/opt/nvidia/*/compute-sanitizer",
    ]
    for site in sys.path:
        if site.endswith("site-packages"):
            patterns += [
                os.path.join(site, "nvidia", "**", "compute-sanitizer"),
            ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def find_cuda_gdb():
    found = shutil.which("cuda-gdb")
    if found:
        return found
    for pattern in ("/usr/local/cuda/bin/cuda-gdb", "/usr/local/cuda-*/bin/cuda-gdb"):
        for path in sorted(glob.glob(pattern)):
            if os.access(path, os.X_OK):
                return path
    return None


def driver_cuda():
    """The CUDA version of the driver as `13-2`, from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return f"{found.group(1)}-{found.group(2)}" if found else None


def load_info():
    root = os.environ.get("KOUT")
    if root and os.path.exists(os.path.join(root, INFO_NAME)):
        try:
            return json.load(open(os.path.join(root, INFO_NAME)))
        except ValueError:
            pass
    return {}


# --- install ---

APT = (
    ". /etc/os-release; arch=$(uname -m); [ \"$arch\" = aarch64 ] && arch=sbsa; "
    "base=https://developer.download.nvidia.com/compute/cuda/repos/"
    "ubuntu${VERSION_ID//./}/$arch; "
    "if [ ! -e /etc/apt/sources.list.d/cuda-ubuntu${VERSION_ID//./}-$arch.list ]; then "
    "curl -fsSL --retry 3 -o /tmp/cuda-keyring.deb $base/cuda-keyring_1.1-1_all.deb && "
    "dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq || exit 41; fi; "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
    "-o Dpkg::Use-Pty=0 {packages}"
)


def sanitizer_test(tool, options):
    """memcheck on a small torch program. Returns `(ok, text)`."""
    program = (
        "import torch; x = torch.arange(1024, device='cuda', dtype=torch.float32);"
        " y = (x * 2).sum(); torch.cuda.synchronize(); print('sum', y.item())"
    )
    rc, text = run_text(
        [tool] + options + [sys.executable, "-c", program],
        min(150, max(30, deadline_left() - 20)),
    )
    ok = rc == 0 and "sum 1047552.0" in text and "ERROR SUMMARY: 0 errors" in text
    return ok, f"rc={rc}\n{text[-3000:]}"


def cmd_install(argv):
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "install.log")
    attempts = []
    tool = find_sanitizer()
    if tool:
        attempts.append(dict(how="found in the image", result="ok", path=tool))
    version = driver_cuda() or "13-2"
    # The first install also brings cuda-gdb, for the GPU core dumps. The
    # others only try to get a sanitizer.
    plan = [
        ("apt", f"cuda-sanitizer-{version} cuda-gdb-{version}"),
        ("apt", f"cuda-sanitizer-{version}"),
        ("apt", "cuda-sanitizer-13-0"),
        ("pip", "nvidia-cuda-sanitizer-api==13.2.87"),
        ("pip", "nvidia-cuda-sanitizer-api"),
    ]
    for index, (how, packages) in enumerate(plan):
        if tool and (index > 0 or find_cuda_gdb()):
            break
        seconds = int(deadline_left()) - 100
        if seconds < 30:
            attempts.append(dict(how=how, packages=packages, result="skipped: stage deadline"))
            continue
        start = time.time()
        if how == "apt":
            if not shutil.which("apt-get"):
                attempts.append(dict(how=how, packages=packages, result="no apt-get"))
                continue
            script = APT.replace("{packages}", packages)
            rc, text = run_text(["bash", "-c", script], seconds)
        else:
            uv = shutil.which("uv")
            if uv:
                cmd = [uv, "pip", "install", "--python", sys.executable, packages]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "-q", packages]
            rc, text = run_text(cmd, seconds)
        with open(log_path, "a") as f:
            f.write(f"\n=== {how} {packages}: rc {rc}\n{text}\n")
        found = find_sanitizer()
        attempts.append(
            dict(
                how=how,
                packages=packages,
                rc=rc,
                seconds=round(time.time() - start, 1),
                sanitizer=found,
                cuda_gdb=find_cuda_gdb(),
                tail=text.strip().splitlines()[-3:],
            )
        )
        tool = tool or found
    trim(log_path, 60000)
    info = dict(tool=tool, attempts=attempts, cuda_gdb=find_cuda_gdb(), driver_cuda=version)
    if tool:
        rc, text = run_text([tool, "--version"], 60)
        info["version"] = text.strip().splitlines()[-2:]
        ok, text = sanitizer_test(tool, MEMCHECK_FULL)
        info["memcheck_options"] = MEMCHECK_FULL
        info["test_full"] = dict(ok=ok, tail=text.splitlines()[-12:])
        if not ok:
            ok_min, text = sanitizer_test(tool, MEMCHECK_MIN)
            info["test_min"] = dict(ok=ok_min, tail=text.splitlines()[-12:])
            if ok_min:
                info["memcheck_options"] = MEMCHECK_MIN
            ok = ok_min
        info["works"] = ok
        info["initcheck_options"] = (
            INITCHECK_FULL if info["memcheck_options"] is MEMCHECK_FULL else INITCHECK_MIN
        )
    else:
        info["works"] = False
    if info.get("cuda_gdb"):
        rc, text = run_text([info["cuda_gdb"], "--version"], 60)
        info["cuda_gdb_version"] = text.strip().splitlines()[:2]
    for path in [os.path.join(OUT, "sanitizer.json")] + (
        [os.path.join(os.environ["KOUT"], INFO_NAME)] if os.environ.get("KOUT") else []
    ):
        with open(path, "w") as f:
            json.dump(info, f, indent=1, default=str)
    kprof(case="install", **{k: v for k, v in info.items() if k != "attempts"})
    for attempt in attempts:
        kprof(case="install-attempt", **attempt)
    return 0 if info["works"] else 1


# --- sanitize ---


def _spec(case, dtype, ms, candidates, flow="alone", **kwargs):
    spec = dict(
        case=case,
        flow=flow,
        mode="gptq",
        dtype=dtype,
        shape="14336x4096",
        ms=list(ms),
        candidates=list(candidates),
        replays=3,
        warmup=1,
        samples=1,
        calls=3,
        dump=False,
        l2_mb=72,
        out=OUT,
    )
    spec.update(kwargs)
    return spec


PRECISE = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
CASES = [
    # (tool, spec, environment)
    ("memcheck", _spec("mc-f16-precise", "float16", [256], ["triton", "F0", "float"], no_graph=True), PRECISE),
    ("memcheck", _spec("mc-f16-graphs", "float16", [256], ["triton", "F0", "float"]), {}),
    ("memcheck", _spec("mc-f16-m64-precise", "float16", [64], ["triton"], no_graph=True), PRECISE),
    ("memcheck", _spec("mc-bf16-precise", "bfloat16", [256, 64], ["triton"], no_graph=True), PRECISE),
    ("initcheck", _spec("ic-f16-precise", "float16", [256], ["triton", "F0"], no_graph=True), PRECISE),
    ("memcheck", _spec("mc-s1r", "float16", [64, 256], ["float", "F0", "n5", "triton"], flow="s1r"), {}),
]


def _small(spec):
    spec = dict(spec)
    spec["shape"] = "512x256"
    spec["ms"] = sorted({min(m, 65) for m in spec["ms"]})
    spec["layers"] = 2
    spec["replays"] = 2
    return spec


def _events(text):
    events = []
    for line in text.splitlines():
        at = line.find("EVENT ")
        if at < 0:
            continue
        try:
            events.append(json.loads(line[at + 6 :]))
        except ValueError:
            continue
    return events


def _sanitizer_lines(text):
    """The report lines of compute-sanitizer: each error with its first
    host frames, and the summary."""
    keep, host = [], 0
    for line in text.splitlines():
        if not line.startswith("========="):
            continue
        body = line[9:].strip()
        if not body:
            continue
        if "Host Frame" in body:
            host += 1
            if host > 4:
                continue
        else:
            host = 0
        keep.append(line[:300])
    return keep


def cmd_sanitize(argv):
    os.makedirs(OUT, exist_ok=True)
    on_gpu = gpu()
    info = load_info()
    tool = info.get("tool") or find_sanitizer()
    summaries = []
    if on_gpu and not tool:
        kprof(case="sanitize", status="not run", reason="no compute-sanitizer")
        return 1
    for name, spec, env in CASES:
        case = spec["case"]
        if not on_gpu:
            spec = _small(spec)
        left = deadline_left()
        if left < 70:
            summary = dict(case=case, status="not run", reason=f"stage deadline ({left:.0f} s left)")
            summaries.append(summary)
            kprof(**summary)
            continue
        child_env = dict(os.environ)
        for key in ("PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_LAUNCH_BLOCKING"):
            child_env.pop(key, None)
        child_env.update(env)
        child = [sys.executable, CRASH_PROBE, "child", json.dumps(spec)]
        options = []
        if on_gpu:
            if name == "memcheck":
                options = info.get("memcheck_options") or MEMCHECK_FULL
            else:
                options = info.get("initcheck_options") or INITCHECK_FULL
        timeout = min(180, left - 40)
        start = time.time()
        rc, text = run_text(([tool] + options if on_gpu else []) + child, timeout, env=child_env)
        if on_gpu and rc != 0 and re.search(r"[Uu]nrecognized|[Uu]nknown option|[Ii]nvalid option", text[:4000]):
            # A sanitizer that refuses an option: the smaller set.
            options = MEMCHECK_MIN if name == "memcheck" else INITCHECK_MIN
            rc, text = run_text([tool] + options + child, min(180, deadline_left() - 40), env=child_env)
        path = os.path.join(OUT, f"sanitizer-{case}.txt")
        with open(path, "w") as f:
            f.write(text)
        trim(path, 60000)
        events = _events(text)
        report = _sanitizer_lines(text)
        errors = [e for e in events if e.get("ev") == "error"]
        checks = [
            {k: e.get(k) for k in ("cand", "phase", "m", "err", "finite", "layers_equal", "same_as_eager", "deterministic")}
            for e in events
            if e.get("ev") == "check"
        ]
        graphs = [
            {k: str(e.get(k))[:160] for k in ("cand", "m", "captured", "skipped") if e.get(k)}
            for e in events
            if e.get("ev") == "graph"
        ]
        tritons = [
            dict(
                grid=e.get("grid"),
                c={k: (e.get("constexprs") or {}).get(k) for k in ("BM", "BNH", "BK", "SPLIT_K", "CONTIGUOUS")},
                stages=e.get("num_stages"),
                regs=e.get("n_regs"),
                launches=e.get("launches"),
            )
            for e in events
            if e.get("ev") == "triton"
        ]
        end = [e for e in events if e.get("ev") == "end"]
        summary_line = [l for l in text.splitlines() if "ERROR SUMMARY" in l][-1:]
        found = re.search(r"ERROR SUMMARY:\s*(\d+)", summary_line[0]) if summary_line else None
        summary = dict(
            case=case,
            tool=name if on_gpu else "none (no CUDA)",
            options=options,
            env=env,
            rc=rc,
            seconds=round(time.time() - start, 1),
            child_status=end[-1]["status"] if end else "no end event",
            sanitizer_errors=int(found.group(1)) if found else None,
            summary_line=summary_line,
            report=report[:60],
            child_errors=[
                {k: str(e.get(k))[:400] for k in ("where", "cand", "m", "error")} for e in errors
            ][:4],
            checks=checks[:12],
            graphs=graphs[:8],
            triton=tritons[:4],
        )
        summaries.append(summary)
        with open(os.path.join(OUT, "summary-sanitize.json"), "w") as f:
            json.dump(summaries, f, indent=1, default=str)
        short = dict(summary)
        short["report"] = report[:24]
        short["checks"] = [
            f"{c['cand']} {c['phase']} m={c['m']} err={c['err']} same={c['same_as_eager']}" for c in checks
        ][:8]
        kprof(**short)
    with open(os.path.join(OUT, "summary-sanitize.json"), "w") as f:
        json.dump(summaries, f, indent=1, default=str)
    return 0


# --- rows ---


def cmd_rows(argv):
    if "--" in argv:
        at = argv.index("--")
        own, flags = argv[:at], argv[at + 1 :]
    else:
        own, flags = argv, []
    p = argparse.ArgumentParser(prog="sx_probe.py rows")
    p.add_argument("--blocking", action="store_true")
    p.add_argument("--coredump", action="store_true")
    p.add_argument("--no-graph", default="")
    p.add_argument("--stop-after", default="")
    args = p.parse_args(own)
    os.makedirs(OUT, exist_ok=True)
    opts = dict(
        no_graph=[n for n in args.no_graph.split(",") if n],
        stop_after=args.stop_after.split("/") if args.stop_after else None,
    )
    env = dict(os.environ)
    env["KSTAGE_OUT"] = OUT
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    if args.blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    core_dir = os.path.join(OUT, "core")
    if args.coredump:
        os.makedirs(core_dir, exist_ok=True)
        env.update(
            CUDA_ENABLE_COREDUMP_ON_EXCEPTION="1",
            CUDA_ENABLE_LIGHTWEIGHT_COREDUMP="1",
            CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION="0",
            CUDA_COREDUMP_FILE=os.path.join(core_dir, "gpucore.%p"),
        )
    deadline = os.environ.get("KSTAGE_DEADLINE")
    kill_at = None
    if deadline:
        # rows.py stops starting cells 20 s + the time of the last group
        # before its deadline; the probe keeps time for its report.
        env["KSTAGE_DEADLINE"] = str(int(deadline) - 90)
        kill_at = int(deadline) - 40
    cmd = [sys.executable, HERE, "rows-child", json.dumps(opts), "--"] + flags
    print("sx rows: " + json.dumps(dict(opts, blocking=args.blocking, coredump=args.coredump, flags=flags)), flush=True)
    out_path = os.path.join(OUT, "rows-output.txt")
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    def on_term(signum, frame):
        _kill(proc)
        print("SIGTERM: the probe stops", flush=True)
        sys.exit(2)

    signal.signal(signal.SIGTERM, on_term)
    killed = []
    if kill_at:
        import threading

        def watchdog():
            while proc.poll() is None:
                if time.time() >= kill_at:
                    killed.append(time.time())
                    _kill(proc)
                    return
                time.sleep(1)

        threading.Thread(target=watchdog, daemon=True).start()
    lines = []
    with open(out_path, "w") as f:
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line.rstrip("\n"))
            if len(lines) > 4000:
                lines = lines[-3000:]
    proc.wait()
    sys.stdout.flush()
    trim(out_path, 120000)
    rc = proc.returncode
    text = "\n".join(lines)
    cells = []
    for line in lines:
        m = ROWS_LINE.match(line)
        if m and m.groups() not in cells:
            cells.append(m.groups())
    crash = [l for l in lines if l.startswith("KSX crash ")]
    done = [l for l in lines if l.startswith("KSX done ")]
    state = {}
    try:
        state = json.load(open(os.path.join(OUT, "state.json")))
    except (OSError, ValueError):
        pass
    other = "\n".join(l for l in lines if not ROWS_LINE.match(l))
    crashed = bool(CUDA_ERROR.search(other)) and (
        bool(crash) or "Traceback" in other or rc not in (0, 1)
    )
    trace = []
    if "Traceback" in text:
        last = text.rfind("Traceback")
        trace = text[last:].splitlines()[-40:]
    summary = dict(
        case=os.path.basename(OUT),
        rc=rc,
        seconds=round(time.time() - start, 1),
        killed_by_probe=bool(killed),
        crashed=crashed,
        cells_printed=len(cells),
        last_cells=[" ".join(c) for c in cells[-3:]],
        stopped=[l for l in lines if l.startswith("stopped:")][-1:],
        # The cell lines of rows.py name the captures that failed; the
        # other lines with a CUDA error are the crash.
        error_lines=[
            l[:400] for l in lines if CUDA_ERROR.search(l) and not ROWS_LINE.match(l)
        ][:12],
        capture_error_lines=len(
            [l for l in lines if CUDA_ERROR.search(l) and ROWS_LINE.match(l)]
        ),
        traceback=[l[:300] for l in trace],
        state=state,
        ksx=(crash or done)[-1:][0][:3000] if (crash or done) else None,
        options=dict(opts, blocking=args.blocking, coredump=args.coredump),
    )
    cores = sorted(glob.glob(os.path.join(core_dir, "*")))
    summary["cores"] = [dict(file=os.path.basename(c), bytes=os.path.getsize(c)) for c in cores]
    gdb = load_info().get("cuda_gdb") or find_cuda_gdb()
    for n, core in enumerate(cores[:2]):
        if not gdb:
            summary.setdefault("cudagdb", []).append("no cuda-gdb")
            break
        rc_gdb, out = run_text(
            [
                gdb, "-batch", "-nx", "-q",
                "-ex", "set pagination off",
                "-ex", "set width 0",
                "-ex", f"target cudacore {core}",
                "-ex", "info cuda kernels",
                "-ex", "bt",
                "-ex", "x/8i $pc",
            ],
            max(20, min(90, kill_at - time.time() + 30 if kill_at else 90)),
        )
        with open(os.path.join(OUT, f"cudagdb-{n}.txt"), "w") as f:
            f.write(out)
        summary.setdefault("cudagdb", []).append(
            [l[:300] for l in out.splitlines() if l.strip()][:40]
        )
    with open(os.path.join(OUT, "summary-rows.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    kprof(**summary)
    if crashed:
        return 1
    return 0 if rc in (0, 1) else 2


def cmd_rows_child(argv):
    opts = json.loads(argv[0])
    flags = argv[2:] if len(argv) > 1 and argv[1] == "--" else argv[1:]
    from benchmarks.quantized_matmul.pod import rows

    torch = getattr(rows, "torch", None)
    cuda = torch is not None and torch.cuda.is_available()
    ring = collections.deque(maxlen=16)
    state = dict(cells_done=0, capture_failures=0, captures=0)
    state_path = os.path.join(rows.OUT, "state.json")
    cells_path = os.path.join(rows.OUT, "cells.jsonl")
    no_graph = set(opts.get("no_graph") or [])
    stop_after = opts.get("stop_after")
    stop_flag = []

    def write_state(**fields):
        state.update(fields)
        state["t"] = round(time.time(), 3)
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(state, default=str))
        os.replace(tmp, state_path)

    def memory():
        if not cuda:
            return None
        stats = torch.cuda.memory_stats()
        return dict(
            allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
            reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 1),
            segments=stats.get("segment.all.current"),
            active=stats.get("active.all.current"),
            device_allocs=stats.get("num_device_alloc"),
            device_frees=stats.get("num_device_free"),
            alloc_retries=stats.get("num_alloc_retries"),
        )

    original_build = rows._build

    def build(name, model, x, want_graph):
        write_state(phase="build", cand=name, kind=None)
        state["building"] = name
        if name in no_graph:
            want_graph = False
        return original_build(name, model, x, want_graph)

    original_capture = rows._capture

    def capture(model, x, ctx):
        name = state.get("building")
        write_state(phase="capture", cand=name, kind=None)
        result = original_capture(model, x, ctx)
        state["captures"] += 1
        if isinstance(result, str):
            state["capture_failures"] += 1
        ring.append((state.get("cell"), name, "capture", "failed" if isinstance(result, str) else "ok"))
        return result

    def tag(name, kind, fn):
        def run():
            if state.get("cand") != name or state.get("kind") != kind:
                write_state(phase="run", cand=name, kind=kind)
            ring.append((state.get("cell"), name, kind))
            return fn()

        return run

    original_cell = rows._cell

    def cell(args, route, base, built, oracles, copy_gbps, deadline):
        key = [base["dtype"], base["mode"], base["shape"], str(base["m"])]
        index = state["cells_done"] + 1
        graphs = {}
        tagged = {}
        for name, item in built.items():
            if isinstance(item, str):
                graphs[name] = "skipped: " + item[:120]
                tagged[name] = item
                continue
            item = dict(item)
            graph = item.get("graph")
            if isinstance(graph, str):
                graphs[name] = graph[:160]
            elif graph is None:
                graphs[name] = "no graph"
            else:
                graphs[name] = "captured"
                item["graph"] = tag(name, "graph", graph)
            item["call"] = tag(name, "eager", item["call"])
            tagged[name] = item
        record = dict(
            level="sxcell",
            cell=index,
            key=key,
            memory=memory(),
            graphs=graphs,
            captures=state["captures"],
            capture_failures=state["capture_failures"],
            t=round(time.time(), 3),
        )
        with open(cells_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        write_state(phase="cell", cell=index, key=key, cand=None, kind=None)
        records = original_cell(args, route, base, tagged, oracles, copy_gbps, deadline)
        state["cells_done"] = index
        write_state(phase="cell done")
        if stop_after and key == [str(v) for v in stop_after]:
            stop_flag.append(index)
        return records

    original_check = rows._check_deadline

    def check_deadline(deadline, reserve):
        if stop_flag:
            reason = "the stop-after cell " + "/".join(stop_after)
            rows.STOPPED.append(reason)
            raise rows.Stop(reason)
        return original_check(deadline, reserve)

    rows._build = build
    rows._capture = capture
    rows._cell = cell
    rows._check_deadline = check_deadline
    write_state(phase="start", pid=os.getpid())
    try:
        rc = rows.main(["run"] + flags)
    except BaseException as error:  # A CUDA error: report, then leave at once.
        print(
            "KSX crash "
            + json.dumps(
                dict(
                    state,
                    ring=list(ring),
                    error=f"{type(error).__name__}: {error}"[:800],
                    memory=None,
                ),
                default=str,
            ),
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    print("KSX done " + json.dumps(dict(state, ring=list(ring)[-4:], rc=rc), default=str), flush=True)
    sys.stdout.flush()
    os._exit(rc if isinstance(rc, int) else 0)


COMMANDS = {
    "install": cmd_install,
    "sanitize": cmd_sanitize,
    "rows": cmd_rows,
    "rows-child": cmd_rows_child,
}


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    try:
        return COMMANDS[argv[0]](argv[1:])
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
