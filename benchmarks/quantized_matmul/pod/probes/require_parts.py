"""Runs parts of a sharded pytest suite at the same time. Fork only; not
part of a PR (SESSION.md section 8, S1x).

    PYTHONPATH=. python benchmarks/quantized_matmul/pod/probes/require_parts.py \
        --parts=3,4,5,6 --of=6 [--runner=require|pytest] [--check_whole] \
        -- <pytest arguments>

Each part is one process: the "require" runner
(`.github/workflows/scripts/pytest_require_kernels.py`) or `python -m
pytest`, with `KSHARD=<part>/<of>` and the shard plugin of this folder
(`-p benchmarks.quantized_matmul.pod.shard`), so a part holds the tests
of the S1r stage require-jax-<part>. The parts run at the same time,
with `XLA_PYTHON_CLIENT_PREALLOCATE=false`: no JAX process takes 75% of
the GPU memory (the V2a stage existing-require did the same).

Each part writes `junit-<part>.xml` and `part-<part>.txt` (its output)
to $KSTAGE_OUT. When the stage limit comes near (KSTAGE_DEADLINE minus
`--margin_s`), each part that still runs gets SIGINT: pytest then stops,
prints its summary and writes its junit file. A part that still runs
`--grace_s` later gets SIGKILL; with `-v` its output still names the
result of each test that ran. Then this script writes
`failures-<part>.json` (the totals, and each failure and error with its
message, as the S1r stages) and prints, per part, the exit code, the
seconds, the last summary line and the FAILED and ERROR lines.

`--check_whole` (the CPU stand-in of the dry run) also runs the whole
selection once without KSHARD and checks that the parts pass as many
tests as the whole.

Exit code: 0 when every part exits 0, else 1.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

RUNNER = ".github/workflows/scripts/pytest_require_kernels.py"
SHARD = "benchmarks.quantized_matmul.pod.shard"
SUMMARY = re.compile(r"(^=+ .*(passed|failed|error|skipped|deselected).* =+$)|"
                     r"(^[0-9]+ (passed|failed|error)s?\b.*)")


def _parse(argv):
    if "--" in argv:
        cut = argv.index("--")
        own, rest = argv[:cut], argv[cut + 1 :]
    else:
        own, rest = argv, []
    p = argparse.ArgumentParser()
    p.add_argument("--parts", required=True)
    p.add_argument("--of", type=int, required=True)
    p.add_argument("--runner", choices=["require", "pytest"], default="require")
    p.add_argument("--margin_s", type=int, default=110)
    p.add_argument("--grace_s", type=int, default=60)
    p.add_argument("--check_whole", action="store_true")
    args = p.parse_args(own)
    return args, rest


def _command(args, rest, junit):
    if args.runner == "require":
        cmd = [sys.executable, RUNNER]
    else:
        cmd = [sys.executable, "-m", "pytest"]
    return cmd + rest + ["-p", SHARD, f"--junitxml={junit}"]


def _totals(junit):
    """The totals and the failures of a junit file, as the S1r stages."""
    root = ET.parse(junit).getroot()
    suites = list(root.iter("testsuite"))
    totals = {
        k: sum(int(s.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    failed = [
        dict(
            test=c.get("classname", "") + "::" + c.get("name", ""),
            kind=e.tag,
            message=(e.get("message") or "")[:600],
            text=(e.text or "")[-1200:],
        )
        for c in root.iter("testcase")
        for e in c
        if e.tag in ("failure", "error")
    ]
    return totals, failed


def _passed(text):
    found = re.findall(r"(?:^|[ =])([0-9]+) passed", text)
    return int(found[-1]) if found else 0


def main(argv):
    args, rest = _parse(argv)
    out = os.environ.get("KSTAGE_OUT") or os.path.abspath("require-parts-out")
    os.makedirs(out, exist_ok=True)
    deadline = float(os.environ.get("KSTAGE_DEADLINE") or 0)
    stop_at = deadline - args.margin_s if deadline else None
    parts = [int(p) for p in args.parts.split(",") if p]
    procs = {}
    start = time.time()
    for part in parts:
        env = dict(os.environ)
        env["KSHARD"] = f"{part}/{args.of}"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        junit = os.path.join(out, f"junit-{part}.xml")
        log = open(os.path.join(out, f"part-{part}.txt"), "w")
        cmd = _command(args, rest, junit)
        procs[part] = dict(
            proc=subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            ),
            log=log,
            junit=junit,
            start=time.time(),
            interrupted=False,
        )
    print(
        f"started parts {parts} of {args.of} at the same time "
        f"(runner {args.runner}); stop at "
        f"{time.strftime('%H:%M:%SZ', time.gmtime(stop_at)) if stop_at else 'none'}",
        flush=True,
    )

    def stop(signum=None, frame=None):
        for item in procs.values():
            if item["proc"].poll() is None:
                try:
                    os.killpg(item["proc"].pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if signum is not None:
            sys.exit(1)

    # The stage timeout sends SIGTERM to this process only: the parts run
    # in their own sessions, so end them here.
    signal.signal(signal.SIGTERM, stop)
    interrupted_at = None
    while any(item["proc"].poll() is None for item in procs.values()):
        now = time.time()
        if stop_at and now >= stop_at and interrupted_at is None:
            interrupted_at = now
            for part, item in procs.items():
                if item["proc"].poll() is None:
                    item["interrupted"] = True
                    os.killpg(item["proc"].pid, signal.SIGINT)
            print("stage limit near: SIGINT to the parts that run", flush=True)
        # pytest takes the SIGINT between two Python steps; a long native
        # step (an XLA compile) delays it.
        if interrupted_at and now >= interrupted_at + args.grace_s:
            stop()
            break
        time.sleep(2)
    rc_all = 0
    passed_parts = 0
    for part, item in procs.items():
        proc = item["proc"]
        proc.wait()
        item["log"].close()
        seconds = time.time() - item["start"]
        text = open(os.path.join(out, f"part-{part}.txt"), errors="replace").read()
        lines = text.splitlines()
        summary = [l for l in lines if SUMMARY.search(l.strip())]
        # The -rf summary lines, and with -v the result of each test.
        bad = [
            l
            for l in lines
            if re.match(r"^(FAILED|ERROR) ", l)
            or re.search(r"::\S+ (FAILED|ERROR)\b", l)
        ]
        rc = proc.returncode
        if rc != 0:
            rc_all = 1
        passed_parts += _passed(text)
        print(
            f"== part {part}/{args.of}: rc={rc} seconds={seconds:.0f}"
            f"{' (interrupted at the stage limit)' if item['interrupted'] else ''}",
            flush=True,
        )
        print("\n".join(summary[-2:]) or "(no summary line)")
        for line in bad[:80]:
            print(line[:400])
        if os.path.exists(item["junit"]):
            try:
                totals, failed = _totals(item["junit"])
                result = dict(
                    totals,
                    rc=rc,
                    seconds=round(seconds),
                    interrupted=item["interrupted"],
                    failed=failed,
                )
                with open(os.path.join(out, f"failures-{part}.json"), "w") as f:
                    json.dump(result, f, indent=1)
                print(f"failures-{part}.json:", totals)
            except (ET.ParseError, OSError) as error:
                print(f"failures-{part}.json: not written ({error})")
        else:
            print(f"failures-{part}.json: not written (no junit file)")
            print("\n".join(lines[-15:]))
    print(f"all parts: {time.time() - start:.0f} s, {passed_parts} passed")
    if args.check_whole:
        env = dict(os.environ)
        env.pop("KSHARD", None)
        cmd = _command(args, rest, os.path.join(out, "junit-whole.xml"))
        whole = subprocess.run(
            cmd, env=env, capture_output=True, text=True
        )
        text = whole.stdout + whole.stderr
        open(os.path.join(out, "whole.txt"), "w").write(text)
        count = _passed(text)
        print(f"whole: rc={whole.returncode} {count} passed; parts {passed_parts}")
        if whole.returncode != 0 or count != passed_parts:
            rc_all = 1
    return rc_all


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
