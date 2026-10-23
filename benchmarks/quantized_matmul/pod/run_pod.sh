#!/bin/bash
# run_pod.sh: the pod bootstrap of the quantized-matmul measurement campaign.
# Plan: quant-tooling/audits/kernel-plan-2026-10-21/plan_v3.md, section 8.7.
# The checklist and the session file format are in SESSION.md.
#
# Usage: run_pod.sh [--loop] <session file>
#
# Loop mode (plan 8.9, SESSION.md section 10): `--loop`, or the header line
# `loop yes` of the session file. After the setup and job 0 (the stages of
# the session file), the script reads the queue branch every KPOLL_S
# seconds and runs each new job of job.txt once. The watchdog stops the pod
# after KIDLE_S seconds with no job (KIDLE_STOP), or at KHARD_UNTIL.
#
# Environment (pod_env.py writes the create-pod values):
#   KERAS_SHA    The measurement commit. The results record it.
#   RUN_ID       The name of the run. The chunk lines carry it.
#   KSTART_BY    UTC seconds. A start after this time runs no stage.
#   KSTAGES_UNTIL  UTC seconds. No stage runs past this time.
#   KHARD_UNTIL  UTC seconds. At this time the watchdog sends the results
#                and stops the pod, also when the main loop hangs.
#   KMODE        "gpu" (default) or "cpu". The CPU mode installs CPU wheels,
#                sets TRITON_INTERPRET=1 and JAX_PLATFORMS=cpu, and runs the
#                "cpu" stages in place of the "gpu" stages.
#   KDRYRUN      "1" only in the local Docker dry run: small timing grids, no
#                route gate for timing and profiling stages, no grace time,
#                no self-stop.
#   KCHUNK_SLEEP The pause after each result chunk, in seconds. Default 0.1.
#   KSRC         The Keras source tree. Default: three folders above this file.
#   KWORK        The work folder. Default: /kpod/work.
#   KRESTART_PROBE_UNTIL  S0 only. UTC seconds. Before this time, the first
#                start exits once with code 3, before the trap is set. The
#                KBOOT lines then show what the pod does after an exit.
#   RUNPOD_POD_ID, RUNPOD_API_KEY  RunPod sets them in each pod. The self-stop
#                uses them. This script never prints the key, and the stage
#                commands do not get it.
#   KDRY_SELFSTOP "1" only in the loop dry run: the stop loop runs also with
#                KDRYRUN=1 (the dry run has a fake runpodctl).
# Loop mode only:
#   KIDLE_S      Seconds with no job before the watchdog stops the pod.
#                Default: the header line `idle_stop_s`, else 900.
#   KPOLL_S      Seconds between two reads of the queue. Default 30.
#   KQUEUE_BRANCH  The queue branch. pod_env.py sets it. Default
#                kernel-pod-queue-<RUN_ID>, the default of push_job.sh.
#   KQUEUE_REFS  The ref list of the fork (git smart HTTP, not cached by
#                GitHub). It gives the head commit of the queue branch.
#   KARCHIVE     The tarball URL of the fork: $KARCHIVE/<SHA>.tar.gz. The
#                code of a job comes from the tarball of its SHA, and job.txt
#                from the tarball of the head commit of the queue branch. A
#                URL of one commit, so a cached copy is always correct.
#                raw.githubusercontent.com is not used: it can keep a 404
#                for 300 s, and its first read of a commit can take more
#                than a minute. None of these reads needs a key.
#
# Log markers, one per line, at the start of the line:
#   KBOOT KPROBE KFILL KINSTALL KSTAGE KTODO KSKIP KFAIL KWARN KLOG KRES
#   KPROF KBEGIN KCHUNK KEND KDONE KEXIT KSTOP KWATCHDOG
#   Loop mode: KJOB KPOLL KLOOP KIDLE_STOP
#
# Steps (plan 8.7): exit trap with self-stop; watchdog; start guards; stage
# deadline and per-stage timeouts; hardware probe; parallel venv installs;
# correctness stages; timing and profiling stages; the results as
# gzip+base64 chunks; KDONE; grace; self-stop. Loop mode (plan 8.9): the
# stages of the session file are job 0; then the jobs of the queue; then the
# idle stop.

set -u

# Package versions (plan 8.4, 2026-10-21). Read the jax-ci, torch, Triton and
# tf versions from the latest Keras GPU CI logs before each session.
JAX_CI=${JAX_CI:-0.10.2}
JAX_NEW=${JAX_NEW:-0.12.0}
JAX_FLOOR=${JAX_FLOOR:-0.11.2}
TORCH=${TORCH:-2.14.1}
TRITON=${TRITON:-3.8.0}
TF=${TF:-2.21.0}
TORCH_CUDA_INDEX=${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu130}
TORCH_CPU_INDEX=${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}
# The Keras runtime dependencies (pyproject.toml) and pytest.
COMMON="absl-py numpy rich namex h5py optree ml-dtypes packaging pytest"
# The dry run appends these flags to each timing and profiling stage. The
# last value of a flag wins, so they replace the grid of the stage.
SMOKE="--shapes=256x512 --m=1,16 --logits_shapes=tiny-tied,tiny-untied"
SMOKE="$SMOKE --logits_m=1,16 --models=tiny --warmup=2 --calls=5"
SMOKE="$SMOKE --copy_mb=64 --l2_mb=0 --max_chain_layers=8"

KLOOP=0
if [ "${1:-}" = --loop ]; then KLOOP=1; shift; fi
SESSION_FILE=$(readlink -f "${1:?usage: run_pod.sh [--loop] <session file>}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KSRC=${KSRC:-$(cd "$HERE/../../.." && pwd)}
KWORK=${KWORK:-/kpod/work}
KMODE=${KMODE:-gpu}
KDRYRUN=${KDRYRUN:-0}
RUN_ID=${RUN_ID:-norun}
KOUT=$KWORK/out
# The name of the results of KOUT on the KBEGIN, KCHUNK and KEND lines.
KNAME=$RUN_ID
VENVS=$KWORK/venvs
export UV_CACHE_DIR=$KWORK/uv-cache
# Each venv gets hard links to the files of the uv cache, so a wheel uses its
# disk space once. S0 used the default mode of uv 0.11 (`clone`), and its
# 40 GB disk was full (SESSION.md section 7). If a hard link fails (for
# example across two file systems), uv makes a full copy and prints a
# warning.
export UV_LINK_MODE=${UV_LINK_MODE:-hardlink}
export KCHUNK_SLEEP=${KCHUNK_SLEEP:-0.1}
export KMODE RUN_ID KOUT
if [ "$KMODE" = cpu ]; then
    export TRITON_INTERPRET=1 JAX_PLATFORMS=cpu
fi
KPOLL_S=${KPOLL_S:-30}
KQUEUE_BRANCH=${KQUEUE_BRANCH:-kernel-pod-queue-$RUN_ID}
KQUEUE_REFS=${KQUEUE_REFS:-https://github.com/JyotinderSingh/keras.git/info/refs?service=git-upload-pack}
KARCHIVE=${KARCHIVE:-https://github.com/JyotinderSingh/keras/archive}

now() { date -u +%FT%TZ; }
# Writes a small file in one step, so that the watchdog never reads half of it.
put() { echo "$2" >"$1.tmp" && mv -f "$1.tmp" "$1"; }

# --- The S0 restart probe. It runs before the trap, so this exit returns. ---
mkdir -p "$KWORK"
if [ -n "${KRESTART_PROBE_UNTIL:-}" ] && [ ! -e "$KWORK/booted" ] &&
    [ "$(date +%s)" -lt "$KRESTART_PROBE_UNTIL" ]; then
    touch "$KWORK/booted"
    echo "KBOOT exit-probe $(now): this start exits with code 3 before the trap"
    exit 3
fi
marker=absent
if [ -e "$KWORK/booted" ]; then marker=kept; fi
echo "KBOOT start $(now) run=$RUN_ID sha=${KERAS_SHA:-} mode=$KMODE marker=$marker"

# --- Self-stop (plan 8.7 step 1). ---
STOP_VIA=none
stop_with() {  # method 0-3
    case $1 in
        # Method 0 is the runpodctl 1.x syntax: the image has runpodctl 1.x
        # (S0). Method 1 is the 2.x syntax.
        0) STOP_VIA="runpodctl stop pod"
            command -v runpodctl >/dev/null 2>&1 &&
                timeout 60 runpodctl stop pod "$RUNPOD_POD_ID" ;;
        1) STOP_VIA="runpodctl pod stop"
            command -v runpodctl >/dev/null 2>&1 &&
                timeout 60 runpodctl pod stop "$RUNPOD_POD_ID" ;;
        # The key goes to curl on stdin (-H @-), not in its arguments. So
        # `ps` and /proc/<pid>/cmdline do not show it. printf is a builtin.
        2) STOP_VIA="REST v2 POST /v2/pods/{id}/action"
            printf 'Authorization: Bearer %s\n' "${RUNPOD_API_KEY:-}" |
                timeout 60 curl -fsS -X POST -o /dev/null -H @- \
                    -H "Content-Type: application/json" -d '{"action":"stop"}' \
                    "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" ;;
        3) STOP_VIA="REST v1 POST /v1/pods/{id}/stop"
            printf 'Authorization: Bearer %s\n' "${RUNPOD_API_KEY:-}" |
                timeout 60 curl -fsS -X POST -o /dev/null -H @- \
                    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" ;;
    esac
}

self_stop() {  # attempt; each attempt starts at the next method
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        STOP_VIA="none: RUNPOD_POD_ID is not set"
        return 1
    fi
    local i
    for i in 0 1 2 3; do
        stop_with $(((i + $1 - 1) % 4)) && return 0
    done
    return 1
}

# A stop that works ends this container in seconds. A method can exit 0 and
# stop nothing (an API error that it does not report). So, 120 s after a
# method exits 0, if this process still runs, the next attempt starts at the
# next method. The loop never returns on a pod.
stop_loop() {
    if [ "$KDRYRUN" = 1 ] && [ "${KDRY_SELFSTOP:-0}" != 1 ]; then
        echo "KSTOP skipped: dry run"
        return
    fi
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        if self_stop "$attempt"; then
            echo "KSTOP ok attempt=$attempt via=$STOP_VIA $(now)"
            sleep 120
            echo "KSTOP no-effect attempt=$attempt via=$STOP_VIA $(now): the pod still runs"
        else
            echo "KSTOP failed attempt=$attempt last=$STOP_VIA $(now). Next attempt in 60 s."
            sleep 60
        fi
    done
}

# The exit trap. It never returns on a pod, so a restart of the container
# never runs the stages again.
WATCHDOG=""
finish() {
    local code=$?
    if [ -n "$WATCHDOG" ]; then kill "$WATCHDOG" 2>/dev/null; fi
    if [ -d "$KOUT" ] && [ ! -e "$KOUT/emitted" ]; then emit_results; fi
    echo "KEXIT $code"
    stop_loop
}

# --- The results: gzip, base64, chunks of at most 4 KB (plan 8.7 step 7). ---
# One log second holds at most 1 / KCHUNK_SLEEP chunks. So a `since` read of
# stream-pod-logs always gets past a read that the byte cap cuts.
emit_results() {
    python3 - "$KOUT" "$KNAME" <<'EOF'
import base64, glob, gzip, hashlib, json, os, sys, time

out, run = sys.argv[1], sys.argv[2]
lines = []


def add(record):
    lines.append(json.dumps(record, separators=(",", ":")))


def tail(path, count):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-count:])


def add_jsonl(path, wrap):
    for text in open(path, errors="replace"):
        text = text.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError:
            record = {"level": "bad-line", "text": text[:2000]}
        add(dict(record, **wrap) if wrap else record)


if os.path.exists(f"{out}/probe.txt"):
    add({"level": "probe", "text": tail(f"{out}/probe.txt", 400)})
if os.path.exists(f"{out}/records.jsonl"):
    add_jsonl(f"{out}/records.jsonl", None)
for path in sorted(glob.glob(f"{out}/install-*.log")):
    name = os.path.basename(path)[:-4]
    add({"level": "log", "stage": name, "text": tail(path, 60)})
for folder in sorted(glob.glob(f"{out}/stages/*")):
    stage = os.path.basename(folder)
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            path = os.path.join(root, file)
            rel = os.path.relpath(path, folder)
            if rel == "bench.jsonl":
                add_jsonl(path, None)  # bench records carry `env`
            elif rel == "log.txt":
                add({"level": "log", "stage": stage, "text": tail(path, 200)})
            elif file.endswith(".jsonl"):
                add_jsonl(path, {"stage": stage, "file": rel})
            elif file.endswith(".json"):
                try:
                    data = json.load(open(path))
                except ValueError:
                    data = tail(path, 400)
                add({"level": "artifact", "stage": stage, "file": rel, "data": data})
            else:
                text = open(path, "rb").read()[-65536:].decode(errors="replace")
                add({"level": "artifact", "stage": stage, "file": rel, "text": text})

blob = gzip.compress(("\n".join(lines) + "\n").encode(), 9)
text = base64.b64encode(blob).decode()
chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
total = len(chunks)
pause = float(os.environ.get("KCHUNK_SLEEP", "0.1"))
print(f"KBEGIN {run} chunks={total} gz_bytes={len(blob)} "
      f"sha256={hashlib.sha256(blob).hexdigest()} records={len(lines)}", flush=True)
for seq, chunk in enumerate(chunks, 1):
    md5 = hashlib.md5(chunk.encode()).hexdigest()[:8]
    print(f"KCHUNK {run} {seq}/{total} {md5} {chunk}", flush=True)
    time.sleep(pause)
print(f"KEND {run} chunks={total}", flush=True)
EOF
    touch "$KOUT/emitted"
}

# --- Stage records. Values are plain tokens, so printf writes valid JSON. ---
record_stage() {  # name kind route venv mode status rc seconds reason
    printf '{"level":"stage","session":"%s","name":"%s","kind":"%s","route":"%s","venv":"%s","mode":"%s","status":"%s","rc":%s,"seconds":%s,"reason":"%s"}\n' \
        "$SESSION" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" >>"$KOUT/records.jsonl"
}

skip_stage() {  # name kind route venv mode reason
    echo "KSKIP $1: $6"
    record_stage "$1" "$2" "$3" "$4" "$5" skip null 0 "$6"
}

# Copies stdin to the pod log as `KLOG <name> | <line>` lines. awk ends each
# line with a newline, also the last line of a file with no final newline.
# So the next marker line starts on a new line.
klog() {  # name
    awk -v p="KLOG $1 | " '{print p $0}'
}

# --- Session header (first pass over the file). ---
SESSION=unnamed VENV_LIST="" INSTALL_MIN=15 GLOBAL_MIN=60 L2_MB=0 LOGCAP_KB=0
KGRACE_S=300 IDLE_S=900
while read -r key value _; do
    case "$key" in
        session) SESSION=$value ;;
        venvs) VENV_LIST=$(sed -n 's/^venvs[[:space:]]*//p' "$SESSION_FILE" | head -1) ;;
        install_timeout) INSTALL_MIN=$value ;;
        global_timeout) GLOBAL_MIN=$value ;;
        grace_s) KGRACE_S=$value ;;
        l2_mb) L2_MB=$value ;;
        logcap_kb) LOGCAP_KB=$value ;;
        loop) KLOOP=1 ;;
        idle_stop_s) IDLE_S=$value ;;
    esac
done <"$SESSION_FILE"
export KL2_MB=$L2_MB
KIDLE_S=${KIDLE_S:-$IDLE_S}
if [ "$KDRYRUN" = 1 ]; then KGRACE_S=0; fi

mkdir -p "$KOUT/stages" "$VENVS"
trap finish EXIT

# --- The watchdog (plan 8.7 step 2, plan 8.9). ---
# Some hangs have no timeout: nvidia-smi on a GPU that fell off the bus, or
# a process in D state that `timeout -k` cannot kill. Then the trap does not
# run. At KHARD_UNTIL the watchdog sends the results so far (if the main
# loop did not send them) and runs the stop loop. It does not use the main
# loop. The trap stops the watchdog.
# Loop mode: the main loop writes $KWORK/busy_until. A job sets it to its
# deadline + 2 min, the end of a job sets it to the time of the end. When
# KIDLE_S seconds pass after that time, the watchdog prints KIDLE_STOP and
# stops the pod in the same way. So the idle stop works while the main loop
# sleeps, waits for a read of the queue, or runs a job that hangs.
# $KWORK/current names the results that the main loop makes now.
hard=${KHARD_UNTIL:-$(($(date +%s) + (INSTALL_MIN + GLOBAL_MIN + 20) * 60))}
put "$KWORK/current" "$KOUT $KNAME"
(
    why=""
    while [ -z "$why" ]; do
        t=$(date +%s) busy=""
        if [ "$t" -ge "$hard" ]; then
            why=hard
            echo "KWATCHDOG $(now): KHARD_UNTIL passed"
        elif [ "$KLOOP" = 1 ] && read -r busy 2>/dev/null <"$KWORK/busy_until" &&
            [ -n "$busy" ] && [ $((t - busy)) -ge "$KIDLE_S" ]; then
            why=idle
            echo "KIDLE_STOP $(now): no job for $((t - busy)) s (KIDLE_S=$KIDLE_S)"
        else
            sleep 10
        fi
    done
    touch "$KWORK/stopping"
    read -r kout name <"$KWORK/current"
    if [ ! -e "$kout/emitted" ]; then KOUT=$kout KNAME=$name-watchdog emit_results; fi
    if [ "$why" = idle ]; then echo "KDONE $RUN_ID $(now) idle stop"; fi
    stop_loop
) &
WATCHDOG=$!

# --- Start guards. A pod runs the stages once. ---
# RunPod clears the container disk when the pod stops or restarts (concept
# container-disk.what-survives). So the time is the main guard, and the
# marker file is a second check. A start that runs no stage sends its
# results under a new run name, so collect.py does not mix them with the
# results of the first start.
if [ -n "${KSTART_BY:-}" ] && [ "$(date +%s)" -gt "$KSTART_BY" ]; then
    echo "KBOOT late start: KSTART_BY passed. No stage runs."
    KNAME=$RUN_ID-late-$(date +%s)
    exit 4
fi
if [ -e "$KWORK/started" ]; then
    echo "KBOOT restart: an earlier start ran the stages. No stage runs."
    KNAME=$RUN_ID-restart-$(date +%s)
    exit 4
fi
touch "$KWORK/started"

# --- Hardware probe (plan 8.7 step 3). Names of RUNPOD_* only, no values. ---
{
    echo "date: $(now)"
    echo "session: $SESSION file=$SESSION_FILE"
    echo "run_id: $RUN_ID keras_sha: ${KERAS_SHA:-} mode: $KMODE dryrun: $KDRYRUN"
    echo "uname: $(uname -srm)"
    grep PRETTY_NAME /etc/os-release
    echo "nproc: $(nproc)"
    echo "memory: $(free -g | sed -n 2p)"
    echo "disk: $(df -h "$KWORK" | tail -1)"
    echo "python3: $(python3 -V 2>&1)"
    echo "pythons: $(ls /usr/bin /usr/local/bin 2>/dev/null | grep -E '^python3\.[0-9]+$' | sort -u | tr '\n' ' ')"
    echo "uv: $(uv --version 2>&1)"
    if command -v runpodctl >/dev/null 2>&1; then
        echo "runpodctl: $(command -v runpodctl) $(timeout 30 runpodctl version 2>&1 | head -1)"
    else
        echo "runpodctl: absent"
    fi
    echo "curl: $(command -v curl || echo absent)"
    echo "git: $(command -v git || echo absent)"
    echo "RUNPOD_ variables: $(env | cut -d= -f1 | grep '^RUNPOD_' | sort | tr '\n' ' ')"
    echo "CUDA_VERSION: ${CUDA_VERSION:-unset}"
    echo "times: KSTART_BY=${KSTART_BY:-unset} KSTAGES_UNTIL=${KSTAGES_UNTIL:-unset} KHARD_UNTIL=$hard grace_s=$KGRACE_S"
    echo "loop: $KLOOP idle_s=$KIDLE_S poll_s=$KPOLL_S branch=$KQUEUE_BRANCH"
    if command -v nvidia-smi >/dev/null 2>&1; then
        timeout 30 nvidia-smi --format=csv,noheader \
            --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem ||
            echo "nvidia-smi: rc $?"
        timeout 30 nvidia-smi || echo "nvidia-smi: rc $?"
    else
        echo "nvidia-smi: absent"
    fi
    echo "keras tree: $(ls "$KSRC" | tr '\n' ' ')"
} >"$KOUT/probe.txt" 2>&1
sed 's/^/KPROBE /' "$KOUT/probe.txt"

# The size cap of stream-pod-logs: KFILL lines of 4000 characters (header
# line logcap_kb). S0 measured the cap (about 240 KB per read, SESSION.md
# section 2.3), so no session file has the line now.
if [ "$LOGCAP_KB" -gt 0 ]; then
    body=$(printf '%4000s' '' | tr ' ' 'x')
    count=$((LOGCAP_KB / 4))
    for ((i = 1; i <= count; i++)); do echo "KFILL $i/$count $body"; done
fi

# --- Venv installs (plan 8.7 step 4, plan 8.4): parallel, one uv cache. ---
if ! command -v uv >/dev/null 2>&1; then
    echo "KFAIL env: uv is not installed"
    exit 2
fi

venv_python() { if [ "$1" = jax-ci ]; then echo 3.11; else echo 3.12; fi; }

venv_packages() {
    case "$1:$KMODE" in
        jax-ci:gpu) echo "jax[cuda13]==$JAX_CI" ;;
        jax-ci:cpu) echo "jax==$JAX_CI" ;;
        jax-new:gpu) echo "jax[cuda13]==$JAX_NEW" ;;
        jax-new:cpu) echo "jax==$JAX_NEW" ;;
        jax-floor:gpu) echo "jax[cuda13]==$JAX_FLOOR" ;;
        jax-floor:cpu) echo "jax==$JAX_FLOOR" ;;
        torch:gpu) echo "--index-url $TORCH_CUDA_INDEX --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==$TORCH triton==$TRITON" ;;
        torch:cpu) echo "--index-url $TORCH_CPU_INDEX --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==$TORCH triton==$TRITON" ;;
        tf:gpu) echo "tensorflow[and-cuda]==$TF" ;;
        tf:cpu) echo "tensorflow==$TF" ;;
        *) echo "UNKNOWN-VENV-$1" ;;
    esac
}

venv_backend() {
    case "$1" in
        jax-*) echo jax ;;
        torch) echo torch ;;
        tf) echo tensorflow ;;
        *) echo "" ;;
    esac
}

# One time limit (install_timeout) covers the two steps and a second try.
# The first try gets two thirds of it. A try that fails (for example on a
# PyPI network error) or stops at its limit (a download that stalls) gets
# one more try in the time that is left; the uv cache keeps the wheels
# that downloaded, so it is fast. The installs do not get the key.
install_venv() {  # name
    local name=$1 start left rc=124 try=0
    start=$(date +%s)
    while [ "$try" -lt 2 ] && [ "$rc" != 0 ]; do
        if [ "$try" = 0 ]; then
            left=$((INSTALL_MIN * 40))
        else
            left=$((start + INSTALL_MIN * 60 - $(date +%s)))
            if [ "$left" -lt 60 ]; then break; fi
        fi
        try=$((try + 1))
        echo "== try $try, limit ${left}s" >>"$KOUT/install-$name.log"
        rm -rf "${VENVS:?}/$name"
        env -u RUNPOD_API_KEY timeout -k 30 "$left" bash -c "uv venv --python $(venv_python "$name") $VENVS/$name && uv pip install --python $VENVS/$name/bin/python $(venv_packages "$name") $COMMON" \
            >>"$KOUT/install-$name.log" 2>&1
        rc=$?
    done
    echo "$rc $(($(date +%s) - start)) $try" >"$KOUT/install-$name.rc"
}

declare -A VENV_OK=()
install_start=$(date +%s)
install_pids=()
for name in $VENV_LIST; do
    install_venv "$name" &
    install_pids+=($!)
done
# Wait for the install jobs only. A bare `wait` also waits for the watchdog.
if [ "${#install_pids[@]}" -gt 0 ]; then wait "${install_pids[@]}"; fi
for name in $VENV_LIST; do
    rc=1 seconds=0 tries=0
    if [ -f "$KOUT/install-$name.rc" ]; then read -r rc seconds tries <"$KOUT/install-$name.rc"; fi
    py=$VENVS/$name/bin/python
    if [ "$rc" = 0 ]; then
        VENV_OK[$name]=1
        "$py" - "$name" "$rc" "$seconds" "$tries" >>"$KOUT/records.jsonl" <<'EOF'
import importlib.metadata as md, json, platform, sys
names = ["jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt", "torch",
         "triton", "tensorflow", "numpy", "ml-dtypes", "pytest"]
versions = {}
for name in names:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        pass
print(json.dumps({"level": "install", "venv": sys.argv[1], "rc": int(sys.argv[2]),
                  "seconds": int(sys.argv[3]), "tries": int(sys.argv[4]),
                  "python": platform.python_version(), "versions": versions}))
EOF
        echo "KINSTALL $name rc=0 seconds=$seconds tries=$tries $(tail -1 "$KOUT/records.jsonl")"
    else
        printf '{"level":"install","venv":"%s","rc":%s,"seconds":%s,"tries":%s}\n' "$name" "$rc" "$seconds" "$tries" >>"$KOUT/records.jsonl"
        echo "KINSTALL $name rc=$rc seconds=$seconds tries=$tries"
        # SESSION.md, section 5 step 9: delete the pod at once only when
        # every remaining stage or job needs this venv. Otherwise the session
        # continues, and the stages of this venv skip.
        echo "KFAIL install $name"
        tail -n 30 "$KOUT/install-$name.log" | klog "install-$name"
    fi
done
echo "KINSTALL all seconds=$(($(date +%s) - install_start))"

# --- Stages (plan 8.7 steps 2, 5 and 6). ---
print_kres() {  # bench.jsonl stage
    [ -s "$1" ] || return 0
    python3 - "$1" "$2" <<'EOF'
import json, os, sys
keys = ("level", "mode", "dtype", "shape", "m", "layers", "candidate", "timing",
        "min_ms", "median_ms", "err", "pct_copy", "skipped")
job = os.environ.get("KJOB_ID")
for text in open(sys.argv[1]):
    try:
        record = json.loads(text)
    except ValueError:
        continue
    line = {"stage": sys.argv[2]}
    if job:
        line["job"] = job
    line.update({k: record[k] for k in keys if record.get(k) is not None})
    if "skipped" in line:
        line["skipped"] = str(line["skipped"])[:120]
    print("KRES " + json.dumps(line, separators=(",", ":")))
EOF
}

# The stage deadline: `minutes` from now, but never after KSTAGES_UNTIL.
# So a slow image pull or install cuts the stages, not the read time.
declare -A PASSED=() FAILED=()
set_deadline() {  # minutes
    deadline=$(($(date +%s) + $1 * 60))
    if [ -n "${KSTAGES_UNTIL:-}" ] && [ "$KSTAGES_UNTIL" -lt "$deadline" ]; then
        deadline=$KSTAGES_UNTIL
    fi
    echo "KSTAGE deadline $(date -u -d "@$deadline" +%FT%TZ)"
}

# The GPU clocks at the start of each job, as a record of the job results.
# The ncu commands of prof.py use --clock-control none, so no stage locks the
# clocks. KWARN when the applications clocks setting limits the SM clock
# (event reason bit 0x2), or when the SM clock is below its maximum and the
# GPU is not idle (bit 0x1 clear). An idle GPU has a low SM clock.
record_clocks() {
    local clocks="no nvidia-smi" reasons=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        clocks=$(timeout 30 nvidia-smi --format=csv,noheader,nounits \
            --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem 2>&1 | head -1)
        reasons=$(timeout 30 nvidia-smi --format=csv,noheader \
            --query-gpu=clocks_event_reasons.active 2>&1 | head -1)
    fi
    python3 - "$KOUT/records.jsonl" "$clocks" "$reasons" <<'EOF'
import json, sys
path, clocks, reasons = sys.argv[1:]
record = {"level": "clocks", "query": clocks[:200], "reasons": reasons[:200]}
try:
    sm, mem, max_sm, max_mem = (int(float(v)) for v in clocks.split(","))
    record.update(sm_mhz=sm, mem_mhz=mem, max_sm_mhz=max_sm, max_mem_mhz=max_mem)
    bits = int(reasons, 16)
    if bits & 2 or (not bits & 1 and sm < max_sm):
        record["warn"] = f"SM clock {sm} of {max_sm} MHz, event reasons {reasons}"
        print("KWARN clocks: " + record["warn"], flush=True)
except ValueError:
    pass
with open(path, "a") as f:
    f.write(json.dumps(record) + "\n")
EOF
}

# The route gate of a `time` or `prof` stage. Prints the reason for a skip,
# or nothing. A route `a+b` needs a passed check of each part, and no failed
# one. The parts `base` and `harness` need no check.
route_gate() {  # route
    local part
    for part in ${1//+/ }; do
        case $part in base | harness) continue ;; esac
        if [ -n "${FAILED[$part]:-}" ]; then
            echo "route $part failed a correctness stage"
            return
        fi
        if [ -z "${PASSED[$part]:-}" ]; then
            echo "route $part has no passed correctness stage"
            return
        fi
    done
}

run_stage() {  # kind name route venv mode minutes command
    local kind=$1 name=$2 route=$3 venv=$4 mode=$5 minutes=$6 cmd=$7
    local dir=$KOUT/stages/$name left limit start rc status seconds backend path
    local reason=""
    # Check the minutes and the command before the arithmetic below:
    # - "5m", "08" or a word (for example $BENCH when a field is missing)
    #   causes an arithmetic error. That error stops the job loop.
    # - 0 gives `timeout 0`, which has no limit.
    # - An empty command passes, because `bash -c ""` exits 0.
    # This check comes first, so that the dry run also shows a bad gpu line.
    if ! [[ $minutes =~ ^[1-9][0-9]*$ ]] || [ -z "$cmd" ]; then
        skip_stage "$name" "$kind" "$route" "$venv" "$mode" "bad minutes or no command"
        return
    fi
    if [ -e "$dir" ]; then
        skip_stage "$name" "$kind" "$route" "$venv" "$mode" "duplicate stage name"
        return
    fi
    if [ "$mode" != any ] && [ "$mode" != "$KMODE" ]; then
        skip_stage "$name" "$kind" "$route" "$venv" "$mode" "mode $mode, run mode $KMODE"
        return
    fi
    if [ "$venv" != - ] && [ -z "${VENV_OK[$venv]:-}" ]; then
        skip_stage "$name" "$kind" "$route" "$venv" "$mode" "venv $venv is not ready"
        return
    fi
    # The route gate. The dry run has no route gate for timing and profiling
    # stages, so it runs each of their command lines once.
    if [ "$kind" != check ] && [ "$KDRYRUN" != 1 ]; then
        reason=$(route_gate "$route")
        if [ -n "$reason" ]; then
            skip_stage "$name" "$kind" "$route" "$venv" "$mode" "$reason"
            return
        fi
    fi
    left=$((deadline - $(date +%s)))
    if [ "$left" -le 0 ]; then
        skip_stage "$name" "$kind" "$route" "$venv" "$mode" "stage deadline"
        return
    fi
    limit=$((minutes * 60))
    if [ "$limit" -gt "$left" ]; then limit=$left; fi
    if [ "$KDRYRUN" = 1 ] && [ "$kind" != check ]; then cmd="$cmd $SMOKE"; fi
    backend=$(venv_backend "$venv")
    path=$PATH
    if [ "$venv" != - ]; then path=$VENVS/$venv/bin:$PATH; fi
    mkdir -p "$dir"
    echo "KSTAGE begin $name kind=$kind route=$route venv=$venv limit=${limit}s $(now)"
    start=$(date +%s)
    (
        cd "$KSRC" || exit 99
        # The stage commands do not need the key. A profiler report (nsys,
        # ncu) can hold the environment, and the results go to the log.
        unset RUNPOD_API_KEY
        # KSTAGE_DEADLINE: the UTC second at which the timeout ends the
        # stage. prof.py keeps its installs inside it.
        export PATH=$path KSTAGE_OUT=$dir KSTAGE_DEADLINE=$((start + limit))
        export BENCH="python -m benchmarks.quantized_matmul.bench --out=$dir/bench.jsonl"
        export PROF="python -m benchmarks.quantized_matmul.pod.prof"
        if [ -n "$backend" ]; then
            export KERAS_BACKEND=$backend
            if [ -d "$KSRC/.github/workflows/config/$backend" ]; then
                export KERAS_HOME=$KSRC/.github/workflows/config/$backend
            fi
        fi
        exec timeout -k 30 "$limit" bash -c "$cmd"
    ) >"$dir/log.txt" 2>&1 </dev/null
    rc=$?
    seconds=$(($(date +%s) - start))
    case $rc in
        0) status=pass ;;
        124 | 137) status=timeout ;;
        *) status=fail ;;
    esac
    # pytest exits 0 when it skips every test. Such a check proves nothing.
    if [ "$kind" = check ] && [ "$status" = pass ] && [[ $cmd == *pytest* ]] &&
        ! grep -qE '(^|[ =])[0-9]+ passed' "$dir/log.txt"; then
        status=fail reason="no test passed"
    fi
    echo "KSTAGE end $name status=$status rc=$rc seconds=$seconds $reason"
    if [ "$status" != pass ]; then
        tail -n 40 "$dir/log.txt" | klog "$name"
    fi
    if [ "$kind" = check ]; then
        if [ "$status" = pass ]; then
            PASSED[$route]=1
        else
            FAILED[$route]=1
            echo "KFAIL $route $name status=$status rc=$rc $reason"
            # A failed device check of a venv stops every later stage of it.
            if [ "$route" = env ] && [ "$venv" != - ]; then unset "VENV_OK[$venv]"; fi
        fi
    fi
    if [ "$kind" = time ]; then print_kres "$dir/bench.jsonl" "$name"; fi
    if [ "$kind" = prof ]; then grep -a '^KPROF ' "$dir/log.txt" | head -n 200; fi
    record_stage "$name" "$kind" "$route" "$venv" "$mode" "$status" "$rc" "$seconds" "$reason"
}

# Runs the stage lines of a file whose kind is in `kinds`, in file order.
run_stages() {  # file kinds...
    local file=$1 kinds=" ${*:2} " line first kind name route venv mode minutes cmd
    while IFS= read -r line || [ -n "$line" ]; do
        read -r first kind _ <<<"$line"
        case "$first" in
            check | time | prof)
                [[ $kinds == *" $first "* ]] || continue
                read -r kind name route venv mode minutes cmd <<<"$line"
                run_stage "$kind" "$name" "$route" "$venv" "$mode" "$minutes" "$cmd"
                ;;
            TODO)
                [[ $kinds == *" $kind "* ]] || continue
                read -r _ kind name route venv mode _ <<<"$line"
                echo "KTODO $name: the command is not final. ${line:0:300}"
                record_stage "$name" "$kind" "$route" "$venv" "$mode" todo null 0 "command not final"
                ;;
        esac
    done <"$file"
}

grace() {
    if [ "$KGRACE_S" -gt 0 ]; then
        echo "KSTOP grace ${KGRACE_S}s: the self-stop starts after it"
        sleep "$KGRACE_S"
    fi
}

if [ "$KLOOP" = 0 ]; then
    record_clocks
    set_deadline "$GLOBAL_MIN"
    run_stages "$SESSION_FILE" check time prof
    emit_results
    echo "KDONE $RUN_ID $(now)"
    grace
    exit 0
fi

# --- Loop mode (plan 8.9, SESSION.md section 10). ---
# job.txt holds the lines `job <id>`, `run <RUN_ID>`, `sha <SHA>`, an
# optional `timeout <minutes>` (default: global_timeout), and stage lines.
POLL_FILE=$KWORK/poll/job.txt
job_field() { awk -v k="$1" '$1 == k {print $2; exit}' "$2"; }

# Reads job.txt at the head commit of the queue branch: the head from the
# ref list, then job.txt from the tarball of that commit. Sets POLL_STATE to
# new, seen (the job ran), other-run (a job of an other RUN_ID), bad or
# error. Nothing here sends a key.
# A commit does not change. So when the head is the same as at the last
# tarball read, and that read gave seen, other-run or bad, the state is the
# same: the poll does not read the tarball again. With several pods, each
# read counts against the GitHub rate limits.
LAST_HEAD="" LAST_STATE="" LAST_ID="" LAST_WHY=""
poll_queue() {
    local head="" id="" sha="" minutes="" why=""
    POLL_STATE=error
    mkdir -p "$KWORK/poll"
    rm -f "$POLL_FILE"
    if ! curl -fsS --max-time 30 -o "$KWORK/poll/refs" "$KQUEUE_REFS" 2>/dev/null; then
        why="the ref list did not download"
    else
        head=$(grep -ao "[0-9a-f]\{40\}[[:space:]]refs/heads/$KQUEUE_BRANCH\$" "$KWORK/poll/refs" |
            head -1 | cut -c1-40)
        if [ -z "$head" ]; then
            why="no branch $KQUEUE_BRANCH"
        elif [ "$head" = "$LAST_HEAD" ] && [[ $LAST_STATE =~ ^(seen|other-run|bad)$ ]]; then
            POLL_STATE=$LAST_STATE id=$LAST_ID
            why="${LAST_WHY:+$LAST_WHY; }head unchanged, tarball not read"
        elif ! (set -o pipefail; curl -fsSL --max-time 60 "$KARCHIVE/$head.tar.gz" |
            tar -xzO --wildcards '*/job.txt') >"$POLL_FILE" 2>/dev/null; then
            why="no job.txt at the head commit, or not yet"
        else
            id=$(job_field job "$POLL_FILE") sha=$(job_field sha "$POLL_FILE")
            minutes=$(job_field timeout "$POLL_FILE")
            # No leading zero: bash reads "08" as a bad octal number.
            if ! [[ $id =~ ^[A-Za-z0-9._-]+$ && $sha =~ ^[0-9a-f]{40}$ && ${minutes:-1} =~ ^[1-9][0-9]*$ ]]; then
                POLL_STATE=bad why="bad job, sha or timeout line"
            elif [ "$(job_field run "$POLL_FILE")" != "$RUN_ID" ]; then
                POLL_STATE=other-run
            elif [ -e "$KWORK/jobs/$id" ]; then
                POLL_STATE=seen
            else
                POLL_STATE=new
            fi
            LAST_HEAD=$head LAST_STATE=$POLL_STATE LAST_ID=$id LAST_WHY=$why
        fi
    fi
    echo "KPOLL $polls $(now) head=${head:0:12} job=${id:--} state=$POLL_STATE${why:+: $why}"
}

download() {  # sha folder log. Three tries, 20 s apart: a new commit can take some seconds to appear.
    local try
    for try in 1 2 3; do
        rm -rf "$2" && mkdir -p "$2" &&
            (set -o pipefail; curl -fsSL --max-time 600 "$KARCHIVE/$1.tar.gz" |
                tar -xz --strip-components=1 -C "$2") >>"$3" 2>&1 && return 0
        echo "download try $try failed" >>"$3"
        sleep 20
    done
    return 1
}

# Runs the job of POLL_FILE: the tarball of its SHA in a new folder, the
# correctness stages, then the timing and profiling stages, then the results
# under the name of the job id, then `KJOB <id> DONE`.
run_job() {
    local id sha minutes start dir status=DONE
    id=$(job_field job "$POLL_FILE") sha=$(job_field sha "$POLL_FILE")
    minutes=$(job_field timeout "$POLL_FILE")
    minutes=${minutes:-$GLOBAL_MIN}
    dir=$KWORK/jobs/$id
    mkdir -p "$dir/out/stages"
    cp "$POLL_FILE" "$dir/job.txt"
    KJOB_ID=$id KNAME=$id KOUT=$dir/out KSRC=$dir/src
    export KJOB_ID KERAS_SHA=$sha
    PASSED=() FAILED=()
    put "$KWORK/current" "$KOUT $KNAME"
    start=$(date +%s)
    echo "KJOB $id START sha=$sha timeout=${minutes}m $(now)"
    set_deadline "$minutes"
    put "$KWORK/busy_until" $((deadline + 120))
    record_clocks
    if download "$sha" "$KSRC" "$dir/download.log"; then
        run_stages "$dir/job.txt" check
        run_stages "$dir/job.txt" time prof
    else
        status="FAIL download"
        tail -n 20 "$dir/download.log" | klog download
    fi
    printf '{"level":"job","job":"%s","sha":"%s","status":"%s","seconds":%s}\n' \
        "$id" "$sha" "$status" $(($(date +%s) - start)) >>"$KOUT/records.jsonl"
    emit_results
    echo "KJOB $id $status $(now)"
    put "$KWORK/busy_until" "$(date +%s)"
}

# Job 0: the stages of the session file, on the tree of KERAS_SHA, in file
# order. Its results also hold the probe and the installs. The job folder
# marks the id as run, so a pushed job with the id <RUN_ID>-j0 shows `seen`
# and does not send a second result stream under the same name.
KJOB_ID=$RUN_ID-j0 KNAME=$RUN_ID-j0
export KJOB_ID
mkdir -p "$KWORK/jobs/$KJOB_ID"
put "$KWORK/current" "$KOUT $KNAME"
echo "KJOB $KJOB_ID START sha=${KERAS_SHA:-} source=$(basename "$SESSION_FILE") $(now)"
set_deadline "$GLOBAL_MIN"
put "$KWORK/busy_until" $((deadline + 120))
record_clocks
run_stages "$SESSION_FILE" check time prof
emit_results
echo "KJOB $KJOB_ID DONE $(now)"
put "$KWORK/busy_until" "$(date +%s)"

# No job starts after KSTAGES_UNTIL. The watchdog does the idle stop.
loop_until=${KSTAGES_UNTIL:-$((hard - KGRACE_S - 120))}
polls=0
while true; do
    if [ -e "$KWORK/stopping" ]; then
        echo "KLOOP end $(now): the watchdog stops the pod"
        wait "$WATCHDOG"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$loop_until" ]; then
        echo "KLOOP end $(now): KSTAGES_UNTIL passed. No job starts after it."
        break
    fi
    polls=$((polls + 1))
    poll_queue
    if [ "$POLL_STATE" = new ] && [ ! -e "$KWORK/stopping" ]; then run_job; fi
    sleep "$KPOLL_S"
done
echo "KDONE $RUN_ID $(now)"
grace
exit 0
