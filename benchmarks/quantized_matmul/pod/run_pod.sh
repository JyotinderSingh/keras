#!/bin/bash
# run_pod.sh: the pod bootstrap of the quantized-matmul measurement campaign.
# Plan: quant-tooling/audits/kernel-plan-2026-10-21/plan_v3.md, section 8.7.
# The checklist and the session file format are in SESSION.md.
#
# Usage: run_pod.sh <session file>
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
#                route gate for timing stages, no grace time, no self-stop.
#   KCHUNK_SLEEP The pause after each result chunk, in seconds. Default 0.1.
#   KSRC         The Keras source tree. Default: three folders above this file.
#   KWORK        The work folder. Default: /kpod/work.
#   KRESTART_PROBE_UNTIL  S0 only. UTC seconds. Before this time, the first
#                start exits once with code 3, before the trap is set. The
#                KBOOT lines then show what the pod does after an exit.
#   RUNPOD_POD_ID, RUNPOD_API_KEY  RunPod sets them in each pod. The self-stop
#                uses them. This script never prints the key.
#
# Log markers, one per line, at the start of the line:
#   KBOOT KPROBE KFILL KINSTALL KSTAGE KTODO KSKIP KFAIL KLOG KRES
#   KBEGIN KCHUNK KEND KDONE KEXIT KSTOP KWATCHDOG
#
# Steps (plan 8.7): exit trap with self-stop; watchdog; start guards; stage
# deadline and per-stage timeouts; hardware probe; parallel venv installs;
# correctness stages; timing stages; the results as gzip+base64 chunks;
# KDONE; grace; self-stop.

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
# The dry run appends these flags to each timing stage. The last value of a
# flag wins, so they replace the grid of the stage.
SMOKE="--shapes=256x512 --m=1,16 --logits_shapes=tiny-tied,tiny-untied"
SMOKE="$SMOKE --logits_m=1,16 --models=tiny --warmup=2 --calls=5"
SMOKE="$SMOKE --copy_mb=64 --l2_mb=0 --max_chain_layers=8"

SESSION_FILE=$(readlink -f "${1:?usage: run_pod.sh <session file>}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KSRC=${KSRC:-$(cd "$HERE/../../.." && pwd)}
KWORK=${KWORK:-/kpod/work}
KMODE=${KMODE:-gpu}
KDRYRUN=${KDRYRUN:-0}
RUN_ID=${RUN_ID:-norun}
KOUT=$KWORK/out
VENVS=$KWORK/venvs
export UV_CACHE_DIR=$KWORK/uv-cache
export KCHUNK_SLEEP=${KCHUNK_SLEEP:-0.1}
export KMODE RUN_ID KOUT
if [ "$KMODE" = cpu ]; then
    export TRITON_INTERPRET=1 JAX_PLATFORMS=cpu
fi

now() { date -u +%FT%TZ; }

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
        0) STOP_VIA="runpodctl pod stop"
            command -v runpodctl >/dev/null 2>&1 &&
                timeout 60 runpodctl pod stop "$RUNPOD_POD_ID" ;;
        1) STOP_VIA="runpodctl stop pod"
            command -v runpodctl >/dev/null 2>&1 &&
                timeout 60 runpodctl stop pod "$RUNPOD_POD_ID" ;;
        2) STOP_VIA="REST v2 POST /v2/pods/{id}/action"
            timeout 60 curl -fsS -X POST -o /dev/null \
                -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
                -H "Content-Type: application/json" -d '{"action":"stop"}' \
                "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" ;;
        3) STOP_VIA="REST v1 POST /v1/pods/{id}/stop"
            timeout 60 curl -fsS -X POST -o /dev/null \
                -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
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
    if [ "$KDRYRUN" = 1 ]; then
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
EMITTED=0 WATCHDOG=""
finish() {
    local code=$?
    if [ -n "$WATCHDOG" ]; then kill "$WATCHDOG" 2>/dev/null; fi
    if [ "$EMITTED" = 0 ] && [ -d "$KOUT" ]; then emit_results; fi
    echo "KEXIT $code"
    stop_loop
}

# --- The results: gzip, base64, chunks of at most 4 KB (plan 8.7 step 7). ---
# One log second holds at most 1 / KCHUNK_SLEEP chunks. So a `since` read of
# stream-pod-logs always gets past a read that the byte cap cuts.
emit_results() {
    EMITTED=1
    python3 - "$KOUT" "$RUN_ID" <<'EOF'
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

# --- Session header (first pass over the file). ---
SESSION=unnamed VENV_LIST="" INSTALL_MIN=15 GLOBAL_MIN=60 L2_MB=0 LOGCAP_KB=0
KGRACE_S=300
while read -r key value _; do
    case "$key" in
        session) SESSION=$value ;;
        venvs) VENV_LIST=$(sed -n 's/^venvs[[:space:]]*//p' "$SESSION_FILE" | head -1) ;;
        install_timeout) INSTALL_MIN=$value ;;
        global_timeout) GLOBAL_MIN=$value ;;
        grace_s) KGRACE_S=$value ;;
        l2_mb) L2_MB=$value ;;
        logcap_kb) LOGCAP_KB=$value ;;
    esac
done <"$SESSION_FILE"
export KL2_MB=$L2_MB
if [ "$KDRYRUN" = 1 ]; then KGRACE_S=0; fi

mkdir -p "$KOUT/stages" "$VENVS"
trap finish EXIT

# --- The watchdog (plan 8.7 step 2). ---
# Some hangs have no timeout: nvidia-smi on a GPU that fell off the bus, or
# a process in D state that `timeout -k` cannot kill. Then the trap does not
# run. At KHARD_UNTIL the watchdog sends the results so far (if the main
# loop did not send them) and runs the stop loop. It does not use the main
# loop. The trap stops the watchdog.
hard=${KHARD_UNTIL:-$(($(date +%s) + (INSTALL_MIN + GLOBAL_MIN + 20) * 60))}
(
    while [ "$(date +%s)" -lt "$hard" ]; do sleep 10; done
    echo "KWATCHDOG $(now): KHARD_UNTIL passed"
    if [ ! -e "$KOUT/emitted" ]; then RUN_ID=$RUN_ID-watchdog emit_results; fi
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
    RUN_ID=$RUN_ID-late-$(date +%s)
    exit 4
fi
if [ -e "$KWORK/started" ]; then
    echo "KBOOT restart: an earlier start ran the stages. No stage runs."
    RUN_ID=$RUN_ID-restart-$(date +%s)
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

# The size cap of stream-pod-logs (S0): KFILL lines of 4000 characters.
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

install_venv() {  # name
    local name=$1 start rc
    start=$(date +%s)
    # One timeout for the two steps.
    timeout -k 30 "$((INSTALL_MIN * 60))" bash -c "uv venv --python $(venv_python "$name") $VENVS/$name && uv pip install --python $VENVS/$name/bin/python $(venv_packages "$name") $COMMON" \
        >"$KOUT/install-$name.log" 2>&1
    rc=$?
    echo "$rc $(($(date +%s) - start))" >"$KOUT/install-$name.rc"
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
    rc=1 seconds=0
    if [ -f "$KOUT/install-$name.rc" ]; then read -r rc seconds <"$KOUT/install-$name.rc"; fi
    py=$VENVS/$name/bin/python
    if [ "$rc" = 0 ]; then
        VENV_OK[$name]=1
        "$py" - "$name" "$rc" "$seconds" >>"$KOUT/records.jsonl" <<'EOF'
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
                  "seconds": int(sys.argv[3]), "python": platform.python_version(),
                  "versions": versions}))
EOF
        echo "KINSTALL $name rc=0 seconds=$seconds $(tail -1 "$KOUT/records.jsonl")"
    else
        printf '{"level":"install","venv":"%s","rc":%s,"seconds":%s}\n' "$name" "$rc" "$seconds" >>"$KOUT/records.jsonl"
        echo "KINSTALL $name rc=$rc seconds=$seconds"
        echo "KFAIL venv:$name"
        tail -n 30 "$KOUT/install-$name.log" | sed "s/^/KLOG install-$name | /"
    fi
done
echo "KINSTALL all seconds=$(($(date +%s) - install_start))"

# --- Stages (plan 8.7 steps 2, 5 and 6). ---
print_kres() {  # bench.jsonl stage
    [ -s "$1" ] || return 0
    python3 - "$1" "$2" <<'EOF'
import json, sys
keys = ("level", "mode", "dtype", "shape", "m", "layers", "candidate", "timing",
        "min_ms", "median_ms", "err", "pct_copy", "skipped")
for text in open(sys.argv[1]):
    try:
        record = json.loads(text)
    except ValueError:
        continue
    line = {"stage": sys.argv[2]}
    line.update({k: record[k] for k in keys if record.get(k) is not None})
    if "skipped" in line:
        line["skipped"] = str(line["skipped"])[:120]
    print("KRES " + json.dumps(line, separators=(",", ":")))
EOF
}

# The stage deadline: global_timeout from now, but never after KSTAGES_UNTIL.
# So a slow image pull or install cuts the stages, not the read time.
declare -A PASSED=() FAILED=()
deadline=$(($(date +%s) + GLOBAL_MIN * 60))
if [ -n "${KSTAGES_UNTIL:-}" ] && [ "$KSTAGES_UNTIL" -lt "$deadline" ]; then
    deadline=$KSTAGES_UNTIL
fi
echo "KSTAGE deadline $(date -u -d "@$deadline" +%FT%TZ)"

run_stage() {  # kind name route venv mode minutes command
    local kind=$1 name=$2 route=$3 venv=$4 mode=$5 minutes=$6 cmd=$7
    local dir=$KOUT/stages/$name left limit start rc status seconds backend path
    local reason=""
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
    # The route gate. The dry run has no route gate for timing stages, so
    # it runs each timing command line once.
    if [ "$kind" = time ] && [ "$route" != base ] && [ "$route" != harness ] &&
        [ "$KDRYRUN" != 1 ]; then
        if [ -n "${FAILED[$route]:-}" ]; then
            skip_stage "$name" "$kind" "$route" "$venv" "$mode" "route $route failed a correctness stage"
            return
        fi
        if [ -z "${PASSED[$route]:-}" ]; then
            skip_stage "$name" "$kind" "$route" "$venv" "$mode" "route $route has no passed correctness stage"
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
    if [ "$KDRYRUN" = 1 ] && [ "$kind" = time ]; then cmd="$cmd $SMOKE"; fi
    backend=$(venv_backend "$venv")
    path=$PATH
    if [ "$venv" != - ]; then path=$VENVS/$venv/bin:$PATH; fi
    mkdir -p "$dir"
    echo "KSTAGE begin $name kind=$kind route=$route venv=$venv limit=${limit}s $(now)"
    start=$(date +%s)
    (
        cd "$KSRC" || exit 99
        export PATH=$path KSTAGE_OUT=$dir
        export BENCH="python -m benchmarks.quantized_matmul.bench --out=$dir/bench.jsonl"
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
        tail -n 40 "$dir/log.txt" | sed "s/^/KLOG $name | /"
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
    record_stage "$name" "$kind" "$route" "$venv" "$mode" "$status" "$rc" "$seconds" "$reason"
}

while IFS= read -r line || [ -n "$line" ]; do
    read -r first _ <<<"$line"
    case "$first" in
        check | time)
            read -r kind name route venv mode minutes cmd <<<"$line"
            run_stage "$kind" "$name" "$route" "$venv" "$mode" "$minutes" "$cmd"
            ;;
        TODO)
            read -r _ kind name route venv mode _ <<<"$line"
            echo "KTODO $name: the command is not final. ${line:0:300}"
            record_stage "$name" "$kind" "$route" "$venv" "$mode" todo null 0 "command not final"
            ;;
    esac
done <"$SESSION_FILE"

emit_results
echo "KDONE $RUN_ID $(now)"
if [ "$KGRACE_S" -gt 0 ]; then
    echo "KSTOP grace ${KGRACE_S}s: the self-stop starts after it"
    sleep "$KGRACE_S"
fi
exit 0
