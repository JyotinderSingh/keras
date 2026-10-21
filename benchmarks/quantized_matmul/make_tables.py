"""Writes the Markdown tables of the JSON lines that `bench.py` writes.

    python -m benchmarks.quantized_matmul.make_tables a.jsonl [b.jsonl ...]

The tables go to stdout, one section per run (backend, device, library
versions and commit): the copy bandwidth, the error of each candidate
against the float64 oracle, the times with their two ratios and the
share of the copy bandwidth, the logits share of each decode step, and
the candidates that did not run.
"""

import collections
import json
import re
import sys

# The fields of `env` that name a run.
RUN = ("backend", "versions", "device", "commit")
PERFORMANCE = (
    "Level",
    "Mode",
    "Dtype",
    "Shape",
    "M",
    "Layers",
    "Timing",
    "Candidate",
    "Min ms",
    "Median ms",
    "x float",
    "x best form",
    "GB/s",
    "% copy",
    "Temp MiB",
    "Compile s",
    "Route",
    "Config",
)


def _fmt(value, spec=".3g"):
    return "-" if value is None else format(value, spec)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _route(record):
    """The evidence of the path that ran: the route counter, the custom
    calls, a weight-size float temp, and the top torch events."""
    hlo = record.get("hlo") or {}
    parts = sorted(record.get("route_counts") or {})
    parts += hlo.get("custom_calls", [])
    weight = (record.get("k") or 0) * (record.get("n") or 0) // 2
    if weight and hlo.get("max_float_elements", 0) >= weight:
        parts.append("weight-size temp")
    parts += list(record.get("events") or {})[:2]
    return ", ".join(parts) or "-"


def correctness(records):
    """The worst `err` over `M` and timings, the reference (F0) first."""
    errors, names = collections.defaultdict(dict), []
    for r in records:
        if r.get("err") is not None:
            cell = errors[r["level"], r["mode"], r["dtype"], r["shape"]]
            cell[r["candidate"]] = max(cell.get(r["candidate"], 0), r["err"])
            if r["candidate"] not in names:
                names.append(r["candidate"])
    names.sort(key=lambda name: (name != "F0", name != "float"))
    header = ["Level", "Mode", "Dtype", "Shape"] + [f"{n} err" for n in names]
    rows = [
        [*key, *(_fmt(cell.get(n), ".2e") for n in names)]
        for key, cell in errors.items()
    ]
    return _table(header, rows)


def performance(records):
    """Times per cell, with the ratios to the float layer and to the best
    form without a kernel. A later record of a cell (the rerun of a chain
    with more layers) replaces an earlier one."""
    cells = collections.defaultdict(dict)
    for r in records:
        if "min_ms" in r and r["level"] != "copy":
            key = r["level"], r["mode"], r["dtype"], r["shape"], r["m"]
            cells[key + (r["timing"],)][r["candidate"]] = r
    rows = []
    for key, found in cells.items():
        # A decode step has its float baseline in the cell of the float
        # decoder.
        floats = cells.get((key[0], "float") + key[2:], {})
        base = found.get("float", floats.get("float", {})).get("min_ms")
        forms = [r["min_ms"] for r in found.values() if r["kind"] == "form"]
        best = min(forms, default=None)
        for name, r in found.items():
            temp = r.get("temp_bytes") or r.get("peak_bytes")
            rows.append(
                [
                    *key[:5],
                    r.get("layers"),
                    key[5],
                    name,
                    _fmt(r["min_ms"]),
                    _fmt(r["median_ms"]),
                    _fmt(base and r["min_ms"] / base, ".2f"),
                    _fmt(best and r["min_ms"] / best, ".2f"),
                    _fmt(r.get("gbps")),
                    _fmt(r.get("pct_copy"), ".0f"),
                    _fmt(temp and temp / 2**20),
                    _fmt(r.get("compile_s"), ".2f"),
                    _route(r),
                    r.get("config") or "-",
                ]
            )
    return _table(PERFORMANCE, rows)


def logits_share(records):
    """The logits time of each decode step, as a share of the step."""
    rows = [
        [r[k] for k in ("shape", "mode", "dtype", "m", "timing", "candidate")]
        + [_fmt(r["min_ms"]), _fmt(r["logits_share"], ".0%")]
        for r in records
        if r.get("logits_share") is not None and r["candidate"] != "head"
    ]
    header = ["Model", "Mode", "Dtype", "M", "Timing", "Candidate"]
    return _table(header + ["Step ms", "Logits share"], rows)


def _reason(text):
    """A reason on one line of a table: no terminal escapes, no `|`."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", " ".join(text.split()))
    return text.replace("|", "\\|")


def skipped(records):
    """The candidates that did not run. An error of several lines goes on
    one line of the table."""
    keys = ("level", "mode", "dtype", "shape", "m", "timing", "candidate")
    rows = [
        [r.get(k) for k in keys] + [_reason(r["skipped"])]
        for r in records
        if r.get("skipped")
    ]
    header = ["Level", "Mode", "Dtype", "Shape", "M", "Timing", "Candidate"]
    return _table(header + ["Reason"], rows)


def section(env, records):
    device, versions = env["device"], env["versions"]
    lines = [f"## {env['backend']} on {device.get('name')}", ""]
    lines.append(f"- Commit: `{env.get('commit')}`; date: {env.get('date')}")
    lines.append(f"- Device: {json.dumps(device, sort_keys=True)}")
    lines.append(f"- Versions: {json.dumps(versions, sort_keys=True)}")
    copies = [r for r in records if r["level"] == "copy" and "gbps" in r]
    copy = ", ".join(f"{r['gbps']:.3g} GB/s ({r['timing']})" for r in copies)
    lines += [f"- Copy bandwidth: {copy or '-'}", ""]
    for title, table in (
        ("Error against the float64 oracle", correctness),
        ("Performance", performance),
        ("Logits share of a decode step", logits_share),
        ("Candidates that did not run", skipped),
    ):
        lines += [f"### {title}", "", table(records)]
    return "\n".join(lines)


def main(paths):
    runs = collections.defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line in filter(str.strip, f):
                record = json.loads(line)
                env = [record["env"][k] for k in RUN]
                runs[json.dumps(env, sort_keys=True)].append(record)
    for records in runs.values():
        print(section(records[0]["env"], records))


if __name__ == "__main__":
    main(sys.argv[1:])
