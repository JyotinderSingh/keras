"""Per-kernel split of one K1 call from an `nsys stats --report cuda_gpu_trace` CSV.

    python nsys_split.py <prefix>_cuda_gpu_trace.csv [cell label]

Each line is `KPROF {json}`, with the cell label in "cell".

For each kernel name (shortened): the count, the median duration and the
median idle gap from the end of the kernel before it (same stream), in us.
One JSON line per name, then the median sum per call pattern found in order
(`_w4a16_kernel` -> next kernels until the next `_w4a16_kernel`).
"""
import csv
import json
import statistics
import sys


def short(name):
    for key in ("_w4a16_kernel", "_splitk_reduce", "reduce_kernel",
                "elementwise_kernel", "vectorized_elementwise", "copy"):
        if key in name:
            return key
    return name[:40]


CELL = sys.argv[2] if len(sys.argv) > 2 else None
rows = []
with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    start = next(c for c in cols if c.startswith("Start"))
    dur = next(c for c in cols if c.startswith("Duration"))
    name = next(c for c in cols if c == "Name")
    strm = next((c for c in cols if c.startswith("Strm")), None)
    for r in reader:
        if not r.get(dur) or "memcpy" in r[name].lower() or "memset" in r[name].lower():
            continue
        rows.append((int(r[start]), int(r[dur]), short(r[name]), r.get(strm)))
rows.sort()
stats = {}
prev_end = {}
for s, d, n, st in rows:
    gap = s - prev_end[st] if st in prev_end else None
    prev_end[st] = s + d
    e = stats.setdefault(n, {"dur": [], "gap": []})
    e["dur"].append(d / 1e3)
    if gap is not None and gap < 50_000:
        e["gap"].append(gap / 1e3)
for n, e in stats.items():
    print("KPROF " + json.dumps({"tool": "nsys_split", "cell": CELL, "kernel": n, "count": len(e["dur"]),
                      "median_us": round(statistics.median(e["dur"]), 3),
                      "gap_before_us": round(statistics.median(e["gap"]), 3) if e["gap"] else None}))
# Calls: a K1 kernel and the kernels after it, until the next K1 kernel.
calls, cur = [], None
for s, d, n, st in rows:
    if n == "_w4a16_kernel":
        if cur:
            calls.append(cur)
        cur = [s, s + d, [n]]
    elif cur:
        cur[1] = s + d
        cur[2].append(n)
if cur:
    calls.append(cur)
by = {}
for s, e, names in calls:
    by.setdefault("+".join(names), []).append((e - s) / 1e3)
for pattern, spans in by.items():
    print("KPROF " + json.dumps({"tool": "nsys_split", "cell": CELL, "call": pattern, "count": len(spans),
                      "median_span_us": round(statistics.median(spans), 3)}))
