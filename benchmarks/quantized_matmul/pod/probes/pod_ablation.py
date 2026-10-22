"""Triton K1 kernel-body ablation for a pod job (KERNEL-BODY analysis,
2026-10-22). Times the tip717 `_w4a16_kernel` and the variants of
`variants.py` at one configuration each, the way tune.py times a call: a
CUDA graph of R calls over weight copies that total >= 4x the L2, CUDA
events, the median of the replays. Checks each variant against the tip717
kernel bit for bit (same configuration, contiguous groups) before timing.

Run from the Keras tree (tip717 or later), torch venv:
    python pod_ablation.py [--variants FILE] [--shapes 4096x4096,14336x4096]
        [--m 1,16] [--modes g_s,g_u,pc] [--replays 30]
Records: one JSON line per (variant, mode, shape, m) in
$KSTAGE_OUT/ablation.jsonl (all records) and as a `KPROF {json}` line for
the first value of --m; `KPROF {json}` summaries per (variant, mode) and per
variant (the geometric mean of tip717 / variant). Unknown flags (the dry-run
SMOKE flags of run_pod.sh) are ignored.
Fields: us (median call), us_kernel (the Triton launch alone, without the
split-K torch.sum and cast), pct_copy (the bytes of codes, scale, zero, x
and y / us / copy bandwidth), same_bits (vs tip717 at the same config).
TRITON_INTERPRET=1: tiny shapes, checks only, no timing.
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.getcwd())
from keras.src.backend.torch.ops import quantized as q  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cpu" if INTERP else "cuda"
G = 128

# name -> (kernel attribute, extra constexprs, BM, BNH, BK, warps, stages,
# split, passes NH). The tip717 best decode rows of t1a-decode-kprof-early.
ROW = (16, 32, 64, 4, 3, 8)
VARIANTS = {
    "tip717": (None, {}, *ROW, False),
    "nh": ("k_nh", {}, *ROW, True),
    "hoist": ("k_hoist", dict(MAGIC=0, UNIFORM=0, GENERAL=1), *ROW, True),
    "hoist_magic": ("k_hoist", dict(MAGIC=1, UNIFORM=0, GENERAL=1), *ROW, True),
    "fast2": ("k_fast2", dict(MAGIC=1, PREFETCH_S=1, EVEN=0, CHECK=1024,
                              GENERAL=1), *ROW, True),
    "fast2_bk128_st2": ("k_fast2", dict(MAGIC=1, PREFETCH_S=1, EVEN=0,
                                        CHECK=1024, GENERAL=1),
                        16, 32, 128, 4, 2, 8, True),
    # BK = G with the register cap of 128 (4 CTAs per SM; AOT: 146 -> 128
    # registers, 8 bytes of stack).
    "fast2_bk128_st2_r128": ("k_fast2", dict(MAGIC=1, PREFETCH_S=1, EVEN=0,
                                             CHECK=1024, GENERAL=1,
                                             maxnreg=128),
                             16, 32, 128, 4, 2, 8, True),
    "fast2_bnh64_sk4": ("k_fast2", dict(MAGIC=1, PREFETCH_S=1, EVEN=0,
                                        CHECK=1024, GENERAL=1),
                        16, 64, 64, 4, 3, 4, True),
    # Uniform groups only (no check, no general loop): the upper bound of
    # the check's cost. bfloat16 only.
    "bf16magic_uniform": ("k_bf16magic", {}, *ROW, True),
}


def load_variants(path):
    spec = importlib.util.spec_from_file_location("variants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights(k, n, mode, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    nh = (n + 1) // 2
    groups = -(-k // G)
    signed = mode in ("g_s", "pc")
    act = mode == "g_act"  # GPTQ codes with an activation-order g_idx
    if signed:
        codes = torch.randint(-128, 128, (k, nh), generator=gen, dtype=torch.int8)
        zero = torch.randint(-8, 8, (groups, n), generator=gen, dtype=torch.int8)
    else:
        codes = torch.randint(0, 256, (k, nh), generator=gen, dtype=torch.uint8)
        zero = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.uint8)
    if mode == "pc":
        scale = torch.empty(n).uniform_(50, 150, generator=gen).to(dtype)
        return [t.to(DEV) for t in (codes, scale, scale, scale)]
    scale = torch.empty(groups, n).uniform_(5e-3, 1.5e-2, generator=gen)
    g_idx = (torch.arange(k) // G).to(torch.float32)
    if act:
        g_idx = g_idx[torch.randperm(k, generator=gen)]
    return [t.to(DEV) for t in (codes, scale.to(dtype), zero, g_idx)]


def launcher(mod, name, mode, m, k, n):
    attr, extra, bm, bnh, bk, warps, stages, split, has_nh = VARIANTS[name]
    kernel = q._w4a16_kernel if attr is None else getattr(mod, attr)
    if INTERP:
        extra = {k: v for k, v in extra.items() if k != "maxnreg"}
    signed = mode in ("g_s", "pc")
    pc = mode == "pc"

    def launch(x, codes, scale, zero, g_idx, kernel_only=False):
        if split > 1:
            y = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
        else:
            y = torch.empty((m, n), dtype=x.dtype, device=x.device)
        grid = (triton_cdiv(m, bm), triton_cdiv(codes.shape[1], bnh), split)
        args = [x, codes, scale, zero, g_idx, y, m, n, k,
                scale.shape[0] if not pc else 1, G if not pc else k]
        if has_nh:
            args.append(codes.shape[1])
        ck = kernel[grid](*args, XOR=8 if signed else 0, PER_CHANNEL=pc,
                     UPCAST_DOT=False, INPUT_PRECISION="ieee", SPLIT_K=split,
                     BM=bm, BNH=bnh, BK=bk, num_warps=warps,
                     num_stages=stages, **extra)
        if ck is not None and not launch.meta:
            launch.meta.update(n_regs=getattr(ck, "n_regs", None),
                               n_spills=getattr(ck, "n_spills", None),
                               shared=getattr(getattr(ck, "metadata", None), "shared", None))
        if kernel_only or split == 1:
            return y
        return torch.sum(y, dim=0).to(x.dtype)

    launch.meta = {}
    return launch


def triton_cdiv(a, b):
    return -(-a // b)


def copy_gbps():
    a = torch.empty(2**28, dtype=torch.uint8, device=DEV)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        b.copy_(a)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    return 2 * a.numel() / statistics.median(times) / 1e9


def graph_us(fn, xs, copies, replays):
    """Median time of one call: a CUDA graph of one call per copy."""
    for c in copies[:2]:
        fn(xs, *c)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in copies:
            fn(xs, *c)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / len(copies))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "variants.py"))
    p.add_argument("--shapes", default="4096x4096,14336x4096,4096x14336,2304x9216")
    p.add_argument("--m", default="1,16")
    p.add_argument("--modes", default="g_s,g_u,pc,g_act")
    p.add_argument("--only", default="")
    p.add_argument("--replays", type=int, default=30)
    a, _ = p.parse_known_args()
    out_dir = os.environ.get("KSTAGE_OUT")
    sink = open(os.path.join(out_dir, "ablation.jsonl"), "a") if out_dir else None
    mod = load_variants(a.variants)
    dtype = torch.float32 if INTERP else torch.bfloat16
    shapes = [(256, 96), (384, 70)] if INTERP else [
        tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    ms = [1, 3] if INTERP else [int(v) for v in a.m.split(",")]
    names = [n for n in VARIANTS if not a.only or n in a.only.split(",")]
    gbps = None if INTERP else copy_gbps()
    l2 = 0 if INTERP else torch.cuda.get_device_properties(0).L2_cache_size
    print("KPROF " + json.dumps(dict(copy_gbps=gbps, l2_mb=l2 / 2**20,
                                     card=None if INTERP else torch.cuda.get_device_name())),
          flush=True)
    summary = {}
    by_mode = {}
    for (k, n), mode in [(s, md) for s in shapes for md in a.modes.split(",")]:
        w0 = weights(k, n, mode, dtype, 0)
        wbytes = sum(t.numel() * t.element_size() for t in w0[:3 if mode != "pc" else 2])
        if mode != "pc":
            wbytes = sum(t.numel() * t.element_size() for t in w0[:3])
        reps = 1 if INTERP else max(1, min(256, math.ceil(4 * l2 / wbytes)))
        copies = [w0] + [[t.clone() for t in w0] for _ in range(reps - 1)]
        for m in ms:
            x = torch.randn(m, k, generator=torch.Generator().manual_seed(1)).to(dtype).to(DEV)
            ref = launcher(mod, "tip717", mode, m, k, n)(x, *w0)
            for name in names:
                if name == "bf16magic_uniform" and (mode in ("pc", "g_act") or dtype != torch.bfloat16):
                    continue
                if name == "fast2_bnh64_sk4" and (k // 64) % 4:
                    continue
                fn = launcher(mod, name, mode, m, k, n)
                out = fn(x, *w0)
                # Bit equality only at the tip717 configuration.
                same = bool(torch.equal(out, ref)) if VARIANTS[name][2:8] == ROW else None
                rec = dict(variant=name, mode=mode, shape=f"{k}x{n}", m=m,
                           same_bits=same, finite=bool(torch.isfinite(out).all()),
                           **fn.meta)
                if not INTERP:
                    us = graph_us(fn, x, copies, a.replays)
                    us_k = graph_us(lambda xx, *c: fn(xx, *c, kernel_only=True),
                                    x, copies, a.replays)
                    moved = wbytes + m * k * 2 + m * n * 2
                    rec.update(us=round(us, 3), us_kernel=round(us_k, 3),
                               pct_copy=round(100 * moved / (us * 1e-6) / 1e9 / gbps, 2),
                               copies=reps)
                    summary.setdefault(name, []).append(us)
                    by_mode.setdefault((name, mode), []).append(us)
                if sink:
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                if m == ms[0]:
                    print("KPROF " + json.dumps(dict(tool="pod_ablation", **rec)), flush=True)
    if not INTERP:
        for (name, mode), times in by_mode.items():
            base_m = by_mode.get(("tip717", mode))
            if base_m and len(times) == len(base_m):
                ratio = math.exp(statistics.mean(
                    math.log(b / t) for b, t in zip(base_m, times)))
                print("KPROF " + json.dumps(dict(
                    tool="pod_ablation", variant=name, mode=mode,
                    geomean_speedup_vs_tip717=round(ratio, 3))), flush=True)
        base = summary.get("tip717")
        for name, times in summary.items():
            if base and len(times) == len(base):
                ratio = math.exp(statistics.mean(
                    math.log(b / t) for b, t in zip(base, times)))
                print("KPROF " + json.dumps(dict(tool="pod_ablation", variant=name,
                                                 geomean_speedup_vs_tip717=round(ratio, 3))),
                      flush=True)


if __name__ == "__main__":
    main()
