"""Per-call split of Triton K1 at decode (pod probe of the memory-access analysis).

Run from the Keras tree (KERAS_BACKEND=torch). Times each candidate as tune.py
does: a CUDA graph of one call per weight copy, the copies totalling >= 4x the
L2, CUDA events around graph replays, the median per call.

Candidates (`--cands`):
  tip     quantized.w4a16_matmul: the kernel, torch.sum and .to (T1a best row)
  kernel  tip `_w4a16_kernel` alone, into a preallocated split-K workspace
  sum     torch.sum(workspace, 0) alone          cast   .to(bfloat16) alone
  v0      variant_kernel.py MODE 0: the tip loop with NH as an argument only
          (vectorized code loads, same per-tile branch), then torch.sum and .to
  v1, v3  variant_kernel.py (NH argument; MODE 1 static fast loop, MODE 3 one
          uniformity check per program then the fast or the general loop),
          then torch.sum and .to
  v1f, v3f  the same, with one Triton split-K reduce (fixed order, bf16 out)
  v1f4, v3f4  v1f, v3f with the scale row prefetched one K step ahead in
          registers (SZ_SPLIT 4); the pipeliner does not make it a cp.async
  v3p     MODE 3 on a permuted g_idx (the general loop; GPTQ act_order)
  tipp    tip on the same permuted g_idx
Prints one `KPROF {json}` line per (shape, mode, cand), and writes the same
records to $KSTAGE_OUT/k1_split.jsonl. Unknown flags are ignored. Under nsys with
--cuda-graph-trace=node each kernel node of the graph shows separately.

  python k1_split.py --shapes=4096x4096,14336x4096 --modes=int4_g128,gptq \
      --cands=tip,tipp,kernel,sum,cast,v0,v1,v3,v1f,v3f,v1f4,v3f4,v3p,v0p --config=16,32,64,4,3,8
"""
import argparse
import importlib.util
import json
import math
import os
import statistics

import torch
import triton
import triton.language as tl

from keras.src.backend.torch.ops import quantized as q

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "variant_kernel", os.path.join(HERE, "variant_kernel.py"))
var = importlib.util.module_from_spec(spec)
spec.loader.exec_module(var)
CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"


@triton.jit
def _splitk_reduce(ws_ptr, y_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(ws_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


def weight(k, n, mode, gen, permuted=False):
    signed = mode == "int4_g128"
    nh = n // 2
    if signed:
        codes = torch.randint(-128, 128, (k, nh), dtype=torch.int8, generator=gen)
        zero = torch.randint(-8, 8, (k // 128, n), dtype=torch.int8, generator=gen)
    else:
        codes = torch.randint(0, 256, (k, nh), dtype=torch.uint8, generator=gen)
        zero = torch.randint(0, 16, (k // 128, n), dtype=torch.uint8, generator=gen)
    scale = (torch.rand((k // 128, n), generator=gen) * 0.02 + 0.001).to(torch.bfloat16)
    g_idx = (torch.arange(k) // 128).to(torch.float32)
    if permuted:
        g_idx = g_idx[torch.randperm(k, generator=gen)]
    return [t.to(DEV) for t in (codes, scale, zero, g_idx)], signed


def candidates(cfg, k, n, signed, m, chunk=1024):
    bm, bnh, bk, warps, stages, split = cfg
    nh = n // 2
    grid = (triton.cdiv(m, bm), triton.cdiv(nh, bnh), split)
    xor = 8 if signed else 0
    ws = torch.empty((split, m, n), dtype=torch.float32, device=DEV)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
    config = q.KernelConfig(*cfg)
    opts = dict(num_warps=warps, num_stages=stages)

    def tip(x, w):
        return q.w4a16_matmul(x, *w, group_size=128, signed=signed,
                              per_channel=False, config=config,
                              upcast_dot=not CUDA)

    def kernel(x, w):
        codes, scale, zero, g_idx = w
        q._w4a16_kernel[grid](
            x, codes, scale, zero, g_idx, ws, m, n, k, scale.shape[0], 128,
            XOR=xor, PER_CHANNEL=False, UPCAST_DOT=not CUDA,
            INPUT_PRECISION="ieee", SPLIT_K=split, BM=bm, BNH=bnh, BK=bk, **opts)
        return ws

    def variant(mode, fused, sz=0):
        def run(x, w):
            codes, scale, zero, g_idx = w
            var._w4a16_kernel[grid](
                x, codes, scale, zero, g_idx, ws, m, n, k, codes.shape[1],
                scale.shape[0], 128, XOR=xor, PER_CHANNEL=False,
                UPCAST_DOT=not CUDA, INPUT_PRECISION="ieee", SPLIT_K=split,
                BM=bm, BNH=bnh, BK=bk, MODE=mode, SZ_SPLIT=sz, CHUNK=chunk, **opts)
            if fused:
                _splitk_reduce[(triton.cdiv(m * n, 1024),)](
                    ws, y, m * n, SPLIT_K=split, BLOCK=1024, num_warps=4)
                return y
            return torch.sum(ws, dim=0).to(torch.bfloat16)
        return run

    return {
        "tip": tip, "tipp": tip, "kernel": kernel,
        "sum": lambda x, w: torch.sum(ws, dim=0),
        "cast": lambda x, w: ws[0].to(torch.bfloat16),
        "v0": variant(0, False), "v0p": variant(0, False),
        "v1": variant(1, False), "v3": variant(3, False),
        "v1f": variant(1, True), "v3f": variant(3, True),
        "v3p": variant(3, True),
        "v1f4": variant(1, True, 4), "v3f4": variant(3, True, 4),
    }


def time_graph(fn, x, copies, samples, warmup=3):
    def run():
        return [fn(x, c) for c in copies]
    if not CUDA:
        run()
        return None, "cpu"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(samples):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / len(copies))
    return statistics.median(out), "graph"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="4096x4096,14336x4096")
    p.add_argument("--modes", default="int4_g128,gptq")
    p.add_argument("--cands", default="tip,tipp,kernel,sum,cast,v0,v1,v3,v1f,v3f,v1f4,v3f4,v3p,v0p")
    p.add_argument("--config", default="16,32,64,4,3,8")
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024,
                   help="g_idx rows per step of the MODE 3 check")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--copy_gbps", type=float, default=918.19)
    a, _ = p.parse_known_args()
    out_dir = os.environ.get("KSTAGE_OUT")
    sink = open(os.path.join(out_dir, "k1_split.jsonl"), "a") if out_dir else None
    cfg = tuple(int(v) for v in a.config.split(","))
    l2 = torch.cuda.get_device_properties(0).L2_cache_size if CUDA else 1 << 20
    gen = torch.Generator().manual_seed(0)
    for shape in a.shapes.split(","):
        k, n = (int(v) for v in shape.split("x"))
        for mode in a.modes.split(","):
            x = torch.randn((a.m, k), generator=gen).to(torch.bfloat16).to(DEV)
            if not CUDA:
                x = x.float()
            (w0, signed) = weight(k, n, mode, gen)
            (wp, _) = weight(k, n, mode, gen, permuted=True)
            nbytes = sum(t.numel() * t.element_size() for t in w0[:3])
            ideal = nbytes + a.m * (k + n) * 2
            count = 1 if not CUDA else max(1, min(256, math.ceil(4 * l2 / nbytes)))
            cands = candidates(cfg, k, n, signed, a.m, a.chunk)
            want = {}
            for name in a.cands.split(","):
                base = wp if name in ("v3p", "tipp", "v0p") else w0
                copies = [base] + [[t.clone() for t in base] for _ in range(count - 1)]
                fn = cands[name]
                if CUDA:
                    torch.cuda.nvtx.range_push(f"{shape}:{mode}:{name}")
                us, method = time_graph(fn, x, copies, a.samples)
                if CUDA:
                    torch.cuda.nvtx.range_pop()
                y = fn(x, copies[0]).float()
                if CUDA:
                    torch.cuda.synchronize()
                row = dict(tool="k1_split", shape=shape, mode=mode, cand=name,
                           config=cfg, m=a.m, copies=count, method=method,
                           chunk=a.chunk)
                if us is not None:
                    row.update(us=round(us, 3))
                    if name not in ("kernel", "sum", "cast"):
                        row["pct_copy"] = round(100 * ideal / (us * 1e-6) / 1e9 / a.copy_gbps, 2)
                if name in ("tip", "tipp"):
                    want[name] = y
                ref = want.get("tipp" if name in ("v3p", "v0p") else "tip")
                if ref is not None and name.startswith("v"):
                    row["err_vs_tip"] = float((y - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
                print("KPROF " + json.dumps(row), flush=True)
                if sink:
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()


if __name__ == "__main__":
    main()
