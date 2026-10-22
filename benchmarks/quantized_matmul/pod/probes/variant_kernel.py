"""Variants of tip717 `_w4a16_kernel` (keras/src/backend/torch/ops/quantized.py:221-350)
for IR experiments. Not Keras code; a probe.

Differences from tip717, selected by constexprs:
- NH is a kernel argument (the frontend passes `codes.shape[1]`), so the
  JIT's divisibility-16 specialization applies to the codes row stride.
  tip717 computes `NH = (N + 1) // 2` in the kernel (line 255), whose
  divisibility the compiler cannot prove.
- MODE 0: the tip717 loop (runtime `if` per K tile, both paths in the loop).
- MODE 1: static fast path only (the frontend guarantees g_idx[k] == k // G):
  no g_idx load, no reduction, no `if`; scale/zero rows loaded at the top level.
- MODE 3: the uniformity check of this program's whole K slice runs once
  before the K loop (CHUNK rows of g_idx per step), then one of two loops
  runs: the fast loop (MODE 1 body) or the general loop (tip717 general body).
- SZ_SPLIT: 0 = tip717 (one 2*BNH row, reshape + split); 1 = two strided
  loads (even and odd columns) per plane, no reshape/split.
"""
import triton
import triton.language as tl


@triton.jit
def _dot(a, b, UPCAST_DOT: tl.constexpr, INPUT_PRECISION: tl.constexpr):
    if UPCAST_DOT:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    return tl.dot(a, b, input_precision=INPUT_PRECISION, out_dtype=tl.float32)


@triton.jit
def _w4a16_kernel(
    x_ptr, codes_ptr, scale_ptr, zero_ptr, g_idx_ptr, y_ptr,
    M, N, K, NH, n_groups, group_size,
    XOR: tl.constexpr, PER_CHANNEL: tl.constexpr, UPCAST_DOT: tl.constexpr,
    INPUT_PRECISION: tl.constexpr, SPLIT_K: tl.constexpr,
    BM: tl.constexpr, BNH: tl.constexpr, BK: tl.constexpr,
    MODE: tl.constexpr, SZ_SPLIT: tl.constexpr, CHUNK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    bytes_ = pid_n * BNH + tl.arange(0, BNH)
    columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
    row_mask = rows < M
    column_mask = columns < N
    even_cols = 2 * bytes_
    odd_cols = 2 * bytes_ + 1
    even_mask = even_cols < N
    odd_mask = odd_cols < N
    dtype = x_ptr.dtype.element_ty
    acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
    acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
    tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    first = pid_k * tiles
    last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
    if PER_CHANNEL:
        for tile in tl.range(first, last):
            ks = tile * BK + tl.arange(0, BK)
            k_mask = ks < K
            x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                        mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                        other=0).to(tl.int32)
            even = (b & 15) ^ XOR
            odd = ((b >> 4) & 15) ^ XOR
            d_even = (even - XOR).to(tl.float32).to(dtype)
            d_odd = (odd - XOR).to(tl.float32).to(dtype)
            acc_even += _dot(x, d_even, UPCAST_DOT, INPUT_PRECISION)
            acc_odd += _dot(x, d_odd, UPCAST_DOT, INPUT_PRECISION)
    elif MODE == 0:
        # The tip717 loop (quantized.py:268-336), with NH as an argument.
        for tile in tl.range(first, last):
            k0 = tile * BK
            ks = k0 + tl.arange(0, BK)
            k_mask = ks < K
            x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                        mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                        other=0).to(tl.int32)
            even = (b & 15) ^ XOR
            odd = ((b >> 4) & 15) ^ XOR
            group = k0 // group_size
            g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
            g = g.to(tl.int32)
            outside = tl.where(k_mask & (g != group), 1, 0)
            if (tl.max(outside, axis=0) == 0) & (group < n_groups):
                row = group * N + columns
                s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                z_row = tl.load(zero_ptr + row, column_mask, other=0)
                s_row_even, s_row_odd = tl.split(tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                z_row_even, z_row_odd = tl.split(tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2)))
                d_even = (even - z_row_even[None, :]).to(tl.float32)
                d_odd = (odd - z_row_odd[None, :]).to(tl.float32)
                d_even, d_odd = d_even.to(dtype), d_odd.to(dtype)
                partial = _dot(x, d_even, UPCAST_DOT, INPUT_PRECISION)
                acc_even += partial * s_row_even[None, :]
                partial = _dot(x, d_odd, UPCAST_DOT, INPUT_PRECISION)
                acc_odd += partial * s_row_odd[None, :]
            else:
                g = tl.minimum(tl.maximum(g, 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_even, s_odd = tl.split(tl.reshape(s.to(tl.float32), (BK, BNH, 2)))
                z_even, z_odd = tl.split(tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2)))
                w_even = ((even - z_even).to(tl.float32) * s_even).to(dtype)
                w_odd = ((odd - z_odd).to(tl.float32) * s_odd).to(dtype)
                acc_even += _dot(x, w_even, UPCAST_DOT, INPUT_PRECISION)
                acc_odd += _dot(x, w_odd, UPCAST_DOT, INPUT_PRECISION)
    else:
        uniform = True
        if MODE == 3:
            # One check of the whole K slice of this program, before the loop.
            lo = first * BK
            hi = tl.minimum(last * BK, K)
            bad = tl.zeros((CHUNK,), dtype=tl.int32)
            for start in tl.range(lo, hi, CHUNK):
                kk = start + tl.arange(0, CHUNK)
                kmask = kk < hi
                g = tl.load(g_idx_ptr + kk, mask=kmask, other=0.0).to(tl.int32)
                bad = bad | tl.where(kmask & (g != kk // group_size), 1, 0)
            last_group = (hi - 1) // group_size
            uniform = (tl.max(bad, axis=0) == 0) & (last_group < n_groups)
        if (MODE == 1) | uniform:
            if SZ_SPLIT == 4:
                # Manual prefetch: the scale row of the first tile.
                s_cur = tl.load(scale_ptr + ((first * BK) // group_size) * N + columns,
                                column_mask & (first < last), other=0.0)
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                            mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                            other=0).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                group = k0 // group_size
                if SZ_SPLIT == 2:
                    pair = 2 * bytes_[:, None] + tl.arange(0, 2)[None, :]
                    pmask = pair < N
                    s2 = tl.load(scale_ptr + group * N + pair, pmask, other=0.0)
                    z2 = tl.load(zero_ptr + group * N + pair, pmask, other=0)
                    s_e, s_o = tl.split(s2.to(tl.float32))
                    z_e, z_o = tl.split(z2.to(tl.int32) + XOR)
                elif SZ_SPLIT == 3:
                    # Prefetch distance 1 in registers is left to the pipeliner:
                    # the scale row as float32 bits through a 2-D (1, 2*BNH) load.
                    row2 = group * N + columns[None, :]
                    s_row = tl.reshape(tl.load(scale_ptr + row2, column_mask[None, :], other=0.0), (2 * BNH,))
                    z_row = tl.reshape(tl.load(zero_ptr + row2, column_mask[None, :], other=0), (2 * BNH,))
                    s_e, s_o = tl.split(tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                    z_e, z_o = tl.split(tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2)))
                elif SZ_SPLIT == 4:
                    # The scale row of the next tile, carried to the next step.
                    nxt = tl.minimum(tile + 1, last - 1)
                    s_nxt = tl.load(scale_ptr + ((nxt * BK) // group_size) * N + columns,
                                    column_mask, other=0.0)
                    z_row = tl.load(zero_ptr + group * N + columns, column_mask, other=0)
                    s_e, s_o = tl.split(tl.reshape(s_cur.to(tl.float32), (BNH, 2)))
                    z_e, z_o = tl.split(tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2)))
                elif SZ_SPLIT == 1:
                    base = group * N
                    s_e = tl.load(scale_ptr + base + even_cols, even_mask, other=0.0).to(tl.float32)
                    s_o = tl.load(scale_ptr + base + odd_cols, odd_mask, other=0.0).to(tl.float32)
                    z_e = tl.load(zero_ptr + base + even_cols, even_mask, other=0).to(tl.int32) + XOR
                    z_o = tl.load(zero_ptr + base + odd_cols, odd_mask, other=0).to(tl.int32) + XOR
                else:
                    row = group * N + columns
                    s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                    z_row = tl.load(zero_ptr + row, column_mask, other=0)
                    s_e, s_o = tl.split(tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                    z_e, z_o = tl.split(tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2)))
                d_even = (even - z_e[None, :]).to(tl.float32).to(dtype)
                d_odd = (odd - z_o[None, :]).to(tl.float32).to(dtype)
                acc_even += _dot(x, d_even, UPCAST_DOT, INPUT_PRECISION) * s_e[None, :]
                acc_odd += _dot(x, d_odd, UPCAST_DOT, INPUT_PRECISION) * s_o[None, :]
                if SZ_SPLIT == 4:
                    s_cur = s_nxt
        else:
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                            mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                            other=0).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0).to(tl.int32)
                g = tl.minimum(tl.maximum(g, 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_even, s_odd = tl.split(tl.reshape(s.to(tl.float32), (BK, BNH, 2)))
                z_even, z_odd = tl.split(tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2)))
                w_even = ((even - z_even).to(tl.float32) * s_even).to(dtype)
                w_odd = ((odd - z_odd).to(tl.float32) * s_odd).to(dtype)
                acc_even += _dot(x, w_even, UPCAST_DOT, INPUT_PRECISION)
                acc_odd += _dot(x, w_odd, UPCAST_DOT, INPUT_PRECISION)
    acc = tl.interleave(acc_even, acc_odd)
    if PER_CHANNEL:
        s = tl.load(scale_ptr + columns, mask=column_mask, other=1.0)
        acc = acc / s.to(tl.float32)[None, :]
    if SPLIT_K > 1:
        rows = pid_k.to(tl.int64) * M + rows
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & column_mask[None, :])
