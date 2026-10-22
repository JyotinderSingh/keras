"""Variants of tip717 `_w4a16_kernel` (keras/src/backend/torch/ops/quantized.py
lines 230-350) for the KERNEL-BODY analysis. Same argument order as tip717,
plus `NH` (the codes row stride, `codes.shape[1]`) and constexpr switches.

- `k_nh`: tip717 with `NH` passed in (line 255 recomputes it as
  `(N + 1) // 2`, which loses the divisibility of the row stride).
- `k_hoist`: `k_nh` + one uniform check per program before the K loop,
  then a fast loop or a general loop (no per-tile reduce / `if`).
  MAGIC: constexpr, the int -> float conversion by the 2^23 magic number.
  ROWS: constexpr, the scale and the zero point as one precomputed float32
  row block per group (Mosaic layout: (n_groups, NH // BNH, 4, BNH)).
"""
import triton
import triton.language as tl


@triton.jit
def _dot(a, b):
    return tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)


@triton.jit
def k_nh(x_ptr, codes_ptr, scale_ptr, zero_ptr, g_idx_ptr, y_ptr, M, N, K,
         n_groups, group_size, NH, XOR: tl.constexpr,
         PER_CHANNEL: tl.constexpr, UPCAST_DOT: tl.constexpr,
         INPUT_PRECISION: tl.constexpr, SPLIT_K: tl.constexpr,
         BM: tl.constexpr, BNH: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    bytes_ = pid_n * BNH + tl.arange(0, BNH)
    columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
    row_mask = rows < M
    column_mask = columns < N
    dtype = x_ptr.dtype.element_ty
    acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
    acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
    tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
    for tile in tl.range(pid_k * tiles, last):
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
        if PER_CHANNEL:
            d_even = (even - XOR).to(tl.float32).to(dtype)
            d_odd = (odd - XOR).to(tl.float32).to(dtype)
            acc_even += _dot(x, d_even)
            acc_odd += _dot(x, d_odd)
        else:
            group = k0 // group_size
            g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
            g = g.to(tl.int32)
            outside = tl.where(k_mask & (g != group), 1, 0)
            if (tl.max(outside, axis=0) == 0) & (group < n_groups):
                row = group * N + columns
                s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                z_row = tl.load(zero_ptr + row, column_mask, other=0)
                s_row_even, s_row_odd = tl.split(
                    tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                z_row_even, z_row_odd = tl.split(
                    tl.reshape(z_row.to(tl.int32) + XOR, (BNH, 2)))
                d_even = (even - z_row_even[None, :]).to(tl.float32)
                d_odd = (odd - z_row_odd[None, :]).to(tl.float32)
                d_even, d_odd = d_even.to(dtype), d_odd.to(dtype)
                partial = _dot(x, d_even)
                acc_even += partial * s_row_even[None, :]
                partial = _dot(x, d_odd)
                acc_odd += partial * s_row_odd[None, :]
            else:
                g = tl.minimum(tl.maximum(g, 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_even, s_odd = tl.split(
                    tl.reshape(s.to(tl.float32), (BK, BNH, 2)))
                z_even, z_odd = tl.split(
                    tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2)))
                w_even = ((even - z_even).to(tl.float32) * s_even).to(dtype)
                w_odd = ((odd - z_odd).to(tl.float32) * s_odd).to(dtype)
                acc_even += _dot(x, w_even)
                acc_odd += _dot(x, w_odd)
    acc = tl.interleave(acc_even, acc_odd)
    if PER_CHANNEL:
        s = tl.load(scale_ptr + columns, mask=column_mask, other=1.0)
        acc = acc / s.to(tl.float32)[None, :]
    if SPLIT_K > 1:
        rows = pid_k.to(tl.int64) * M + rows
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & column_mask[None, :])


@triton.jit
def _unpack(b, z_even, z_odd, XOR: tl.constexpr, MAGIC: tl.constexpr,
            dtype: tl.constexpr):
    """d = q - z' in the dot dtype; z_even/z_odd are int32 (MAGIC=0) or the
    float32 2^23 + z' (MAGIC=1)."""
    even = (b & 15) ^ XOR
    odd = ((b >> 4) & 15) ^ XOR
    if MAGIC:
        # 2^23 + q exactly, then minus 2^23 + z': an exact small integer.
        f_even = (even | 0x4B000000).to(tl.float32, bitcast=True)
        f_odd = (odd | 0x4B000000).to(tl.float32, bitcast=True)
        d_even = (f_even - z_even).to(dtype)
        d_odd = (f_odd - z_odd).to(dtype)
    else:
        d_even = (even - z_even).to(tl.float32).to(dtype)
        d_odd = (odd - z_odd).to(tl.float32).to(dtype)
    return d_even, d_odd


@triton.jit
def k_hoist(x_ptr, codes_ptr, scale_ptr, zero_ptr, g_idx_ptr, y_ptr, M, N, K,
            n_groups, group_size, NH, XOR: tl.constexpr,
            PER_CHANNEL: tl.constexpr, UPCAST_DOT: tl.constexpr,
            INPUT_PRECISION: tl.constexpr, SPLIT_K: tl.constexpr,
            BM: tl.constexpr, BNH: tl.constexpr, BK: tl.constexpr,
            MAGIC: tl.constexpr = 0, UNIFORM: tl.constexpr = 0,
            GENERAL: tl.constexpr = 1):
    """UNIFORM=1: the frontend asserts contiguous groups (no check).
    GENERAL=0: compile no general loop (to see the fast loop alone)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    bytes_ = pid_n * BNH + tl.arange(0, BNH)
    columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
    row_mask = rows < M
    column_mask = columns < N
    dtype = x_ptr.dtype.element_ty
    acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
    acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
    tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    first = pid_k * tiles
    last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
    uniform = True
    if not PER_CHANNEL:
        if not UNIFORM:
            # One check per program: every K tile of this split lies in the
            # group `k0 // G`, and that group exists.
            bad = tl.zeros((BK,), dtype=tl.int32)
            for tile in tl.range(first, last):
                ks = tile * BK + tl.arange(0, BK)
                k_mask = ks < K
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0).to(tl.int32)
                bad = bad | tl.where(k_mask & (g != (tile * BK) // group_size),
                                     1, 0)
            uniform = (tl.max(bad, axis=0) == 0) & (
                ((last - 1) * BK) // group_size < n_groups)
    if uniform:
        for tile in tl.range(first, last):
            k0 = tile * BK
            ks = k0 + tl.arange(0, BK)
            k_mask = ks < K
            x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                        mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                        other=0).to(tl.int32)
            if PER_CHANNEL:
                if MAGIC:
                    zf = 8388608.0 + XOR
                    d_even, d_odd = _unpack(b, zf, zf, XOR, MAGIC, dtype)
                else:
                    d_even, d_odd = _unpack(b, XOR, XOR, XOR, MAGIC, dtype)
                acc_even += _dot(x, d_even)
                acc_odd += _dot(x, d_odd)
            else:
                row = (k0 // group_size) * N + columns
                s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                z_row = tl.load(zero_ptr + row, column_mask, other=0)
                s_even, s_odd = tl.split(
                    tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                if MAGIC:
                    zz = z_row.to(tl.float32) + (8388608.0 + XOR)
                else:
                    zz = z_row.to(tl.int32) + XOR
                z_even, z_odd = tl.split(tl.reshape(zz, (BNH, 2)))
                d_even, d_odd = _unpack(b, z_even[None, :], z_odd[None, :],
                                        XOR, MAGIC, dtype)
                acc_even += _dot(x, d_even) * s_even[None, :]
                acc_odd += _dot(x, d_odd) * s_odd[None, :]
    else:
        if GENERAL:
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None] & k_mask[None, :],
                            other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                            mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                            other=0).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
                g = tl.minimum(tl.maximum(g.to(tl.int32), 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_e, s_o = tl.split(tl.reshape(s.to(tl.float32), (BK, BNH, 2)))
                z_e, z_o = tl.split(
                    tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2)))
                w_even = ((even - z_e).to(tl.float32) * s_e).to(dtype)
                w_odd = ((odd - z_o).to(tl.float32) * s_o).to(dtype)
                acc_even += _dot(x, w_even)
                acc_odd += _dot(x, w_odd)
    acc = tl.interleave(acc_even, acc_odd)
    if PER_CHANNEL:
        s = tl.load(scale_ptr + columns, mask=column_mask, other=1.0)
        acc = acc / s.to(tl.float32)[None, :]
    if SPLIT_K > 1:
        rows = pid_k.to(tl.int64) * M + rows
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & column_mask[None, :])


@triton.jit
def k_fast2(x_ptr, codes_ptr, scale_ptr, zero_ptr, g_idx_ptr, y_ptr, M, N, K,
            n_groups, group_size, NH, XOR: tl.constexpr,
            PER_CHANNEL: tl.constexpr, UPCAST_DOT: tl.constexpr,
            INPUT_PRECISION: tl.constexpr, SPLIT_K: tl.constexpr,
            BM: tl.constexpr, BNH: tl.constexpr, BK: tl.constexpr,
            MAGIC: tl.constexpr = 1, PREFETCH_S: tl.constexpr = 1,
            EVEN: tl.constexpr = 0, CHECK: tl.constexpr = 1024,
            GENERAL: tl.constexpr = 1):
    """`k_hoist` + a chunked uniform check, the scale row of the next tile
    loaded one tile ahead (PREFETCH_S), and no K/N masks when EVEN (the
    frontend sets it when K % BK == 0 and NH % BNH == 0)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    bytes_ = pid_n * BNH + tl.arange(0, BNH)
    columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
    row_mask = rows < M
    column_mask = columns < N
    dtype = x_ptr.dtype.element_ty
    acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
    acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
    tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    first = pid_k * tiles
    last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
    uniform = True
    if not PER_CHANNEL:
        # One check per program, CHECK rows per load: row k of tile t must be
        # in group (t * BK) // G, and the last tile's group must exist.
        bad = tl.zeros((CHECK,), dtype=tl.int32)
        for start in tl.range(first * BK, last * BK, CHECK):
            ks = start + tl.arange(0, CHECK)
            k_mask = ks < tl.minimum(last * BK, K)
            g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0).to(tl.int32)
            want = ((ks // BK) * BK) // group_size
            bad = bad | tl.where(k_mask & (g != want), 1, 0)
        uniform = (tl.max(bad, axis=0) == 0) & (
            ((last - 1) * BK) // group_size < n_groups)
    if uniform:
        if not PER_CHANNEL:
            s_next = tl.load(scale_ptr + (first * BK // group_size) * N + columns,
                             column_mask, other=0.0)
        for tile in tl.range(first, last):
            k0 = tile * BK
            ks = k0 + tl.arange(0, BK)
            if EVEN:
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None], other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :]
                            ).to(tl.int32)
            else:
                k_mask = ks < K
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                            mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                            other=0).to(tl.int32)
            if PER_CHANNEL:
                zf = 8388608.0 + XOR
                if MAGIC:
                    d_even, d_odd = _unpack(b, zf, zf, XOR, MAGIC, dtype)
                else:
                    d_even, d_odd = _unpack(b, XOR, XOR, XOR, MAGIC, dtype)
                acc_even += _dot(x, d_even)
                acc_odd += _dot(x, d_odd)
            else:
                row = (k0 // group_size) * N + columns
                if PREFETCH_S:
                    s_row = s_next
                    nxt = tl.minimum(tile + 1, last - 1) * BK // group_size
                    s_next = tl.load(scale_ptr + nxt * N + columns,
                                     column_mask, other=0.0)
                else:
                    s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
                if EVEN:
                    z_row = tl.load(zero_ptr + row)
                else:
                    z_row = tl.load(zero_ptr + row, column_mask, other=0)
                s_even, s_odd = tl.split(
                    tl.reshape(s_row.to(tl.float32), (BNH, 2)))
                if MAGIC:
                    zz = z_row.to(tl.float32) + (8388608.0 + XOR)
                else:
                    zz = z_row.to(tl.int32) + XOR
                z_even, z_odd = tl.split(tl.reshape(zz, (BNH, 2)))
                d_even, d_odd = _unpack(b, z_even[None, :], z_odd[None, :],
                                        XOR, MAGIC, dtype)
                acc_even += _dot(x, d_even) * s_even[None, :]
                acc_odd += _dot(x, d_odd) * s_odd[None, :]
    else:
        if GENERAL:
            for tile in tl.range(first, last):
                k0 = tile * BK
                ks = k0 + tl.arange(0, BK)
                k_mask = ks < K
                x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                            mask=row_mask[:, None] & k_mask[None, :],
                            other=0.0)
                b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                            mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                            other=0).to(tl.int32)
                even = (b & 15) ^ XOR
                odd = ((b >> 4) & 15) ^ XOR
                g = tl.load(g_idx_ptr + ks, mask=k_mask, other=0.0)
                g = tl.minimum(tl.maximum(g.to(tl.int32), 0), n_groups - 1)
                offsets = g[:, None] * N + columns[None, :]
                mask = k_mask[:, None] & column_mask[None, :]
                s = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
                z = tl.load(zero_ptr + offsets, mask=mask, other=0)
                s_e, s_o = tl.split(tl.reshape(s.to(tl.float32), (BK, BNH, 2)))
                z_e, z_o = tl.split(
                    tl.reshape(z.to(tl.int32) + XOR, (BK, BNH, 2)))
                w_even = ((even - z_e).to(tl.float32) * s_e).to(dtype)
                w_odd = ((odd - z_o).to(tl.float32) * s_o).to(dtype)
                acc_even += _dot(x, w_even)
                acc_odd += _dot(x, w_odd)
    acc = tl.interleave(acc_even, acc_odd)
    if PER_CHANNEL:
        s = tl.load(scale_ptr + columns, mask=column_mask, other=1.0)
        acc = acc / s.to(tl.float32)[None, :]
    if SPLIT_K > 1:
        rows = pid_k.to(tl.int64) * M + rows
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & column_mask[None, :])


@triton.jit
def k_bf16magic(x_ptr, codes_ptr, scale_ptr, zero_ptr, g_idx_ptr, y_ptr, M, N,
                K, n_groups, group_size, NH, XOR: tl.constexpr,
                PER_CHANNEL: tl.constexpr, UPCAST_DOT: tl.constexpr,
                INPUT_PRECISION: tl.constexpr, SPLIT_K: tl.constexpr,
                BM: tl.constexpr, BNH: tl.constexpr, BK: tl.constexpr):
    """Uniform-only fast loop (grouped), d built in bfloat16 directly:
    bits 0x4300 | q = 128 + q, minus the bfloat16 128 + z'."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    bytes_ = pid_n * BNH + tl.arange(0, BNH)
    columns = pid_n * 2 * BNH + tl.arange(0, 2 * BNH)
    row_mask = rows < M
    column_mask = columns < N
    acc_even = tl.zeros((BM, BNH), dtype=tl.float32)
    acc_odd = tl.zeros((BM, BNH), dtype=tl.float32)
    tiles = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    first = pid_k * tiles
    last = tl.minimum((pid_k + 1) * tiles, tl.cdiv(K, BK))
    for tile in tl.range(first, last):
        k0 = tile * BK
        ks = k0 + tl.arange(0, BK)
        k_mask = ks < K
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(codes_ptr + ks[:, None] * NH + bytes_[None, :],
                    mask=k_mask[:, None] & (bytes_ < NH)[None, :],
                    other=0).to(tl.int32)
        row = (k0 // group_size) * N + columns
        s_row = tl.load(scale_ptr + row, column_mask, other=0.0)
        z_row = tl.load(zero_ptr + row, column_mask, other=0)
        s_even, s_odd = tl.split(tl.reshape(s_row.to(tl.float32), (BNH, 2)))
        zb = (z_row.to(tl.float32) + (128.0 + XOR)).to(tl.bfloat16)
        z_even, z_odd = tl.split(tl.reshape(zb, (BNH, 2)))
        even = ((b & 15) ^ XOR) | 0x4300
        odd = (((b >> 4) & 15) ^ XOR) | 0x4300
        d_even = even.to(tl.int16).to(tl.bfloat16, bitcast=True) - z_even[None, :]
        d_odd = odd.to(tl.int16).to(tl.bfloat16, bitcast=True) - z_odd[None, :]
        acc_even += _dot(x, d_even) * s_even[None, :]
        acc_odd += _dot(x, d_odd) * s_odd[None, :]
    acc = tl.interleave(acc_even, acc_odd)
    if SPLIT_K > 1:
        rows = pid_k.to(tl.int64) * M + rows
    offsets = rows[:, None] * N + columns[None, :]
    tl.store(y_ptr + offsets, acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & column_mask[None, :])
