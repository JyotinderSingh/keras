# Benchmark the quantized contraction routes

`bench.py` measures the forward pass of a 4-bit weight-only quantized
contraction. It compares the routes of `keras/src/quantizers/kernels.py`
with the path of today (the reference path) and with the float layer.
`make_tables.py` writes Markdown tables from the results.

Run both from the repository root, so that `keras` is this checkout.
Run one backend per process. `bench.py` prints `keras.__file__` and the
library versions when it starts.

```shell
KERAS_BACKEND=jax python -m benchmarks.quantized_matmul.bench \
    --out=/path/to/results/jax.jsonl
python -m benchmarks.quantized_matmul.make_tables \
    /path/to/results/jax.jsonl > tables.md
```

`--out` is required. Write the results outside the repository. Each
cell appends one JSON line per candidate and timing to it, so a stopped
run keeps the cells that finished.

## Smoke run

These commands run every level on CPU with tiny shapes and few calls.
Each one takes less than two minutes on an Apple M-series CPU.

```shell
KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu \
    python -m benchmarks.quantized_matmul.bench \
    --out=/tmp/smoke/torch.jsonl --shapes=256x512 --m=1,16 \
    --logits_shapes=tiny-tied,tiny-untied --logits_m=1,16 --models=tiny \
    --warmup=2 --calls=5 --copy_mb=64 --l2_mb=0
KERAS_BACKEND=jax python -m benchmarks.quantized_matmul.bench \
    --out=/tmp/smoke/jax.jsonl --shapes=256x512 --m=1,16 \
    --logits_shapes=tiny-tied,tiny-untied --logits_m=1,16 --models=tiny \
    --warmup=2 --calls=5 --copy_mb=64 --l2_mb=0
KERAS_BACKEND=tensorflow python -m benchmarks.quantized_matmul.bench \
    --out=/tmp/smoke/tf.jsonl --shapes=256x512 --m=1,16 \
    --logits_shapes=tiny-tied,tiny-untied --logits_m=1,16 --models=tiny \
    --warmup=2 --calls=5 --copy_mb=64 --l2_mb=0
python -m benchmarks.quantized_matmul.make_tables /tmp/smoke/*.jsonl
```

## Levels

| Level | What runs |
|---|---|
| `raw` | The contraction on the stored tensors of a `Dense` layer, without the layer call: `(M, K)` inputs, `(M, N)` outputs |
| `dense` | One real `Dense` layer. The numbers include the frontend of the layer |
| `chain` | Distinct `Dense` layers on one input, with one sync per call, at `M <= 16`. The chain has at least 8 layers, and quantized weights of at least 4x the L2 cache: `--l2_mb`, else the L2 size of the torch CUDA device, else 0 (JAX and TensorFlow on a GPU need `--l2_mb`). A cell above the copy bandwidth runs once more with twice the layers |
| `logits` | The reverse projection of a tied or untied `ReversibleEmbedding`, int4 in groups of 128 and per channel |
| `model` | One decode step of a decoder with random weights and its int4 logits, at `M <= 16`. A stand-in for the attention mixes the query, key and value outputs elementwise, so no KV cache is read. The `head` candidate runs the logits alone, and each step record holds the logits share |

The `model` level builds Gemma2-2B-like and Llama-3-8B-like decoders
(`MODELS`). The `logits` level uses the tables of five models
(`LOGITS`). The `tiny` entries are for smoke runs.

## Modes

`int4_g128`, `int4_pc` (per channel), `gptq` (4-bit, groups of 128),
`gptq_act` (the same, with a random activation-order `g_idx`), `awq`
(4-bit, groups of 128), and the baselines `int8` (W8A8) and `int8_wo`
(weight-only). The stored tensors hold random values.

## Candidates

| Candidate | Kind | What runs |
|---|---|---|
| `float` | float | The float layer in the compute dtype |
| `F0` | form | The path of today, under `kernels.policy("off")` |
| `F1` | form | The two nibble planes dequantized with a broadcast of the group rows, then one matmul per plane |
| `F3` | form | Sum-then-scale: the zero-subtracted planes contract with the inputs in one batched matmul per plane, and the group scales multiply the float32 partial sums |
| A route of `kernels._ROUTES` | route (`n5`: form) | The route, under `kernels.force_route(name)` |
| `head` | head | The logits of a decode step alone (`model` level) |

F1 and F3 are functions of this harness, in `keras.ops`. They run as the
routes `harness_f1` and `harness_f3`, so a layer runs them through the
policy switch. They need contiguous groups, so `gptq_act` skips them.
`--routes` selects the routes to run (default: every route except
`reference_wrapped`). A route runs only where its `available` check
passes at the shape of the cell; at the `model` level, at each distinct
`(K, N)` of the projections of a block. Elsewhere it is recorded as
skipped, "not available at this shape". `--logits_routes` names the
routes of the `logits` and `model` levels.

The route counter `kernels.route_counts` proves that a route ran. A
candidate whose route the counter does not count is recorded as skipped.
For example, a calibration mode whose forward pass does not select a
route skips F1 and F3 at the `dense`, `chain` and `model` levels.

## Method

- At least `--warmup` calls (default 5) per candidate, then `--calls`
  timed calls (default 30). The candidates of a cell interleave call by
  call. The records give the minimum and the median.
- `graph` timing: a `jit` with `block_until_ready` (JAX), a
  `tf.function` with `jit_compile=True` (TensorFlow), or a CUDA-graph
  replay (torch on CUDA). `eager` timing: the eager call with one sync.
  torch on CPU and MPS has the eager timing only.
- A failed CUDA-graph capture is recorded as skipped, with its error.
  A Python scalar in a captured call is a host-to-device copy, which
  the capture refuses, so the harness builds its constants as device
  tensors once. The capture of each candidate is on the S0 checklist
  (the first GPU session of the measurement campaign).
- A candidate that fails to build or to warm up is recorded as skipped,
  with its error, and the other candidates still run.
- The roofline is a copy of `--copy_mb` MiB (default 1024) on the
  device. Each record gives the achieved GB/s (stored weights and
  activations, read once) and the percent of the copy bandwidth.
- `err` is `max|y - y_oracle| / max|y_oracle|` over the first 64 rows
  and 4096 columns. The oracle is a float64 NumPy contraction of the
  stored tensors: `oracle_contract` of the quantization test utilities
  for the projections, and a reverse contraction of the harness for the
  logits.
- `make_tables.py` gives two ratios per cell: the time over the time of
  the float layer, and the time over the best form without a kernel
  (F0, F1, F3 or `n5`).

## Record fields

`level`, `mode`, `dtype`, `shape`, `k`, `n`, `m`, `layers`, `rerun`
(`dense` and `chain`: 1 for the rerun of a chain with twice the
layers), `tied` (`logits`), `candidate`, `kind`, `config`, `timing`,
`min_ms`, `median_ms`, `calls`, `bytes`, `gbps`, `pct_copy`, `err`,
`logits_share` (`model`: the time of `head` over the time of the
step), `skipped` (the reason, in place of the times, for a candidate
that did not run), and the route evidence:
`route_counts`, `hlo` (JAX and TensorFlow graph timing: the custom
calls, the fusion kinds and the largest float temp), `events` (torch:
the profiler events with the most device time on CUDA, else the most
calls), `temp_bytes` and `compile_s` (JAX), `peak_bytes` (torch on
CUDA, eager). `env` holds the backend, `keras.__file__`, the library
versions, the device (name, platform, compute capability, driver), the
commit (`git rev-parse HEAD` of the checkout, else `$KERAS_SHA`), the
date, and the copy bandwidth.

## Suites

- `--suite tpu`: the TPU cells, with the `graph` timing only. The
  `dense`, `logits`, `chain` (8 layers) and `model` (Gemma2-2B-like)
  levels, for `int4_g128`, `int4_pc`, `gptq` and `awq`, in bfloat16
  and float32, at every shape and `M`. For `int4_g128` in bfloat16 at
  `M` 1 and 256, a Pallas route also runs a block sweep of `BNH`, `BK`
  and `BM`, each configuration under a route name of its own. A
  configuration whose `BNH` does not tile `N / 2`, or whose `BK` does
  not tile `K`, is recorded as skipped.
- `--suite tpu-decode`: the decode subset for other JAX versions:
  `dense` and `logits` at `M` 1 and 16 in bfloat16, at K = N = 4096,
  for `float`, `F0`, `F3` and the routes, with the `graph` timing only.

A flag on the command line overrides the value of the suite.
`run_tpu.sh` passes `--logits_routes=<the Pallas K-packed route>` for
the logits stage, so that the `logits` level runs that route beside F0.
