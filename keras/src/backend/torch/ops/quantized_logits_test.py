"""The Triton logits kernel (`quantized.w4a16_logits`) as an op.

The module runs where Triton imports, on a CUDA device or under the
Triton interpreter (`TRITON_INTERPRET=1`, which the torch CPU CI job
sets), as `quantized_matmul_test.py` does. Under the interpreter, the
ops cast the dot operands to float32 (`quantized.INTERPRETED`), and a
test compiles the kernel for CUDA devices in a subprocess. The tables
come from real int4 `ReversibleEmbedding` layers, tied and untied, in
groups of 128 and 256 and per-channel. `LookupHandlers` stores int8
codes only, so each table has signed codes. The route suites of
`lookup_test.py` run the route through real layers, on CPU through the
interpret route `"triton_interpret"` of `kernels_test_utils`. The labels
G1 to G8 are those of `kernels_test_utils`.
"""

import dataclasses
import itertools
import json
import os
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.quantizers import kernels
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.testing.test_utils import named_product

INTERPRETED = False
if backend.backend() == "torch":
    import torch

    from keras.src.backend.torch.ops import quantized

    INTERPRETED = quantized.INTERPRETED
    if quantized.triton is None:
        SKIP = "Triton does not import."
    elif not (torch.cuda.is_available() or INTERPRETED):
        SKIP = "No CUDA device, and the Triton interpreter is off."
    else:
        SKIP = None
    if quantized.triton is not None:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
else:
    SKIP = "The Triton kernels are kernels of the torch backend."

pytestmark = pytest.mark.skipif(SKIP is not None, reason=str(SKIP))


# name -> (M, D, V).
SHAPES = {
    # Four K tiles of 128 rows: four groups of 128, or two of 256.
    "decode": (1, 512, 256),
    # Ragged `M`, an odd `D` with a partial last group and K tile, and a
    # `V` that is not a multiple of a block or of 128.
    "ragged": (33, 331, 200),
    # `D` and `V` pass the alignment gate of the route. The inputs are
    # `(2, 3, D)`.
    "layer": (6, 256, 384),
}
# Compute dtype -> (G1 floor, G2 bound), as in `quantized_matmul_test.py`.
TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "float16": (2e-4, 2**-9),
    "bfloat16": (2**-7 if INTERPRETED else 1e-3, 2**-7),
}
BLOCKS = {"g128": 128, "g256": 256, "per_channel": -1}
# Compute capability -> the shared memory of one block, in bytes.
SHARED_MEMORY = {(8, 0): 166912, (8, 9): 101376, (9, 0): 232448}
POINTERS = {"bfloat16": "*bf16", "float16": "*fp16", "float32": "*fp32"}


def _layer(
    tied,
    block_size,
    shape="layer",
    policy="float32",
    reverse_dtype=None,
    dim=None,
    vocabulary=None,
):
    """An int4 `ReversibleEmbedding` with the `(D, V)` of `SHAPES[shape]`."""
    _, shape_dim, shape_vocabulary = SHAPES[shape]
    dim, vocabulary = dim or shape_dim, vocabulary or shape_vocabulary
    rng = np.random.default_rng(0)
    layer = layers.ReversibleEmbedding(
        vocabulary,
        dim,
        tie_weights=tied,
        reverse_dtype=reverse_dtype,
        dtype=policy,
    )
    layer.build()
    layer._embeddings.assign(rng.standard_normal((vocabulary, dim)))
    if not tied:
        layer.reverse_embeddings.assign(rng.standard_normal((dim, vocabulary)))
    config = Int4QuantizationConfig(block_size=block_size)
    layer.quantize("int4", config=config)
    return layer


def _inputs(shape, seed=1):
    m, dim, _ = SHAPES[shape]
    if shape == "layer":
        return _random((2, 3, dim), seed)
    return _random((m, dim), seed)


def _random(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _stored(layer):
    """The view of the stored table that the route reads, and its
    orientation: the forward table `(V, D)` of a tied layer, or the
    reverse table `(D, V)` of an untied one."""
    strategy = strategy_registry.get_strategy("int4")
    if layer.tie_weights:
        return strategy._get_lookup_quantized_weight(layer, None), "tied"
    view = strategy._get_reverse_lookup_quantized_weight(layer, None)
    return view, "untied"


def _table(layer):
    """`_stored` over the tensors that the reverse lookup of the layer on
    its own reads: under a mixed policy, in the compute dtype."""
    view, orientation = _stored(layer)
    with AutocastScope(layer.compute_dtype):
        return view.with_tensors(view.read_tensors()), orientation


def _direct(layer, x, **kwargs):
    """A direct `w4a16_logits` call on the inputs of the layer: `(M, V)`."""
    table, orientation = _table(layer)
    x = ops.convert_to_tensor(x)
    x = ops.cast(x, layer.reverse_dtype or layer.compute_dtype)
    return quantized.w4a16_logits(
        ops.reshape(x, (-1, layer.output_dim)),
        table.codes,
        table.scale,
        table.zero_point,
        group_size=table.scheme.group_size,
        per_channel=table.scheme.group_size is None,
        orientation=orientation,
        **kwargs,
    )


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _bits(x):
    """The bits of `x` as float32, so that signed zeros differ too."""
    return _numpy(x).view("int32")


def _oracle(layer, x):
    """The float64 logits of `x` against the table of the layer.

    It shares no code with Keras. It unpacks the codes along `D`, the low
    nibble first, with sign extension, and it dequantizes them with the
    scale values that the route reads. The inputs are rounded to the
    reverse dtype.
    """
    table, orientation = _table(layer)
    codes = ops.convert_to_numpy(table.codes)
    scale = _numpy(table.scale).astype("float64")
    zero = table.zero_point
    if zero is not None:
        zero = ops.convert_to_numpy(zero).astype("int64")
    if orientation == "untied":
        # As a tied table: `(V, ceil(D/2))`, and `(V, n_groups)`.
        codes, scale = codes.T, scale.T
        zero = None if zero is None else zero.T
    packed = codes.astype("int64")
    q = np.stack([packed & 15, (packed >> 4) & 15], axis=-1)
    q = q.reshape(len(codes), -1)[:, : layer.output_dim]
    q = np.where(q > 7, q - 16, q)
    if zero is None:
        weight = q / scale[:, None]
    else:
        # Column `d` of the table is in group `d // G`.
        groups = np.arange(layer.output_dim) // table.scheme.group_size
        weight = (q - zero[:, groups]) * scale[:, groups]
    dtype = layer.reverse_dtype or layer.compute_dtype
    rounding = ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype
    x = np.asarray(x, "float32").astype(rounding).astype("float64")
    return x.reshape(-1, layer.output_dim) @ weight.T


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _exact_data(layer, m, extreme=None, seed=2):
    """Small integers and powers of two, so that every result is exact.

    The codes differ from the zero points by at most 2, or, with
    `extreme`, they are all at the minimum, all at the maximum, or
    alternate between the two along `D`, with the zero points at both
    ends of the range. Returns `m` rows of inputs with eight nonzero
    entries of +1 or -1 each.
    """
    view, orientation = _stored(layer)
    rng = np.random.default_rng(seed)
    shape = (layer.input_dim, layer.output_dim)
    if extreme is None:
        codes = rng.integers(-1, 2, shape)
    else:
        codes = {
            "minimum": np.full(shape, -8),
            "maximum": np.full(shape, 7),
            "alternating": np.where(np.arange(shape[1]) % 2, 7, -8)
            * np.ones((shape[0], 1), int),
        }[extreme]
    packed = pack_int4(codes.astype("int8"), axis=-1)[0]
    view.codes.assign(
        packed if orientation == "tied" else ops.transpose(packed)
    )
    powers = rng.integers(1, 4, view.scale.shape)
    # A grouped scale multiplies and a per-channel scale divides.
    sign = -1 if view.zero_point is not None else 1
    view.scale.assign(2.0 ** (sign * powers))
    if view.zero_point is not None:
        if extreme is None:
            zero = rng.integers(-1, 2, view.zero_point.shape)
        else:
            zero = rng.choice([-8, 7], view.zero_point.shape)
        view.zero_point.assign(zero.astype("int8"))
    x = np.zeros((m, shape[1]), "float32")
    for row in x:
        row[rng.choice(shape[1], 8, replace=False)] = rng.choice([-1, 1], 8)
    return x


def _compile_jobs():
    """The compilations of `test_compiles_for_cuda`.

    Each default row of `_LOGITS_CONFIGS` compiles with the largest K
    tile, 128, which needs the most shared memory: in float16 and in
    float32 (IEEE and TF32 dots), for tied and untied grouped tables. A
    bfloat16 tile has the bytes of a float16 tile, and a per-channel table
    reads no scale or zero point in the loop. So bfloat16 and split-K
    compile for the first row with a K tile of 32, and per-channel tables
    for the last row only. One more job compiles a bfloat16 decode launch
    on a tied grouped table, as `_compile_for_cuda` specializes it.
    """
    jobs = []

    def add(
        dtype,
        orientation,
        per_channel,
        m,
        precision="ieee",
        decode=False,
        **kwargs,
    ):
        config = quantized.kernel_config(
            None, dtype, m, None, quantized._LOGITS_CONFIGS
        )
        config = dataclasses.replace(config, **kwargs)
        job = {
            "dtype": dtype,
            "orientation": orientation,
            "per_channel": per_channel,
            "precision": precision,
            "decode": decode,
            "config": dataclasses.asdict(config),
        }
        jobs.append(job)

    for (dtype, precision), m, orientation in itertools.product(
        (("float16", "ieee"), ("float32", "ieee"), ("float32", "tf32")),
        (1, 17, 65, 257),
        ("tied", "untied"),
    ):
        add(dtype, orientation, False, m, precision, block_k=128)
    for orientation in ("tied", "untied"):
        add("bfloat16", orientation, False, 1, block_k=32)
        for dtype in ("float16", "float32"):
            add(dtype, orientation, True, 257, block_k=128)
    add("float32", "tied", False, 1, block_k=32, split_k=4)
    add("bfloat16", "tied", False, 1, decode=True, block_k=128)
    return jobs


def _compile_for_cuda(capability, jobs):
    """Compiles `_w4a16_kernel` with `PACK_K` for a CUDA device, on any
    host.

    A subprocess with `TRITON_INTERPRET=0` runs it, as in
    `quantized_matmul_test.py`. A launch specializes a value of 1 as a
    constant, and a value that is a multiple of 16 with that divisibility.
    In each job, a stride of 1 is a constant: the stored codes, scale and
    zero point are contiguous along `D` (tied) or along `V` (untied), and a
    per-channel scale is `(V,)`. The other pointers and sizes have the
    divisibility by 16. A real launch can have other values: a decode
    launch has `M = 1`, and in a tied grouped table, the scale and the
    zero point have `stride_sn = n_groups`, which is 18 for Gemma2-2B. So
    the decode job has `M = 1` as a constant, and no divisibility on
    `n_groups` and `stride_sn`. Prints the shared memory of each job, in
    bytes, as a JSON list.
    """
    kernel = quantized._w4a16_kernel
    target = GPUTarget("cuda", 10 * capability[0] + capability[1], 32)
    shared = []
    for job in jobs:
        config = quantized.KernelConfig(**job["config"])
        x = POINTERS[job["dtype"]]
        per_channel = job["per_channel"]
        # Per-channel tables pass the scale as the zero point, and each
        # table passes it as the group index.
        signature = {
            "x_ptr": x,
            "codes_ptr": "*i8",
            "scale_ptr": "*fp32",
            "zero_ptr": "*fp32" if per_channel else "*i8",
            "g_idx_ptr": "*fp32",
            "y_ptr": "*fp32" if config.split_k > 1 else x,
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "n_groups": "i32",
            "group_size": "i32",
            # `w4a16_logits` passes 0: it reads the codes through the
            # strides.
            "NH": "i32",
            "stride_cb": "i32",
            "stride_cn": "i32",
            "stride_sg": "i32",
            "stride_sn": "i32",
        }
        if job["orientation"] == "tied":
            ones = ["stride_cb", "stride_sg"]
        else:
            ones = ["stride_cn", "stride_sn"]
        if per_channel:
            ones[1] = "stride_sn"
        unaligned = []
        if job["decode"]:
            ones.append("M")
            unaligned = ["n_groups", "stride_sn"]
        constants = dict.fromkeys(ones, 1)
        divisible = {
            (kernel.arg_names.index(name),): [["tt.divisibility", 16]]
            for name in signature
            if name not in constants and name not in unaligned
        }
        constants.update(
            {
                "XOR": 8,
                "PER_CHANNEL": per_channel,
                "PACK_K": True,
                "CONTIGUOUS": True,
                "UPCAST_DOT": False,
                "INPUT_PRECISION": job["precision"],
                "SPLIT_K": config.split_k,
                "BM": config.block_m,
                "BNH": config.block_nh,
                "BK": config.block_k,
            }
        )
        signature.update(dict.fromkeys(constants, "constexpr"))
        compiled = quantized.triton.compile(
            ASTSource(kernel, signature, constants, divisible),
            target=target,
            options={
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
            },
        )
        shared.append(compiled.metadata.shared)
    print(json.dumps(shared))


class W4A16LogitsTest(testing.TestCase):
    """Direct calls of the kernel, on the tables of real layers."""

    @parameterized.named_parameters(
        named_product(
            orientation=["tied", "untied"],
            block=sorted(BLOCKS),
            dtype=sorted(TOLERANCES),
            shape=["decode", "ragged"],
        )
    )
    def test_matches_the_oracle(self, orientation, block, dtype, shape):
        """G1, G2, G4 and G5 against the oracle and the reference path.

        The layer is float32 with a reverse dtype, so the 16-bit cases
        have a float32 scale with 16-bit activations.
        """
        layer = _layer(
            orientation == "tied", BLOCKS[block], shape, reverse_dtype=dtype
        )
        x = _inputs(shape)
        oracle = _oracle(layer, x)
        with kernels.policy("off"):
            reference = _numpy(layer(x, reverse=True))
        y = _direct(layer, x)
        self.assertEqual(y.dtype, getattr(torch, dtype))
        again = _direct(layer, x)
        y = _numpy(y)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAllEqual(_numpy(again), y)
        floor, bound = TOLERANCES[dtype]
        err = _err(y, oracle)
        self.assertLessEqual(err, 1.25 * _err(reference, oracle) + floor)
        self.assertLessEqual(err, bound)

    @parameterized.named_parameters(
        named_product(
            orientation=["tied", "untied"],
            block=sorted(BLOCKS),
            dtype=["float32", "bfloat16"],
        )
    )
    def test_exact_data(self, orientation, block, dtype):
        """G3: small integers and powers of two give the oracle exactly."""
        layer = _layer(
            orientation == "tied", BLOCKS[block], "ragged", reverse_dtype=dtype
        )
        x = _exact_data(layer, 33)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x, reverse=True)), oracle)
        config = quantized.KernelConfig(32, 32)
        y = _direct(layer, x, config=config)
        self.assertEqual(y.dtype, getattr(torch, dtype))
        self.assertAllEqual(_numpy(y), oracle)

    @parameterized.named_parameters(
        named_product(
            orientation=["tied", "untied"],
            extreme=["minimum", "maximum", "alternating"],
        )
    )
    def test_extreme_codes(self, orientation, extreme):
        """G3 in float32 with codes and zero points at the range ends."""
        layer = _layer(orientation == "tied", 128, "ragged")
        x = _exact_data(layer, 33, extreme)
        oracle = _oracle(layer, x).astype("float32")
        self.assertAllEqual(_numpy(_direct(layer, x)), oracle)

    @parameterized.named_parameters(("tied", "tied"), ("untied", "untied"))
    def test_split_k(self, orientation):
        """G3 and G5 with split-K: the partials add in a fixed order."""
        layer = _layer(orientation == "tied", 128, "ragged")
        x = _exact_data(layer, 33)
        config = quantized.KernelConfig(32, 32, split_k=2)
        y = _numpy(_direct(layer, x, config=config))
        self.assertAllEqual(y, _oracle(layer, x).astype("float32"))
        self.assertAllEqual(_numpy(_direct(layer, x, config=config)), y)

    @parameterized.named_parameters(("g128", 128), ("per_channel", -1))
    def test_orientations_and_input_forms(self, block_size):
        """The two orientations of one table give the same bits, and so do
        strided inputs. `M = 0` gives `(0, V)`."""
        layer = _layer(True, block_size, "ragged")
        x = ops.convert_to_tensor(_inputs("ragged"))
        expected = _numpy(_direct(layer, x))
        table, _ = _table(layer)

        def untied(tensor):
            if tensor is None or len(tensor.shape) == 1:
                return tensor
            return tensor.t().contiguous()

        y = quantized.w4a16_logits(
            x,
            untied(table.codes),
            untied(table.scale),
            untied(table.zero_point),
            group_size=table.scheme.group_size,
            per_channel=table.scheme.group_size is None,
            orientation="untied",
        )
        self.assertAllEqual(_numpy(y), expected)
        strided = x.new_zeros((x.shape[0], 2 * x.shape[1]))
        strided[:, ::2] = x
        self.assertFalse(strided[:, ::2].is_contiguous())
        self.assertAllEqual(_numpy(_direct(layer, strided[:, ::2])), expected)
        empty = _direct(layer, x[:0])
        self.assertEqual(tuple(empty.shape), (0, layer.input_dim))
        self.assertEqual(empty.dtype, torch.float32)

    @parameterized.named_parameters(
        ("group_48", 48, None),  # The K tile of the group size, 32.
        ("group_64_tile_128", 64, 128),
        ("group_128_tile_16", 128, 16),
        ("per_channel_tile_16", -1, 16),
    )
    def test_frontend_refuses_the_k_tile(self, block_size, block_k):
        """The frontend refuses a K tile of fewer than 32 rows or one that
        spans two groups, before the launch, also for `M = 0`."""
        layer = _layer(True, block_size)
        config = None
        if block_k is not None:
            config = quantized.KernelConfig(16, 32, block_k=block_k)
        for m in (6, 0):
            x = _random((m, layer.output_dim), 1)
            with self.assertRaisesRegex(ValueError, "at least 32 rows"):
                _direct(layer, x, config=config)

    @parameterized.named_parameters(
        ("tied_float32", "tied", 128, "float32", 1),
        ("untied_bfloat16_per_channel", "untied", -1, "bfloat16", 1),
        ("tied_float16_split_k", "tied", 128, "float16", 2),
    )
    def test_op_registration(self, orientation, block_size, dtype, split_k):
        """The schema and the fake implementation of the custom op."""
        layer = _layer(orientation == "tied", block_size, reverse_dtype=dtype)
        table, _ = _table(layer)
        x = ops.cast(ops.convert_to_tensor(_random((3, 256), 1)), dtype)
        per_channel = table.scheme.group_size is None
        zero = table.scale if per_channel else table.zero_point
        group_size = 256 if per_channel else 128
        # The tensors, then group size, per-channel, orientation, the
        # blocks, warps, stages, split-K and input precision.
        args = (x, table.codes, table.scale, zero, group_size, per_channel)
        args += (orientation, 16, 32, 128, 4, 3, split_k, "ieee")
        torch.library.opcheck(
            torch.ops.keras.w4a16_logits.default,
            args,
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_compiles_for_cuda(self):
        """Each variant compiles for sm_80, sm_89 and sm_90, and each
        default row fits in the shared memory of one block there.

        Subprocesses with `TRITON_INTERPRET=0` compile the jobs of each
        device in parallel. The interpreter does not apply the rules of
        the compiler.
        """
        root = os.path.dirname(os.path.dirname(keras.__file__))
        env = dict(
            os.environ,
            KERAS_BACKEND="torch",
            PYTHONPATH=root,
            TRITON_INTERPRET="0",
        )
        code = (
            "import json, sys\n"
            "from keras.src.backend.torch.ops import quantized_logits_test\n"
            "quantized_logits_test._compile_for_cuda(\n"
            "    json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
            ")\n"
        )
        jobs = _compile_jobs()
        processes = []
        for capability in SHARED_MEMORY:
            arguments = [json.dumps(capability), json.dumps(jobs)]
            process = subprocess.Popen(
                [sys.executable, "-c", code, *arguments],
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((capability, process))
        for capability, process in processes:
            stdout, stderr = process.communicate(timeout=600)
            with self.subTest(capability=capability):
                self.assertEqual(process.returncode, 0, stderr[-4000:])
                shared = json.loads(stdout.splitlines()[-1])
                self.assertLen(shared, len(jobs))
                for job, size in zip(jobs, shared):
                    limit = SHARED_MEMORY[capability]
                    self.assertLessEqual(size, limit, job)

    @pytest.mark.skipif(
        backend.backend() != "torch" or not torch.cuda.is_available(),
        reason="Measures the memory of a CUDA device.",
    )
    def test_tied_table_has_no_copy_on_cuda(self):
        """A routed call on a tied table allocates no temporary of the
        size of the table: the kernel reads the table through its strides.
        The reference path dequantizes the whole table."""
        layer = _layer(True, 128, dim=1024, vocabulary=8192)
        x = ops.convert_to_tensor(_random((1, 1024), 1))
        table_bytes = layer._embeddings.value.nbytes

        def peak(policy):
            with kernels.policy(policy):
                layer(x, reverse=True)  # Compiles the kernel.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.memory_allocated()
                layer(x, reverse=True)
                torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - start

        kernels.route_counts.clear()
        routed = peak("require")
        self.assertEqual(dict(kernels.route_counts), {"triton": 2})
        self.assertLess(routed, table_bytes // 4)
        self.assertGreater(peak("off"), table_bytes)


class TritonLogitsRouteTest(testing.TestCase):
    """The checks of the route, and the route on hardware. The route suites
    of `lookup_test.py` run it through real layers."""

    @pytest.mark.skipif(
        backend.backend() != "torch" or not torch.cuda.is_available(),
        reason="Runs the compiled kernel on a CUDA device.",
    )
    def test_require_runs_the_kernel_on_cuda(self):
        """G8 on hardware: `"require"` runs the Triton logits kernel."""
        layer = _layer(True, 128, reverse_dtype="bfloat16")
        x = _inputs("layer")
        activities = [torch.profiler.ProfilerActivity.CUDA]
        kernels.route_counts.clear()
        with kernels.policy("require"):
            with torch.profiler.profile(activities=activities) as profile:
                y = layer(x, reverse=True)
                torch.cuda.synchronize()
        self.assertEqual(dict(kernels.route_counts), {"triton": 1})
        names = [event.key for event in profile.key_averages()]
        self.assertTrue(any("_w4a16_kernel" in name for name in names))
        direct = ops.reshape(_direct(layer, x), y.shape)
        self.assertAllEqual(_bits(y), _bits(direct))

    @parameterized.named_parameters(
        ("aligned", 256, 384, 128, True),
        ("group_256", 512, 384, 256, True),
        ("one_group", 256, 384, 512, True),
        ("per_channel", 256, 384, None, True),
        ("vocabulary_200", 256, 200, 128, False),
        ("dim_192", 192, 384, 128, False),
        ("group_48", 256, 384, 48, False),
        ("group_64", 256, 384, 64, False),
        ("group_96", 256, 384, 96, False),
    )
    def test_logits_gate(self, d, v, group, expected):
        """The shape check of the logits: `D` and `V` multiples of 128, and
        groups of 128 or 256 rows, one group or a per-channel scale. The
        kernel takes no float64 compute."""
        (kernel,) = quantized.KERNELS
        self.assertNotIn("float64", kernel.dtypes)
        available = kernel.logits_available("cuda-sm89", 2, d, v, group, "")
        self.assertEqual(available, expected)

    def test_gate(self):
        """The route passes a CUDA device, with groups of 128 or 256 rows,
        one group or a per-channel scale, and refuses int32 overflows. The
        logits have their own configuration rows."""
        available = kernels._LOGITS_ROUTES["triton"].available
        device_class = "cuda-sm89"
        _, dim, vocabulary = SHAPES["layer"]
        for block_size, d in (
            (128, dim),
            (256, 2 * dim),
            (512, dim),
            (-1, dim),
        ):
            table, _ = _stored(_layer(True, block_size, dim=d))
            self.assertTrue(
                available(device_class, 6, d, vocabulary, table, "float32"),
                block_size,
            )
        m = 2**31 // vocabulary + 1
        self.assertFalse(
            available(device_class, m, dim, vocabulary, table, "float32")
        )
        rows = quantized._LOGITS_CONFIGS
        config = quantized.kernel_config((8, 9), "float32", 300, 128, rows)
        self.assertIs(config, rows[(None, "float32", "257+", None)])
