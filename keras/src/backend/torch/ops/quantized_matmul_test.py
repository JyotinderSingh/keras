"""Triton K1 (`quantized.w4a16_matmul`) against the float64 oracle.

The module runs where Triton imports, on a CUDA device or under the
Triton interpreter (`TRITON_INTERPRET=1`, which the torch CPU CI job
sets). The interpreter computes a wrong bfloat16 `tl.dot`, so there the
bfloat16 cases cast the dot operands to float32 (`upcast_dot`). It also
rounds float32 to bfloat16 toward zero, not to the nearest value. The
interpreter runs the kernel as Python, so it does not apply the rules
of the compiler: a test compiles the kernel for CUDA devices in a
subprocess. The weights come from real quantized layers: int4 (int8
storage), and GPTQ and AWQ (uint8 storage).
"""

import dataclasses
import itertools
import json
import os
import subprocess
import sys

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
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.testing.test_utils import named_product

INTERPRETED = False
if backend.backend() == "torch":
    import torch
    from torch._dynamo.testing import CompileCounter

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
    SKIP = "Triton K1 is a kernel of the torch backend."

pytestmark = pytest.mark.skipif(SKIP is not None, reason=str(SKIP))

# name -> (mode, group size, order of the groups). A group size of -1 is
# per-channel int4, and one group for GPTQ. The groups are contiguous, in
# GPTQ's activation order, or contiguous groups in a random row order. Only
# GPTQ with several groups reads `g_idx`, so only it can hold groups that
# are not contiguous. The other grouped modes pass no `g_idx`, and the
# kernel runs its `CONTIGUOUS` variant.
MODES = {
    "int4_g32": ("int4", 32, None),
    "int4_g128": ("int4", 128, None),
    "gptq_g32_shuffled": ("gptq", 32, "shuffled"),
    "int4_per_channel": ("int4", -1, None),
    "gptq_g64": ("gptq", 64, None),
    "gptq_one_group": ("gptq", -1, None),
    "gptq_act_order": ("gptq", 64, "activation"),
    "awq_g64": ("awq", 64, None),
}
# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
LAYERS = {
    # Ragged `M` and `K`, and an odd `N` that is not a multiple of a block.
    "dense_ragged": (None, 75, (33, 200)),
    "dense_decode": (None, 128, (1, 256)),
    "query_ndh": ("btd,ndh->btnh", (None, 4, 16), (2, 5, 128)),
    "output_nhd": ("btnh,nhd->btd", (None, 96), (2, 5, 4, 32)),
}
# Compute dtype -> (G1 floor, G2 bound), by dot precision. The
# interpreter's bfloat16 rounding toward zero takes up to one bfloat16 ulp
# (2^-7) of the output, not half of one, so its floor is that ulp.
TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "float16": (2e-4, 2**-9),
    "bfloat16": (2**-7 if INTERPRETED else 1e-3, 2**-7),
}
POLICIES = {
    "float32": "float32",
    "float16": "mixed_float16",
    "bfloat16": "mixed_bfloat16",
}
# Compute capability -> the shared memory of one block, in bytes. The
# gate takes sm_75 for float16 only.
SHARED_MEMORY = {
    (7, 5): 65536,
    (8, 0): 166912,
    (8, 6): 101376,
    (8, 9): 101376,
    (9, 0): 232448,
}
POINTERS = {"bfloat16": "*bf16", "float16": "*fp16", "float32": "*fp32"}
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
# (per-channel, signed): per-channel int4, grouped int4, GPTQ and AWQ.
VARIANTS = ((True, True), (False, True), (False, False))


def _layer(case, mode, dtype="float32", bias=False, seed=0):
    """A layer of `LAYERS[case]`, quantized in the `MODES` entry `mode`."""
    equation, output, input_shape = LAYERS[case]
    policy = POLICIES[dtype]
    if equation is None:
        layer = layers.Dense(output, use_bias=bias, dtype=policy)
    else:
        bias_axes = equation[-1] if bias else None
        layer = layers.EinsumDense(
            equation, output, bias_axes=bias_axes, dtype=policy
        )
    layer.build(input_shape)
    rng = np.random.default_rng(seed)
    layer._kernel.assign(rng.standard_normal(layer._kernel.shape))
    name, group_size, order = MODES[mode]
    if name == "int4":
        config = Int4QuantizationConfig(block_size=group_size)
        layer.quantize("int4", config=config)
    else:
        kwargs = {"activation_order": True} if order == "activation" else {}
        config = calibration_config(name, group_size=group_size, **kwargs)
        batch = rng.standard_normal((16,) + input_shape[1:]).astype("f4")
        calibrate_layer(layer, config, batch)
    if order == "shuffled":
        g_idx = ops.convert_to_numpy(layer.g_idx)
        layer.g_idx.assign(rng.permutation(g_idx))
    return layer


def _inputs(case, seed=1):
    shape = LAYERS[case][2]
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _view(layer):
    """The layer's weight view over the tensors that its forward reads."""
    with AutocastScope(layer.compute_dtype):
        view = layer._quantized_weight()
        return view.with_tensors(view.read_tensors())


def _call(layer, x, form=None, **kwargs):
    """`w4a16_matmul` of the `(M, K)` tensor `x` with the layer's weight.

    `form` gives the tensor that the call passes for each stored tensor.
    """
    view = _view(layer)

    def operand(tensor):
        return tensor if tensor is None or form is None else form(tensor)

    return quantized.w4a16_matmul(
        x,
        operand(view.codes),
        operand(view.scale),
        operand(view.zero_point),
        operand(view.g_idx),
        group_size=view.scheme.group_size,
        signed=view.codes.dtype == torch.int8,
        per_channel=view.scheme.group_size is None,
        awq_scales=operand(view.input_scales),
        **kwargs,
    )


def _direct(layer, x, **kwargs):
    """A direct kernel call on the inputs of the layer: `(M, N)`."""
    x = ops.convert_to_tensor(x, layer.compute_dtype)
    rows = layer._quantized_weight().codes.shape[0]
    return _call(layer, ops.reshape(x, (-1, rows)), **kwargs)


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _oracle(layer, x):
    view = _view(layer)

    def read(tensor):
        return None if tensor is None else _numpy(tensor)

    return oracle_contract(
        x,
        ops.convert_to_numpy(view.codes),
        read(view.scale),
        columns=view.layout.orig_len,
        kernel_shape=view.shape,
        permutation=view.permutation,
        zero_point=read(view.zero_point),
        g_idx=read(view.g_idx),
        group_size=view.scheme.group_size,
        input_scales=read(view.input_scales),
        equation=getattr(layer, "equation", None),
        dtype=layer.compute_dtype,
    ).reshape(-1, view.layout.orig_len)


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _exact_data(layer, m, extreme=None, seed=2):
    """Small integers and powers of two, so that every result is exact.

    The codes differ from the zero points by at most 2, or, with
    `extreme`, they are all at the minimum, all at the maximum, or
    alternate between the two, with the zero points at both ends of the
    range. Returns `m` rows of inputs with eight nonzero entries of +1 or
    -1 each.
    """
    view = layer._quantized_weight()
    rng = np.random.default_rng(seed)
    rows, columns = view.codes.shape[0], view.layout.orig_len
    signed = view.codes.dtype == "int8"
    low, high = (-8, 7) if signed else (0, 15)
    center = 0 if signed else 8
    if extreme is None:
        codes = center + rng.integers(-1, 2, (rows, columns))
    else:
        codes = {
            "minimum": np.full((rows, columns), low),
            "maximum": np.full((rows, columns), high),
            "alternating": np.where(np.arange(columns) % 2, high, low)
            * np.ones((rows, 1), int),
        }[extreme]
    dtype = "int8" if signed else "uint8"
    view.codes.assign(pack_int4(codes.astype(dtype), axis=-1, dtype=dtype)[0])
    powers = rng.integers(1, 4, view.scale.shape)
    # A grouped scale multiplies and a per-channel scale divides.
    sign = -1 if view.zero_point is not None else 1
    view.scale.assign(2.0 ** (sign * powers))
    if view.zero_point is not None:
        if extreme is None:
            zero = center + rng.integers(-1, 2, view.zero_point.shape)
        else:
            zero = rng.choice([low, high], view.zero_point.shape)
        view.zero_point.assign(zero.astype(dtype))
    if view.input_scales is not None:
        view.input_scales.assign(rng.choice([1.0, 2.0], (rows,)))
    x = np.zeros((m, rows), "float32")
    for row in x:
        row[rng.choice(rows, 8, replace=False)] = rng.choice([-1, 1], 8)
    return x


def _compile_jobs(capability):
    """The compilations of `test_compiles_for_cuda` for one device.

    Each default row of `_CONFIGS` compiles with the largest K tile, 128,
    which needs the most shared memory: in float16 and in float32 (IEEE
    and TF32 dots), for per-channel and grouped int4. A bfloat16 tile has
    the bytes of a float16 tile, and GPTQ and AWQ codes have the bytes of
    int4 codes. So bfloat16, GPTQ and AWQ compile with the K tile 32
    only, as does split-K. Each row of the device compiles with its own K
    tile, or 128 when it has none: a bfloat16 row in float16, and a row
    with `None` in place of `G` per channel and grouped. A grouped
    16-bit row also compiles with a float32 scale, which needs the most
    shared memory. Each row is the one that the frontend fits to the
    device (`fit_config`). Each grouped job compiles twice: with a `g_idx`
    (the check and both loops) and as the `CONTIGUOUS` variant (the fast
    loop only), with the same estimate. The split-K sum compiles for each
    output dtype with 4 splits, and with the splits of each row of the
    device. sm_75 compiles the float16 jobs only.
    """
    jobs = []

    def add(dtype, variant, config, precision="ieee", scale=None, fit=False):
        per_channel, signed = variant
        scale = scale or dtype
        # Per-channel int4 passes its scale as the zero point and as the
        # group index. A grouped `g_idx` is float32.
        sizes = (BYTES[dtype],) + (BYTES[scale],) * 3
        if not per_channel:
            sizes = (BYTES[dtype], BYTES[scale], 1, 4)

        def estimate(config):
            return quantized.shared_bytes(
                config, config.block_k, sizes, per_channel, capability
            )

        if fit:
            config = quantized.fit_config(
                config,
                config.block_k,
                sizes,
                per_channel,
                capability,
                SHARED_MEMORY[capability],
            )
        for contiguous in (False,) if per_channel else (False, True):
            job = {
                "dtype": dtype,
                "scale": scale,
                "per_channel": per_channel,
                "signed": signed,
                "contiguous": contiguous,
                "precision": precision,
                "config": dataclasses.asdict(config),
                "estimate": estimate(config),
            }
            if job not in jobs and (capability >= (8, 0) or dtype == "float16"):
                jobs.append(job)

    def add_sum(dtype, split_k):
        job = {"kernel": "splitk_sum", "dtype": dtype, "split_k": split_k}
        if job not in jobs and (capability >= (8, 0) or dtype == "float16"):
            jobs.append(job)

    for (dtype, precision), m, variant in itertools.product(
        (("float16", "ieee"), ("float32", "ieee"), ("float32", "tf32")),
        (1, 17, 65, 257),
        VARIANTS[:2],
    ):
        config = quantized.kernel_config(None, dtype, m, None)
        config = dataclasses.replace(config, block_k=128)
        add(dtype, variant, config, precision, fit=True)
        if dtype == "float16" and variant == VARIANTS[1]:
            add(dtype, variant, config, precision, "float32", fit=True)
    for dtype, variant in itertools.product(POINTERS, VARIANTS):
        if dtype == "bfloat16" or variant == VARIANTS[2]:
            config = quantized.kernel_config(None, dtype, 1, 32)
            add(dtype, variant, dataclasses.replace(config, block_k=32))
    config = quantized.kernel_config(None, "float32", 1, 32)
    config = dataclasses.replace(config, block_k=32, split_k=4)
    add("float32", VARIANTS[2], config)
    for dtype in POINTERS:
        add_sum(dtype, 4)
    for (device, dtype, _, group), config in quantized._CONFIGS.items():
        if device != capability:
            continue
        if config.split_k > 1:
            add_sum(dtype, config.split_k)
        config = dataclasses.replace(config, block_k=config.block_k or 128)
        dtype = "float16" if dtype == "bfloat16" else dtype
        for variant in VARIANTS[:2] if group is None else VARIANTS[1:2]:
            add(dtype, variant, config, fit=True)
            if dtype != "float32" and variant == VARIANTS[1]:
                add(dtype, variant, config, scale="float32", fit=True)
    return jobs


def _compile_for_cuda(capability, jobs):
    """Compiles `_w4a16_kernel` and `_splitk_sum` for a CUDA device, on
    any host.

    A subprocess with `TRITON_INTERPRET=0` runs it, so that `triton.jit`
    gives a kernel that compiles. The Triton wheel holds `ptxas`, so no
    GPU is necessary. Each pointer and each size has the divisibility by
    16 that a launch specializes on (aligned tensors, sizes that are
    multiples of 16), which needs the most shared memory. Prints the
    shared memory of each job, in bytes, as a JSON list.
    """
    target = GPUTarget("cuda", 10 * capability[0] + capability[1], 32)
    shared = []
    for job in jobs:
        if job.get("kernel") == "splitk_sum":
            kernel = quantized._splitk_sum
            signature = {
                "ws_ptr": "*fp32",
                "y_ptr": POINTERS[job["dtype"]],
                "MN": "i32",
            }
            constants = {
                "SPLIT_K": job["split_k"],
                "BLOCK": quantized._SUM_BLOCK,
            }
            divisible = {
                (kernel.arg_names.index(name),): [["tt.divisibility", 16]]
                for name in signature
            }
            signature.update(dict.fromkeys(constants, "constexpr"))
            compiled = quantized.triton.compile(
                ASTSource(kernel, signature, constants, divisible),
                target=target,
            )
            shared.append(compiled.metadata.shared)
            continue
        kernel = quantized._w4a16_kernel
        config = quantized.KernelConfig(**job["config"])
        x = POINTERS[job["dtype"]]
        scale = POINTERS[job["scale"]]
        codes = "*i8" if job["signed"] else "*u8"
        per_channel = job["per_channel"]
        # Per-channel int4 passes its scale as the zero point and as the
        # group index, and a contiguous call as the group index.
        signature = {
            "x_ptr": x,
            "codes_ptr": codes,
            "scale_ptr": scale,
            "zero_ptr": scale if per_channel else codes,
            "g_idx_ptr": (
                scale if per_channel or job["contiguous"] else "*fp32"
            ),
            "y_ptr": "*fp32" if config.split_k > 1 else x,
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "n_groups": "i32",
            "group_size": "i32",
            "NH": "i32",
            # `w4a16_matmul` passes zeros: it reads no strides.
            "stride_cb": "i32",
            "stride_cn": "i32",
            "stride_sg": "i32",
            "stride_sn": "i32",
        }
        divisible = {
            (kernel.arg_names.index(name),): [["tt.divisibility", 16]]
            for name in signature
        }
        constants = {
            "XOR": 8 if job["signed"] else 0,
            "PER_CHANNEL": per_channel,
            "PACK_K": False,
            "CONTIGUOUS": job["contiguous"],
            "UPCAST_DOT": False,
            "INPUT_PRECISION": job["precision"],
            "SPLIT_K": config.split_k,
            "BM": config.block_m,
            "BNH": config.block_nh,
            "BK": config.block_k,
        }
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


class W4A16MatmulTest(testing.TestCase):
    """Direct calls of the kernel, on the tensors of real layers."""

    @parameterized.named_parameters(
        named_product(
            mode=sorted(MODES),
            case=["dense_ragged", "dense_decode"],
            dtype=sorted(TOLERANCES),
        )
    )
    def test_matches_the_oracle(self, mode, case, dtype):
        """G1, G2, G4 and G5 against the oracle and the reference path."""
        layer = _layer(case, mode, dtype)
        x = _inputs(case)
        oracle = _oracle(layer, x)
        with kernels.policy("off"):
            reference = _numpy(layer(x))
        y = _direct(layer, x, upcast_dot=INTERPRETED)
        self.assertEqual(y.dtype, getattr(torch, dtype))
        again = _direct(layer, x, upcast_dot=INTERPRETED)
        y = _numpy(y)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAllEqual(_numpy(again), y)
        floor, bound = TOLERANCES[dtype]
        err = _err(y, oracle)
        self.assertLessEqual(err, 1.25 * _err(reference, oracle) + floor)
        self.assertLessEqual(err, bound)

    @parameterized.named_parameters(
        named_product(
            mode=sorted(MODES), dtype=["float32", "bfloat16"], split_k=[1, 4]
        )
    )
    def test_exact_data(self, mode, dtype, split_k):
        """G3: small integers and powers of two give the oracle exactly."""
        layer = _layer("dense_ragged", mode, dtype)
        x = _exact_data(layer, 33)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        config = quantized.KernelConfig(32, 32, split_k=split_k)
        y = _direct(layer, x, config=config, upcast_dot=INTERPRETED)
        self.assertEqual(y.dtype, getattr(torch, dtype))
        self.assertAllEqual(_numpy(y), oracle)

    @parameterized.named_parameters(
        named_product(
            mode=["int4_g32", "int4_per_channel", "gptq_act_order", "awq_g64"],
            extreme=["minimum", "maximum", "alternating"],
        )
    )
    def test_extreme_codes(self, mode, extreme):
        """G3 in float32 with codes and zero points at the range ends."""
        layer = _layer("dense_ragged", mode)
        x = _exact_data(layer, 33, extreme)
        oracle = _oracle(layer, x).astype("float32")
        self.assertAllEqual(_numpy(_direct(layer, x)), oracle)

    @parameterized.named_parameters(
        named_product(mode=["gptq_g64", "gptq_act_order"])
    )
    def test_out_of_range_groups(self, mode):
        """The general path clamps each group to `[0, n_groups - 1]`."""
        layer = _layer("dense_ragged", mode)
        x = _exact_data(layer, 33)
        n_groups = layer._quantized_weight().scale.shape[0]
        g_idx = np.array(ops.convert_to_numpy(layer.g_idx))
        g_idx[::7], g_idx[3::7] = -1, n_groups + 3
        layer.g_idx.assign(np.clip(g_idx, 0, n_groups - 1))
        oracle = _oracle(layer, x).astype("float32")
        layer.g_idx.assign(g_idx)
        self.assertAllEqual(_numpy(_direct(layer, x)), oracle)

    @parameterized.named_parameters(
        ("group_16_tile_32", 16, 32), ("group_48_tile_32", 48, 32)
    )
    def test_groups_that_a_k_tile_spans(self, group_size, block_k):
        """Without `g_idx`, a K tile that spans two groups takes the
        general loop on the groups `k // G`: G3 holds."""
        _, output, input_shape = LAYERS["dense_ragged"]
        layer = layers.Dense(output, use_bias=False)
        layer.build(input_shape)
        config = Int4QuantizationConfig(block_size=group_size)
        layer.quantize("int4", config=config)
        self.assertIsNone(layer._quantized_weight().g_idx)
        x = _exact_data(layer, 33)
        oracle = _oracle(layer, x).astype("float32")
        config = quantized.KernelConfig(32, 32, block_k=block_k)
        self.assertAllEqual(_numpy(_direct(layer, x, config=config)), oracle)

    def test_contiguous_follows_the_k_tile(self):
        """`CONTIGUOUS` holds for a call with no `g_idx` when each K tile
        lies in one group: `G >= K`, or a K tile that divides `G`. A call
        with a `g_idx` (GPTQ with several groups) never takes it. The
        default K tile gives it for groups of 32, 64 and 128."""
        g_idx = torch.zeros((256,))
        cases = (
            # g_idx, group size, K, K tile, CONTIGUOUS
            (None, 16, 256, 16, True),
            (None, 16, 256, 32, False),
            (None, 48, 256, 16, True),
            (None, 48, 256, 32, False),
            (None, 24, 256, 16, False),
            (None, 96, 200, 32, True),
            (None, 256, 200, 128, True),
            (None, 64, 256, 128, False),
            (g_idx, 64, 256, 64, False),
            (g_idx, 256, 256, 128, False),
        )
        for g, group_size, k, block_k, expected in cases:
            self.assertEqual(
                quantized._contiguous(g, group_size, k, block_k),
                expected,
                (group_size, k, block_k, g is None),
            )
        for group_size in (32, 64, 128):
            block_k = quantized._block_k(group_size, 4096)
            self.assertTrue(
                quantized._contiguous(None, group_size, 4096, block_k)
            )

    def test_split_k_is_deterministic(self):
        layer = _layer("dense_decode", "gptq_g64")
        x = _inputs("dense_decode")
        config = quantized.KernelConfig(16, 32, split_k=4)
        y = _numpy(_direct(layer, x, config=config))
        self.assertAllEqual(_numpy(_direct(layer, x, config=config)), y)
        self.assertLessEqual(_err(y, _oracle(layer, x)), 1e-5)

    @parameterized.named_parameters(
        named_product(mode=["int4_g32", "int4_per_channel", "awq_g64"])
    )
    def test_input_forms(self, mode):
        """Strided, sliced and transposed inputs, strided views of the
        stored tensors, and `M = 0`."""
        layer = _layer("dense_ragged", mode)
        m, k = LAYERS["dense_ragged"][2]
        x = ops.convert_to_tensor(_inputs("dense_ragged"))
        expected = _numpy(_direct(layer, x))
        strided = x.new_zeros((m, 2 * k))
        strided[:, ::2] = x
        sliced = x.new_zeros((m + 1, k + 3))
        sliced[1:, 3:] = x
        forms = {
            "strided": strided[:, ::2],
            "sliced": sliced[1:, 3:],
            "transposed": x.t().contiguous().t(),
        }
        for name, form in forms.items():
            with self.subTest(name):
                self.assertFalse(form.is_contiguous())
                self.assertAllEqual(_numpy(_call(layer, form)), expected)

        def stored_form(tensor):
            # The values of `tensor`, with a stride of 2 along the last axis.
            view = torch.repeat_interleave(tensor, 2, dim=-1)[..., ::2]
            self.assertFalse(view.is_contiguous())
            return view

        y = _call(layer, x, form=stored_form)
        self.assertAllEqual(_numpy(y), expected)
        empty = _call(layer, x[:0])
        self.assertEqual(tuple(empty.shape), (0, 75))
        self.assertEqual(empty.dtype, torch.float32)

    @parameterized.named_parameters(
        ("float32", "float32", 1, "int4_g32"),
        ("float16", "float16", 1, "int4_g32"),
        ("bfloat16", "bfloat16", 1, "int4_g32"),
        ("float16_split_k", "float16", 4, "int4_g32"),
        ("float32_g_idx", "float32", 1, "gptq_g32_shuffled"),
    )
    def test_op_registration(self, dtype, split_k, mode):
        """The schema and the fake implementation of the custom op, with
        contiguous groups (int4) and with a `g_idx` (GPTQ)."""
        layer = _layer("dense_decode", mode, dtype)
        view = _view(layer)
        x = ops.convert_to_tensor(_inputs("dense_decode"), layer.compute_dtype)
        contiguous = view.g_idx is None
        # With contiguous groups, the scale stands in for `g_idx`.
        g_idx = view.scale if contiguous else view.g_idx
        tensors = (x, view.codes, view.scale, view.zero_point, g_idx)
        # group size, signed, per-channel, the blocks, warps, stages,
        # split-K, input precision, upcast, contiguous groups.
        options = (32, mode.startswith("int4"), False, 16, 32, 32, 4, 3)
        options += (split_k, "ieee", False, contiguous)
        torch.library.opcheck(
            torch.ops.keras.w4a16_matmul.default,
            tensors + options,
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_config_table(self):
        """The most specific row: device, dtype, M bucket and `G`."""
        row = quantized.KernelConfig(16, 16, block_k=32, split_k=2)
        default = quantized._CONFIGS[(None, None, "1-16", None)]
        # A capability with no rows of its own.
        key = ((1, 0), "bfloat16", "1-16", 128)
        quantized._CONFIGS[key] = row
        try:
            config = quantized.kernel_config((1, 0), "bfloat16", 16, 128)
            self.assertIs(config, row)
            config = quantized.kernel_config((1, 0), "bfloat16", 16, 64)
            self.assertIs(config, default)
            config = quantized.kernel_config((1, 0), "bfloat16", 17, 128)
            self.assertIs(
                config, quantized._CONFIGS[(None, None, "17-64", None)]
            )
            # A float32 tile needs more shared memory: its own default row.
            config = quantized.kernel_config((1, 0), "float32", 65, 128)
            self.assertIs(
                config, quantized._CONFIGS[(None, "float32", "65-256", None)]
            )
        finally:
            del quantized._CONFIGS[key]
        # The sm_89 rows: `G` 128 has its own row, and per-channel int4
        # (`G` None) and any other `G` take the row with `None`.
        rows = quantized._CONFIGS
        grouped = rows[((8, 9), "bfloat16", "1-16", 128)]
        other = rows[((8, 9), "bfloat16", "1-16", None)]
        for group, want in ((128, grouped), (None, other), (64, other)):
            config = quantized.kernel_config((8, 9), "bfloat16", 1, group)
            self.assertIs(config, want)
        # A row that serves every `G` has a K tile that divides each `G`.
        for (_, _, _, group), config in rows.items():
            if group is None and config.block_k is not None:
                self.assertEqual(32 % config.block_k, 0)

    def test_fit_config(self):
        """A row that does not fit in the shared memory of one block gets
        fewer columns, and a row that fits does not change."""
        row = quantized._CONFIGS[(None, None, "257+", None)]
        # bfloat16 `x` with a float32 scale, an int8 zero point and a
        # float32 `g_idx`. The general loop stages each of these tiles, so
        # fewer stages do not make the estimate smaller.
        sizes = (2, 4, 1, 4)
        for capability, limit in SHARED_MEMORY.items():
            if capability < (8, 0):
                # No asynchronous copies: the row does not change.
                config = quantized.fit_config(
                    row, 128, sizes, False, capability, limit
                )
                self.assertIs(config, row)
                continue
            with self.subTest(capability=capability):
                config = quantized.fit_config(
                    row, 128, sizes, False, capability, limit
                )
                # The 99 KB of sm_86 and sm_89 take 32 columns. sm_80 and
                # sm_90 keep the row.
                if limit == SHARED_MEMORY[(8, 9)]:
                    want = dataclasses.replace(row, block_nh=32)
                    self.assertEqual(config, want)
                else:
                    self.assertIs(config, row)
                estimate = quantized.shared_bytes(
                    config, 128, sizes, False, capability
                )
                self.assertLessEqual(estimate, limit)
        # A bfloat16 scale fits in 99 KB.
        config = quantized.fit_config(
            row, 128, (2, 2, 1, 4), False, (8, 6), 101376
        )
        self.assertIs(config, row)
        # Per channel, the stages come first.
        row = quantized.KernelConfig(128, 64, num_stages=4)
        config = quantized.fit_config(
            row, 128, (2, 2, 2, 2), True, (8, 6), 101376
        )
        self.assertEqual(config, dataclasses.replace(row, num_stages=3))

    def test_gate(self):
        """Gate 5 passes for this device and refuses int32 overflows."""
        layer = _layer("dense_ragged", "int4_g32")
        available = kernels._ROUTES["triton"].available
        view = layer._quantized_weight()
        self.assertTrue(available(33, 200, 75, view))
        self.assertFalse(available(2**31 // 200 + 1, 200, 75, view))
        version = quantized.triton.__version__
        self.assertTrue(kernels.tested(version, kernels.TRITON_TESTED))

    def test_float32_precision_follows_torch(self):
        """The precision mapping, in a new process per setter.

        The setting is process-global. Every setter of torch 2.14 sets
        `torch.backends.cuda.matmul.fp32_precision`, and the frontend
        never calls `torch.get_float32_matmul_precision()`.
        """
        setters = {
            "none": ("", "ieee"),
            "cuda_matmul": (
                "torch.backends.cuda.matmul.fp32_precision = 'tf32'",
                "tf32",
            ),
            "global": ("torch.backends.fp32_precision = 'tf32'", "tf32"),
            "allow_tf32": (
                "torch.backends.cuda.matmul.allow_tf32 = True",
                "tf32",
            ),
            "high": ("torch.set_float32_matmul_precision('high')", "tf32"),
        }
        root = os.path.dirname(os.path.dirname(keras.__file__))
        env = dict(os.environ, KERAS_BACKEND="torch", PYTHONPATH=root)
        processes = {}
        for name, (setter, _) in setters.items():
            code = (
                "import torch\n"
                "from keras.src.backend.torch.ops import quantized\n"
                f"{setter}\n"
                "print(quantized.float32_input_precision())\n"
            )
            processes[name] = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        for name, process in processes.items():
            stdout, stderr = process.communicate(timeout=120)
            with self.subTest(name):
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout.split()[-1], setters[name][1])

    def test_compiles_for_cuda(self):
        """Each variant compiles for sm_80, sm_86, sm_89 and sm_90, and
        in float16 for sm_75. Each default row and each row of the device,
        as the frontend fits it, fits in the shared memory of one block
        there, and no job needs more than its estimate (`shared_bytes`).

        Subprocesses with `TRITON_INTERPRET=0` compile the float32 jobs
        and the other jobs of each device in parallel. The interpreter
        does not apply the rules of the compiler, for example one type
        for a name that both branches of a runtime `if` assign.
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
            "from keras.src.backend.torch.ops import quantized_matmul_test\n"
            "quantized_matmul_test._compile_for_cuda(\n"
            "    json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
            ")\n"
        )
        processes = []
        for capability, float32 in itertools.product(
            SHARED_MEMORY, (False, True)
        ):
            jobs = [
                job
                for job in _compile_jobs(capability)
                if (job["dtype"] == "float32") == float32
            ]
            if not jobs:
                continue
            arguments = [json.dumps(capability), json.dumps(jobs)]
            process = subprocess.Popen(
                [sys.executable, "-c", code, *arguments],
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((capability, jobs, process))
        for capability, jobs, process in processes:
            stdout, stderr = process.communicate(timeout=600)
            with self.subTest(capability=capability):
                self.assertEqual(process.returncode, 0, stderr[-4000:])
                shared = json.loads(stdout.splitlines()[-1])
                self.assertLen(shared, len(jobs))
                for job, size in zip(jobs, shared):
                    limit = SHARED_MEMORY[capability]
                    self.assertLessEqual(size, limit, job)
                    if "estimate" in job:
                        self.assertLessEqual(size, job["estimate"], job)


class TritonRouteTest(testing.TestCase):
    """The route through real layers under `force_route("triton")`."""

    @parameterized.named_parameters(
        named_product(
            case=["dense_ragged", "query_ndh", "output_nhd"],
            mode=[
                "int4_g32",
                "int4_per_channel",
                "gptq_g64",
                "gptq_act_order",
                "awq_g64",
            ],
        )
    )
    def test_layer_runs_the_kernel(self, case, mode):
        """G8: the layer output is a direct kernel call, bit for bit. G7:
        the input gradient is the one of the wrapped reference path."""
        layer = _layer(case, mode, bias=True)
        x = _inputs(case)
        kernels.route_counts.clear()
        with kernels.force_route("triton"):
            y = layer(x)
            gradient = input_gradient(layer, x)
        self.assertEqual(dict(kernels.route_counts), {"triton": 2})
        direct = ops.add(ops.reshape(_direct(layer, x), y.shape), layer.bias)
        self.assertAllEqual(_numpy(y), _numpy(direct))
        with kernels.force_route("reference_wrapped"):
            self.assertAllEqual(gradient, input_gradient(layer, x))

    @parameterized.named_parameters(
        ("float16", "float16"), ("bfloat16", "bfloat16")
    )
    def test_mixed_precision_layer_runs_the_kernel(self, dtype):
        """G8 with the scales read in the compute dtype."""
        layer = _layer("query_ndh", "awq_g64", dtype)
        x = _inputs("query_ndh")
        with kernels.force_route("triton"):
            y = layer(x)
        direct = ops.reshape(_direct(layer, x), y.shape)
        self.assertAllEqual(_numpy(y), _numpy(direct))

    def test_compiled_layer_is_one_graph(self):
        """Under `torch.compile`, a float32 layer with the route is one
        graph: the frontend reads the float32 precision as a constant."""
        layer = _layer("dense_ragged", "int4_g32", bias=True)
        x = ops.convert_to_tensor(_inputs("dense_ragged"))
        counter = CompileCounter()
        torch._dynamo.reset()
        with kernels.force_route("triton"):
            y = torch.compile(lambda v: layer(v), backend=counter)(x)
            expected = layer(x)
        self.assertEqual(counter.frame_count, 1)
        self.assertAllEqual(_numpy(y), _numpy(expected))

    @pytest.mark.skipif(
        backend.backend() != "torch" or not torch.cuda.is_available(),
        reason="Runs the compiled kernel on a CUDA device.",
    )
    def test_require_runs_the_kernel_on_cuda(self):
        """G8 on hardware: `"require"` runs the Triton kernel."""
        layer = _layer("dense_ragged", "int4_g128", "bfloat16")
        x = _inputs("dense_ragged")
        activities = [torch.profiler.ProfilerActivity.CUDA]
        kernels.route_counts.clear()
        with kernels.policy("require"):
            with torch.profiler.profile(activities=activities) as profile:
                y = layer(x)
                torch.cuda.synchronize()
        self.assertEqual(dict(kernels.route_counts), {"triton": 1})
        names = [event.key for event in profile.key_averages()]
        self.assertTrue(any("_w4a16_kernel" in name for name in names))
        self.assertAllEqual(_numpy(y), _numpy(_direct(layer, x)))
