"""Mosaic GPU K1 (`mosaic_gpu_w4a16_matmul`).

Most tests need no GPU. `kernels.mosaic_gpu_lowering` lowers the kernel
for each compute capability, with a check of every untiled GMEM copy.
`kernels.force_route("mosaic_gpu_interpret")`, or an allow-list row of
this CPU for that route, runs the kernel in GPU interpret mode through
real layers, with `jnp.dot` in place of `plgpu.mma`. `GPUTest` runs the
real route on a CUDA GPU that passes the device gate, and the same cases
in interpret mode elsewhere. The module needs JAX 0.11.2 or newer, which
has `plgpu.mma`, so Keras CI skips it.
"""

import contextlib
import dataclasses
import itertools
import json
import os

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.jax.ops import quantized
from keras.src.quantizers import kernels
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.testing.test_utils import named_product
from keras.src.utils.module_utils import tensorflow as tf

pytestmark = pytest.mark.skipif(
    backend.backend() != "jax" or not quantized.plgpu.available,
    reason=(
        "Mosaic GPU K1 needs the JAX backend and JAX 0.11.2 or newer, which "
        f"has `plgpu.mma`. Installed: JAX {jax.__version__}."
    ),
)

CAPABILITIES = ("8.0", "8.6", "8.9", "9.0")
# An `M` in each bucket.
BUCKET_ROWS = {"1-16": 4, "17-64": 33, "65-256": 130, "257+": 300}
# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
CASES = {
    "dense": (None, 128, (3, 256)),
    "qkv_dnh": ("btd,dnh->btnh", (None, 4, 32), (2, 5, 256)),
    "query_ndh": ("btd,ndh->btnh", (None, 4, 32), (2, 5, 256)),
    "output_nhd": ("btnh,nhd->btd", (None, 128), (2, 5, 4, 64)),
}
# keras-hub presets: name -> ((projection, count per block, K, N), ...),
# from their published configs. K is the contracted size, and N the
# product of the free kernel axes.
PRESETS = {
    "gemma2_2b": (
        ("query", 1, 2304, 2048),
        ("key_value", 2, 2304, 1024),
        ("attention_output", 1, 2048, 2304),
        ("gate_up", 2, 2304, 9216),
        ("down", 1, 9216, 2304),
    ),
    "gemma3_1b": (
        ("query", 1, 1152, 1024),
        ("key_value", 2, 1152, 256),
        ("attention_output", 1, 1024, 1152),
        ("gate_up", 2, 1152, 6912),
        ("down", 1, 6912, 1152),
    ),
    "llama2_7b": (
        ("query_key_value", 3, 4096, 4096),
        ("attention_output", 1, 4096, 4096),
        ("gate_up", 2, 4096, 11008),
        ("down", 1, 11008, 4096),
    ),
    "llama3_8b": (
        ("query", 1, 4096, 4096),
        ("key_value", 2, 4096, 1024),
        ("attention_output", 1, 4096, 4096),
        ("gate_up", 2, 4096, 14336),
        ("down", 1, 14336, 4096),
    ),
}
PRESETS["mistral_7b"] = PRESETS["llama3_8b"]
# name -> (BM, BNH, stages, split-K) of a `GPUTest` case. The decode
# configuration needs about 42.5 KB of SMEM, below the 48 KB that a launch
# gets without an opt-in, and the prefill one about 83 KB.
GPU_CONFIGS = {"decode": (16, 32, 3, 2), "prefill": (64, 64, 2, 1)}
# The blocks of each plane in an error map: the worst ones.
MAP_BLOCKS = 8


def _float_layer(case, dtype, autocast=True):
    """A built float layer of `case` with a random kernel."""
    equation, output, input_shape = case
    if equation is None:
        layer = layers.Dense(
            output, use_bias=False, dtype=dtype, autocast=autocast
        )
    else:
        layer = layers.EinsumDense(
            equation, output, dtype=dtype, autocast=autocast
        )
    layer.build(input_shape)
    rng = np.random.default_rng(0)
    layer._kernel.assign(rng.standard_normal(layer._kernel.shape))
    return layer


def _layer(case, mode, group_size, dtype="mixed_bfloat16", autocast=True):
    """A layer of `case` in `mode`: `"int4"` (int8 storage, per-channel
    with a `group_size` of -1), or `"gptq"`, `"gptq_act_order"` or
    `"awq"` (uint8 storage)."""
    layer = _float_layer(case, dtype, autocast)
    if mode == "int4":
        config = Int4QuantizationConfig(block_size=group_size)
        layer.quantize("int4", config=config)
        return layer
    kwargs = {"activation_order": True} if mode == "gptq_act_order" else {}
    config = calibration_config(
        mode.split("_")[0], group_size=group_size, **kwargs
    )
    calibrate_layer(layer, config, _random((16,) + case[2][1:], seed=4))
    return layer


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _oracle(layer, x):
    """The float64 oracle on the stored tensors, with the scales in the
    compute dtype, as the layer reads them."""
    view = layer._quantized_weight()

    def read(tensor, scale=False):
        if tensor is None:
            return None
        value = ops.convert_to_numpy(tensor)
        if scale:
            value = value.astype(ml_dtypes.finfo(layer.compute_dtype).dtype)
        return value

    return oracle_contract(
        x,
        read(view.codes),
        read(view.scale, scale=True),
        columns=view.layout.orig_len,
        kernel_shape=view.shape,
        permutation=view.permutation,
        zero_point=read(view.zero_point),
        g_idx=read(view.g_idx),
        group_size=view.scheme.group_size,
        input_scales=read(view.input_scales, scale=True),
        equation=getattr(layer, "equation", None),
        dtype=layer.compute_dtype,
    )


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _counts(layer, x):
    """The routes that one call of `layer` selects."""
    kernels.route_counts.clear()
    layer(x)
    return dict(kernels.route_counts)


def _direct(layer, x):
    """A direct call of the selected route, on the tensors the layer reads."""
    geometry = layer._quantization_geometry()
    with AutocastScope(layer.compute_dtype):
        view = layer._quantized_weight()
        view = view.with_tensors(view.read_tensors())
        inputs = ops.cast(x, layer.compute_dtype)
        route = kernels.select_route(layer, geometry, view, inputs)
        return kernels.run_route(route, layer, geometry, view, inputs)


def _limit(layer, x, oracle):
    """The largest error that G1 and G2 allow: 1.25 times the reference
    path's plus a floor, and at most an absolute bound."""
    floor, bound = (1e-3, 2**-7)
    if layer.compute_dtype == "float16":
        floor, bound = (2e-4, 2**-9)
    with kernels.policy("off"):
        reference = _numpy(layer(x))
    return min(1.25 * _err(reference, oracle) + floor, bound)


def _check_accuracy(test, layer, x, y):
    """G1, G2 and G4: the error against the oracle is at most `_limit`."""
    oracle = _oracle(layer, x)
    y = _numpy(y)
    test.assertTrue(np.all(np.isfinite(y)))
    test.assertLessEqual(_err(y, oracle), _limit(layer, x, oracle))


def _exact_data(mode, dtype, m):
    """G3 data: a layer with small integer codes and zero points and
    power-of-two scales, and `m` inputs that keep every sum exact. `K`
    spans eight groups."""
    group = -1 if mode == "int4_per_channel" else 128
    layer = _layer((None, 512, (1, 1024)), mode.split("_")[0], group, dtype)
    view = layer._quantized_weight()
    rng = np.random.default_rng(2)
    shape = (1024, 512)
    if mode.startswith("int4"):
        codes = rng.integers(-1, 2, shape).astype("int8")
        layer._kernel.assign(pack_int4(codes, axis=-1)[0])
    else:
        codes = rng.integers(7, 10, shape).astype("uint8")
        layer.quantized_kernel.assign(pack_int4(codes, -1, "uint8")[0])
    powers = rng.integers(1, 4, layer.kernel_scale.shape)
    # A grouped scale multiplies and a per-channel scale divides.
    sign = 1 if mode == "int4_per_channel" else -1
    layer.kernel_scale.assign(2.0 ** (sign * powers))
    if view.zero_point is not None:
        low = -1 if mode == "int4" else 7
        zero = rng.integers(low, low + 3, layer.kernel_zero.shape)
        layer.kernel_zero.assign(zero.astype(codes.dtype))
    if mode == "awq":
        layer.awq_scales.assign(2.0 ** rng.integers(0, 2, 1024))
    # Eight nonzero inputs per row keep every sum exact. Seven random
    # signs have an odd sum, and the eighth sign repeats its sign, so
    # each row sums to at least 2 in magnitude. Then a per-channel
    # kernel that does not subtract `XOR` fails.
    x = np.zeros((m, 1024), "float32")
    for row in x:
        signs = rng.choice([-1, 1], 7)
        signs = np.append(signs, np.sign(signs.sum()))
        row[rng.choice(1024, 8, replace=False)] = signs
    return layer, x


def _block_errors(got, want, s, config, tolerance):
    """The error of `got` in each block `(m, n, s)` of the `GPU_CONFIGS`
    entry `config` and each plane, against `want`. An element is wrong
    when its error exceeds `tolerance`. The relative error divides by the
    largest value of `want`."""
    block_m, block_n = GPU_CONFIGS[config][:2]
    (m, n), scale = want.shape, np.max(np.abs(want)) or 1.0
    for i, j, p in itertools.product(
        range(-(-m // block_m)), range(n // 2 // block_n), (0, 1)
    ):
        rows = slice(i * block_m, (i + 1) * block_m)
        columns = slice(2 * j * block_n + p, 2 * (j + 1) * block_n, 2)
        diff = np.abs(got[rows, columns] - want[rows, columns])
        wrong = np.argwhere(~(diff <= tolerance))
        first = None
        if len(wrong):
            r = i * block_m + wrong[0][0]
            c = 2 * (j * block_n + wrong[0][1]) + p
            first = [int(r), int(c), float(got[r, c]), float(want[r, c])]
        yield {
            "m": i,
            "n": j,
            "s": s,
            "plane": ("even", "odd")[p],
            "max_abs_err": float(np.max(diff)),
            "max_rel_err": float(np.max(diff) / scale),
            "first_wrong": first,
        }


@contextlib.contextmanager
def _routes(rows=None, **sources):
    """Adds `sources` to the route table and sets the allow-list to
    `rows`.

    Only `"auto"` reads the allow-list, and a test runner can set
    `"require"` for the process. So the block runs under `"auto"`. A block
    inside this block can set a different policy. The tables and the
    policy are as before when the block ends.
    """
    allow_list = dict(kernels._ALLOW_LIST)
    kernels._ROUTES.update(sources)
    kernels._ALLOW_LIST.clear()
    kernels._ALLOW_LIST.update(rows or {})
    try:
        with kernels.policy("auto"):
            yield
    finally:
        for name in sources:
            del kernels._ROUTES[name]
        kernels._ALLOW_LIST.clear()
        kernels._ALLOW_LIST.update(allow_list)


def _row(dtype, m, route, config=None):
    """An allow-list row of the default device that runs `route` with
    `config`."""
    key = ("jax", kernels._device_class(None), dtype, kernels._m_bucket(m))
    return {key: ((route, config),)}


def _real(**changes):
    """The route `"mosaic_gpu"` with the kernel and the gate 5 of the
    interpret route, so its other values (the dtypes, the group guard and
    the version range) apply on a CPU host."""
    interpret = kernels._ROUTES["mosaic_gpu_interpret"]
    return dataclasses.replace(
        kernels._ROUTES["mosaic_gpu"],
        fn=interpret.fn,
        available=interpret.available,
        **changes,
    )


def _row_route():
    """`(host, route, sources)` for a test of the allow-list rows.

    On a GPU host, `host` is its device class, and the real route
    `"mosaic_gpu"` runs. Interpret mode fails there below compute
    capability 9.0: the pipeline copies from GMEM to SMEM with no barrier
    (`cp.async`), which interpret mode does not support. On another host,
    `host` is `None`, and `sources` add the route `"real"`: the values of
    the real route with the kernel of the interpret route. As in
    `GPUTest`, the real route needs an equal jax and jaxlib version in
    `MOSAIC_GPU_TESTED_JAX`.
    """
    if jax.devices()[0].platform == "gpu":
        return kernels._device_class(None), "mosaic_gpu", {}
    return None, "real", {"real": _real()}


def _config(m, n, block_n, split, block_m=16, stages=3):
    """A configuration with `BNH = block_n`, by default a decode one, whose
    split-K rule gives `split` for `M = m` and `N = n`."""
    blocks = -(-m // block_m) * (n // 2 // block_n)
    return (block_m, block_n, stages, blocks * split)


def _lower(capability, m, k, n, group_size, codes_dtype, config, **kwargs):
    """The text of the module that `mosaic_gpu_w4a16_matmul` lowers to for
    `compute_capability`."""
    per_channel = group_size is None
    dtype = kwargs.pop("dtype", "bfloat16")
    groups = 1 if per_channel else k // group_size
    args = (
        jax.ShapeDtypeStruct((m, k), dtype),
        jax.ShapeDtypeStruct((k, n // 2), codes_dtype),
        jax.ShapeDtypeStruct((n,) if per_channel else (groups, n), "float32"),
        None if per_channel else jax.ShapeDtypeStruct((groups, n), codes_dtype),
    )

    def call(x, codes, scale, zero):
        return quantized.mosaic_gpu_w4a16_matmul(
            x,
            codes,
            scale,
            zero,
            group_size=group_size,
            signed=codes_dtype == "int8",
            per_channel=per_channel,
            config=config,
            **kwargs,
        )

    with kernels.mosaic_gpu_lowering(capability):
        exported = jax.export.export(jax.jit(call), platforms=["cuda"])(*args)
    return exported.mlir_module()


def _rows_sum(rows, block):
    """A Mosaic GPU kernel that sums the untiled blocks `block` of `rows`
    along axis 0."""
    plgpu = quantized.plgpu

    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct(block[1:], "float32"),
        grid=(1,),
        grid_names=("i",),
    )
    def kernel(rows_g, out_g):
        spec = plgpu.BlockSpec(block, lambda s: (s, 0, 0))
        out_g[...] = plgpu.emit_pipeline(
            lambda _, rows_s, acc: acc + rows_s[0],
            grid=(rows.shape[0],),
            in_specs=[spec],
            max_concurrent_steps=2,
            init_carry=jnp.zeros(block[1:], "float32"),
        )(rows_g)

    return kernel(rows)


class ConfigTest(testing.TestCase):
    @parameterized.named_parameters(
        ("one_split", 8, 8, 1, 1),
        ("two_splits", 8, 8, 16, 2),
        ("four_splits", 8, 8, 32, 4),
        ("eight_splits", 8, 8, 64, 8),
        ("largest_split", 8, 8, 256, 8),
        ("blocks_enough", 8, 64, 256, 4),
        ("six_steps", 6, 8, 256, 2),
        ("three_steps", 3, 8, 256, 1),
    )
    def test_split_k(self, steps, blocks, min_blocks, expected):
        self.assertEqual(
            quantized._split_k(steps, blocks, min_blocks), expected
        )

    def test_configuration_rows(self):
        """Each row has blocks that the MMA tiles, and fits in the SMEM of
        sm_86 and sm_89 (99 KB per block)."""
        self.assertLen(quantized.MOSAIC_GPU_CONFIGS, 32)
        for row in quantized.MOSAIC_GPU_CONFIGS.values():
            block_m, block_n, stages, min_blocks = row
            self.assertIn(block_m, (16, 32, 64))
            self.assertIn(block_n, (32, 64, 128))
            # The accumulator tile of the warps along N.
            self.assertEqual(block_n % (8 * (4 // (block_m // 16))), 0)
            smem = stages * (2 * block_m * 128 + 128 * block_n + 16 * block_n)
            self.assertLessEqual(smem + 4 * 128 * block_n, 99 * 1024)
            self.assertGreaterEqual(min_blocks, 1)

    def test_tuned_rows(self):
        """The rows of the sweeps on an RTX 4090 (8.9) and an H100 SXM
        (9.0), for bfloat16 and float16. The other keys keep the start
        values of their SM count."""
        tuned = {
            ("8.9", "1-16"): (16, 32, 2, 128),
            ("8.9", "65-256"): (32, 32, 2, 1024),
            ("8.9", "257+"): (32, 32, 2, 1),
            ("9.0", "1-16"): (16, 64, 4, 132),
        }
        for key, row in quantized.MOSAIC_GPU_CONFIGS.items():
            capability, _, bucket = key
            sms = quantized._SMS[capability]
            start = {
                "1-16": (16, 32, 3, 2 * sms),
                "17-64": (32, 64, 2, 2 * sms),
                "65-256": (64, 64, 2, 1),
                "257+": (64, 64, 2, 1),
            }
            expected = tuned.get((capability, bucket), start[bucket])
            self.assertEqual(row, expected, key)

    def test_zero_point_of_another_dtype(self):
        """The kernel subtracts 2**23 + z + XOR in float32, which is exact
        only for an integer zero point. A float32 zero point raises."""
        args = (
            jax.ShapeDtypeStruct((1, 256), "bfloat16"),
            jax.ShapeDtypeStruct((256, 64), "int8"),
            jax.ShapeDtypeStruct((2, 128), "float32"),
            jax.ShapeDtypeStruct((2, 128), "float32"),
        )

        def call(x, codes, scale, zero):
            return quantized.mosaic_gpu_w4a16_matmul(
                x,
                codes,
                scale,
                zero,
                group_size=128,
                signed=True,
                per_channel=False,
                config=(16, 32, 2, 1),
            )

        with self.assertRaisesRegex(ValueError, "int8 or uint8 zero point"):
            jax.eval_shape(call, *args)


class LoweringTest(testing.TestCase):
    """The lowering for compute capability 8.0 to 9.0, with no untiled
    GMEM copy of a block that is not contiguous."""

    def setUp(self):
        super().setUp()
        try:
            with kernels.mosaic_gpu_lowering("8.9"):
                pass
        except NotImplementedError as e:
            self.skipTest(str(e))

    @parameterized.named_parameters(
        named_product(capability=CAPABILITIES, dtype=["bfloat16", "float16"])
    )
    def test_configuration_rows(self, capability, dtype):
        """Each row of the capability, for int8 and uint8 codes and per
        channel. `N / 2` spans several blocks, and `K` eight groups."""
        for bucket, m in BUCKET_ROWS.items():
            config = quantized.MOSAIC_GPU_CONFIGS[(capability, dtype, bucket)]
            for group_size, codes in (
                (128, "int8"),
                (128, "uint8"),
                (None, "int8"),
            ):
                text = _lower(
                    capability,
                    m,
                    1024,
                    512,
                    group_size,
                    codes,
                    config,
                    dtype=dtype,
                )
                self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(
        named_product(capability=CAPABILITIES, split=[1, 2, 4, 8])
    )
    def test_split_k(self, capability, split):
        config = _config(1, 512, 32, split)
        text = _lower(capability, 1, 1024, 512, 256, "int8", config)
        self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(
        named_product(capability=CAPABILITIES, bucket=["1-16", "17-64", "257+"])
    )
    def test_keras_hub_shapes(self, capability, bucket):
        """Each projection shape of the presets, in AWQ's form (uint8
        codes and input scales)."""
        config = quantized.MOSAIC_GPU_CONFIGS[(capability, "bfloat16", bucket)]
        shapes = {(k, n) for rows in PRESETS.values() for _, _, k, n in rows}
        for k, n in sorted(shapes):
            text = _lower(
                capability,
                BUCKET_ROWS[bucket],
                k,
                n,
                128,
                "uint8",
                config,
                input_scales=jnp.ones((k,), "bfloat16"),
            )
            self.assertIn("mosaic_gpu_v2", text)

    def test_layer_inputs_of_another_dtype(self):
        """Through a layer with `autocast=False` and a row of the real
        route, bfloat16 inputs lower to a Mosaic GPU call. float32 inputs,
        which `plgpu.mma` cannot take, lower to the reference path."""
        layer = _layer(CASES["dense"], "int4", 128, autocast=False)
        # The real route, with the device gate of the interpret route.
        real = dataclasses.replace(
            kernels._ROUTES["mosaic_gpu"],
            available=kernels._ROUTES["mosaic_gpu_interpret"].available,
            version=None,
        )
        config = quantized.MOSAIC_GPU_CONFIGS[("8.9", "bfloat16", "1-16")]
        call = jax.jit(lambda v: layer(v))
        for inputs, lowered in (("bfloat16", True), ("float32", False)):
            args = jax.ShapeDtypeStruct(CASES["dense"][2], inputs)
            with _routes(_row("bfloat16", 3, "real", config), real=real):
                with kernels.mosaic_gpu_lowering("8.9"):
                    exported = jax.export.export(call, platforms=["cuda"])(args)
            text = exported.mlir_module()
            self.assertEqual("mosaic_gpu_v2" in text, lowered)

    @parameterized.named_parameters(named_product(capability=CAPABILITIES))
    def test_guard_only_where_g_idx_is_read(self, capability):
        """Through a layer and the real route: int4 and AWQ in groups of
        128 and GPTQ with one group fix their groups, so the module holds
        the Mosaic GPU call and no guard (`stablehlo.case`) or reference
        contraction. GPTQ in groups of 128 holds the guard, whose
        reference branch contracts the dequantized weight."""
        real = dataclasses.replace(
            kernels._ROUTES["mosaic_gpu"],
            available=kernels._ROUTES["mosaic_gpu_interpret"].available,
            version=None,
        )
        key = (capability, "bfloat16", "1-16")
        row = _row("bfloat16", 3, "real", quantized.MOSAIC_GPU_CONFIGS[key])
        args = jax.ShapeDtypeStruct(CASES["dense"][2], "float32")
        for mode, group, guarded in (
            ("int4", 128, False),
            ("awq", 128, False),
            ("gptq", -1, False),
            ("gptq", 128, True),
        ):
            layer = _layer(CASES["dense"], mode, group)
            with _routes(row, real=real):
                # A new function each time: a traced one keeps its route.
                call = jax.jit(lambda v: layer(v))
                with kernels.mosaic_gpu_lowering(capability):
                    exported = jax.export.export(call, platforms=["cuda"])(args)
            text = exported.mlir_module()
            self.assertIn("mosaic_gpu_v2", text, mode)
            self.assertEqual(text.count("stablehlo.case"), guarded, mode)
            self.assertEqual("stablehlo.dot_general" in text, guarded, mode)

    @parameterized.named_parameters(named_product(capability=["8.9", "9.0"]))
    def test_copy_check_refuses_a_block_that_is_not_contiguous(
        self, capability
    ):
        """The block `(1, 4, 32)` of `(8, 4, 256)` lowers and passes
        interpret mode, but an untiled `cp.async` copy reads its rows 1 to
        3 from the next columns of row 0."""
        for shape, contiguous in (((8, 4, 32), True), ((8, 4, 256), False)):
            call = jax.jit(lambda rows: _rows_sum(rows, (1, 4, 32)))
            args = (jax.ShapeDtypeStruct(shape, "float32"),)
            with kernels.mosaic_gpu_lowering(capability):
                if contiguous:
                    jax.export.export(call, platforms=["cuda"])(*args)
                    continue
                with self.assertRaisesRegex(ValueError, "not one contiguous"):
                    jax.export.export(call, platforms=["cuda"])(*args)

    def test_each_capability_traces_again(self):
        """The pipeline reads the capability at trace time (`cp.async`
        below 9.0, TMA from 9.0), so each lowering of one signature
        traces the kernel again."""
        config = _config(1, 512, 32, 1)
        for capability in ("8.9", "9.0", "8.9"):
            text = _lower(capability, 1, 1024, 512, 128, "int8", config)
            self.assertIn("mosaic_gpu_v2", text)

    def test_lowering_restores_the_private_names(self):
        mosaic = jax.experimental.mosaic.gpu
        context = mosaic.launch_context.LaunchContext
        names = (mosaic.utils._infer_arch, context.async_copy)
        with kernels.mosaic_gpu_lowering("8.0"):
            self.assertEqual(mosaic.utils._infer_arch(), (8, 0))
            self.assertIsNot(context.async_copy, names[1])
        self.assertEqual((mosaic.utils._infer_arch, context.async_copy), names)


class InterpretTest(testing.TestCase):
    """The kernel in GPU interpret mode, through real layers."""

    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": "int4_g128", "mode": "int4", "group": 128},
                {"testcase_name": "int4_g256", "mode": "int4", "group": 256},
                {
                    "testcase_name": "int4_per_channel",
                    "mode": "int4",
                    "group": -1,
                },
                {"testcase_name": "gptq_g128", "mode": "gptq", "group": 128},
                {"testcase_name": "gptq_g256", "mode": "gptq", "group": 256},
                {"testcase_name": "awq_g128", "mode": "awq", "group": 128},
            ],
            dtype=["mixed_bfloat16", "mixed_float16"],
        )
    )
    def test_against_the_oracle(self, mode, group, dtype):
        """G1, G2, G4 and G5 at `M` 1, 3 and 17, split-K 1, 2 and 4, and
        `BNH` 32 and 64, each value twice. `N / 2` spans several blocks,
        and `K` several groups."""
        layer = _layer((None, 512, (1, 1024)), mode, group, dtype)
        for m, block_n, split in (
            (1, 32, 4),
            (1, 64, 1),
            (3, 32, 1),
            (3, 64, 2),
            (17, 32, 2),
            (17, 64, 4),
        ):
            x = _random((m, 1024), seed=m)
            config = _config(m, 512, block_n, split)
            row = _row(layer.compute_dtype, m, "mosaic_gpu_interpret", config)
            kernels.route_counts.clear()
            with _routes(row):
                y, again = layer(x), layer(x)
            self.assertEqual(
                dict(kernels.route_counts), {"mosaic_gpu_interpret": 2}
            )
            self.assertAllEqual(_numpy(again), _numpy(y))
            _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        ("int4", "int4", 128),
        ("int4_per_channel", "int4", -1),
        ("awq", "awq", 128),
    )
    def test_configuration_rows(self, mode, group):
        """`force_route` takes the rows of compute capability 8.9: `BM` 16
        and 32. The layer output equals a direct call of the route (G8)."""
        layer = _layer((None, 512, (1, 1024)), mode, group)
        for m in BUCKET_ROWS.values():
            x = _random((m, 1024), seed=m)
            with kernels.force_route("mosaic_gpu_interpret"):
                self.assertEqual(_counts(layer, x), {"mosaic_gpu_interpret": 1})
                y, direct = layer(x), _direct(layer, x)
            self.assertAllEqual(_numpy(direct), _numpy(y))
            _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        ("gptq_one_group", "gptq", -1, 1024),
        ("awq_one_group", "awq", -1, 1024),
        ("int4_partial_group", "int4", 256, 1152),
    )
    def test_group_shapes(self, mode, group, k):
        """One group of `K` rows (`group_size=-1`), and `G = 256` with
        `K = 1152`, whose last group has 128 rows. The mode fixes the
        groups, so the view reads no `g_idx` and the route runs with no
        guard. The layer output equals a direct call of the route (G8),
        and G1, G2 and G4 pass."""
        layer = _layer((None, 512, (1, k)), mode, group)
        view = layer._quantized_weight()
        self.assertIsNone(view.g_idx)
        group = min(view.scheme.group_size, k)
        g_idx = ops.convert_to_numpy(layer.g_idx)
        self.assertAllEqual(g_idx, np.arange(k) // group)
        for m in (3, 17):
            x = _random((m, k), seed=m)
            with kernels.force_route("mosaic_gpu_interpret"):
                self.assertEqual(_counts(layer, x), {"mosaic_gpu_interpret": 1})
                y, direct = layer(x), _direct(layer, x)
            self.assertAllEqual(_numpy(direct), _numpy(y))
            _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        named_product(
            mode=["int4", "int4_per_channel", "gptq", "awq"],
            dtype=["mixed_bfloat16", "mixed_float16"],
        )
    )
    def test_exact_data(self, mode, dtype):
        """G3: small integers and power-of-two scales give the oracle bit
        for bit. `N / 2` spans eight blocks and `K` eight groups, so a
        wrong nibble, plane, group or block shows."""
        layer, x = _exact_data(mode, dtype, 3)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        config = _config(3, 512, 32, 2)
        row = _row(layer.compute_dtype, 3, "mosaic_gpu_interpret", config)
        with _routes(row):
            self.assertEqual(_counts(layer, x), {"mosaic_gpu_interpret": 1})
            self.assertAllEqual(_numpy(layer(x)), oracle)

    @parameterized.named_parameters(
        named_product(
            case=sorted(CASES),
            mode=["int4", "int4_per_channel", "gptq", "awq"],
        )
    )
    def test_layers(self, case, mode):
        """G8: the output of `Dense` and of a reshape-compatible
        `EinsumDense` equals a direct call of the route, eager and under
        `jax.jit`. G7: the input gradient equals that of the wrapped
        reference path bit for bit."""
        group = -1 if mode == "int4_per_channel" else 128
        layer = _layer(CASES[case], mode.split("_")[0], group)
        x = _random(CASES[case][2])
        with kernels.force_route("reference_wrapped"):
            expected = input_gradient(layer, x)
        with kernels.force_route("mosaic_gpu_interpret"):
            self.assertEqual(_counts(layer, x), {"mosaic_gpu_interpret": 1})
            y = _numpy(layer(x))
            self.assertAllEqual(_numpy(_direct(layer, x)), y)
            self.assertAllEqual(_numpy(jax.jit(lambda v: layer(v))(x)), y)
            self.assertAllEqual(input_gradient(layer, x), expected)
        _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        ("int4", "int4", 128),
        ("int4_per_channel", "int4", -1),
        ("gptq", "gptq", 128),
        ("awq", "awq", 128),
    )
    def test_eager_call_compiles_once(self, mode, group):
        """A second eager call with the same shapes, dtypes and static
        arguments compiles nothing. Its output equals that of the function
        body run op by op bit for bit."""
        layer = _layer((None, 512, (1, 1024)), mode, group)
        view = layer._quantized_weight()
        with AutocastScope(layer.compute_dtype):
            view = view.with_tensors(view.read_tensors())
        grouped = view.scheme.group_size is not None
        args = (
            ops.cast(_random((3, 1024)), layer.compute_dtype),
            view.codes,
            view.scale,
            view.zero_point,
        )
        options = {
            "group_size": view.scheme.group_size if grouped else None,
            "signed": mode == "int4",
            "per_channel": not grouped,
            "config": _config(3, 512, 32, 2),
            "input_scales": view.input_scales,
            "interpret": True,
        }
        compiles = []

        def count(event, duration_secs, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles.append(duration_secs)

        first = quantized.mosaic_gpu_w4a16_matmul(*args, **options)
        jax.monitoring.register_event_duration_secs_listener(count)
        try:
            second = quantized.mosaic_gpu_w4a16_matmul(*args, **options)
        finally:
            jax.monitoring.unregister_event_duration_listener(count)
        self.assertEqual(compiles, [])
        self.assertAllEqual(_numpy(second), _numpy(first))
        body = quantized.mosaic_gpu_w4a16_matmul.__wrapped__(*args, **options)
        self.assertAllEqual(_numpy(first), _numpy(body))


class GPUTest(testing.TestCase):
    """The kernel on the default device: the real route `"mosaic_gpu"` on
    a CUDA GPU that passes the device gate, else the interpret route. An
    allow-list row gives the configuration, so on a GPU the real route
    needs an equal jax and jaxlib version in `MOSAIC_GPU_TESTED_JAX`. With
    `KSTAGE_OUT` set, a case that fails writes its error maps there."""

    def setUp(self):
        super().setUp()
        capability = getattr(jax.devices()[0], "compute_capability", None)
        gpu = capability in kernels._MOSAIC_GPU_CAPABILITIES
        self.route = "mosaic_gpu" if gpu else "mosaic_gpu_interpret"

    def assertMatches(self, layer, x, oracle, config, limit):
        """Through a row of the route with the configuration `config`, the
        error of `layer(x)` against `oracle` is at most `limit`."""
        m, n = oracle.shape
        block_m, block_n, stages, split = GPU_CONFIGS[config]
        row = _config(m, n, block_n, split, block_m, stages)
        kernels.route_counts.clear()
        with _routes(_row(layer.compute_dtype, m, self.route, row)):
            y = _numpy(layer(x))
            self.assertEqual(dict(kernels.route_counts), {self.route: 1})
            err = _err(y, oracle)
            if not err <= limit and os.environ.get("KSTAGE_OUT"):
                self._write_error_maps(layer, x, y, oracle, config, limit)
        self.assertLessEqual(err, limit)

    def _write_error_maps(self, layer, x, y, oracle, config, limit):
        """Writes `mosaic_error_map_<case>.json` to `$KSTAGE_OUT`, with the
        worst blocks `(m, n, s)` of each plane. The reference of split `s`
        is the oracle on the inputs of its K steps and zeros elsewhere:
        the layer output on those inputs is the partial sum of `s`."""
        block_m, block_n, stages, split = GPU_CONFIGS[config]
        (m, k), n = x.shape, oracle.shape[1]
        tolerance = limit * np.max(np.abs(oracle))
        rows_k = k // split
        blocks = {"even": [], "odd": []}
        for s in range(split):
            got, want = y, oracle
            if split > 1:
                part = np.zeros_like(x)
                steps = slice(s * rows_k, (s + 1) * rows_k)
                part[:, steps] = x[:, steps]
                got, want = _numpy(layer(part)), _oracle(layer, part)
            for block in _block_errors(got, want, s, config, tolerance):
                blocks[block["plane"]].append(block)

        def worst(block):
            return np.nan_to_num(block["max_abs_err"], nan=np.inf)

        case = self._testMethodName.removeprefix("test_")
        error_map = {
            "case": case,
            "dtype": layer.compute_dtype,
            "jax": jax.__version__,
            "shape": {
                "M": m,
                "K": k,
                "N": n,
                "G": layer._quantized_weight().scheme.group_size,
            },
            "config": {
                "BM": block_m,
                "BNH": block_n,
                "BK": quantized._BK,
                "STAGES": stages,
                "SPLIT_K": split,
            },
            "wrong_blocks": {
                plane: sum(block["first_wrong"] is not None for block in rows)
                for plane, rows in blocks.items()
            },
            "blocks": [
                block
                for rows in blocks.values()
                for block in sorted(rows, key=worst, reverse=True)[:MAP_BLOCKS]
            ],
        }
        name = f"mosaic_error_map_{case}.json"
        with open(os.path.join(os.environ["KSTAGE_OUT"], name), "w") as f:
            json.dump(error_map, f, indent=1)

    @parameterized.named_parameters(
        named_product(
            config=sorted(GPU_CONFIGS),
            mode=["int4", "int4_per_channel", "gptq"],
            dtype=["bfloat16", "float16"],
        )
    )
    def test_exact_data(self, config, mode, dtype):
        """G3 with `N / 2` over several blocks and `K` over eight groups:
        `M = 3` with the decode configuration, and `M = 80`, two blocks of
        rows, with the prefill one."""
        layer, x = _exact_data(
            mode, f"mixed_{dtype}", 3 if config == "decode" else 80
        )
        self.assertMatches(layer, x, _oracle(layer, x), config, 0.0)

    @parameterized.named_parameters(
        named_product(
            config=sorted(GPU_CONFIGS),
            mode=["int4", "int4_per_channel", "gptq"],
            dtype=["bfloat16", "float16"],
        )
    )
    def test_against_the_oracle(self, config, mode, dtype):
        """G1 and G2 on random data with `K = N = 1024`: `M = 1` with the
        decode configuration, and `M = 80` with the prefill one."""
        group = -1 if mode == "int4_per_channel" else 128
        case = (None, 1024, (1, 1024))
        layer = _layer(case, mode.split("_")[0], group, f"mixed_{dtype}")
        x = _random((1 if config == "decode" else 80, 1024))
        oracle = _oracle(layer, x)
        self.assertMatches(layer, x, oracle, config, _limit(layer, x, oracle))


class GateTest(testing.TestCase):
    def assertReference(self, layer, x, route="mosaic_gpu_interpret"):
        """`force_route(route)` runs the reference path."""
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route(route):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters(
        ("group_32", (None, 128, (2, 256)), 32, "mixed_bfloat16"),
        ("group_64", (None, 128, (2, 256)), 64, "mixed_bfloat16"),
        ("columns_96", (None, 96, (2, 256)), 128, "mixed_bfloat16"),
        ("odd_columns", (None, 127, (2, 256)), 128, "mixed_bfloat16"),
        ("rows_192", (None, 128, (2, 192)), -1, "mixed_bfloat16"),
        ("float32", (None, 128, (2, 256)), 128, "float32"),
        ("no_rows", (None, 128, (0, 256)), 128, "mixed_bfloat16"),
    )
    def test_refusals(self, case, group, dtype):
        """`G` 32 and 64, unaligned `N` and `K`, float32 compute and
        `M = 0` run the reference path, on the interpret route and with
        the values of the real route."""
        layer = _layer(case, "int4", group, dtype)
        with _routes(real=_real()):
            for route in ("mosaic_gpu_interpret", "real"):
                self.assertReference(layer, _random(case[2]), route)

    @parameterized.named_parameters(("grouped", 128), ("per_channel", -1))
    def test_inputs_of_another_dtype(self, group):
        """A layer with `autocast=False` leaves its inputs uncast. The
        kernel takes inputs in the compute dtype only, so float32 and
        float16 inputs run the reference path. bfloat16 inputs run the
        kernel, with the output dtype of the reference path."""
        layer = _layer(CASES["dense"], "int4", group, autocast=False)
        x = _random(CASES["dense"][2])
        for inputs in ("float32", "float16"):
            self.assertReference(layer, x.astype(inputs))
        x = ops.cast(x, "bfloat16")
        with kernels.policy("off"):
            expected = layer(x)
        with kernels.force_route("mosaic_gpu_interpret"):
            self.assertEqual(_counts(layer, x), {"mosaic_gpu_interpret": 1})
            self.assertDType(layer(x), expected.dtype)

    def test_activation_order_takes_the_reference_branch(self):
        """GPTQ's activation order scatters the groups, so the guard of
        `run_route` runs the reference path, eager and under `jax.jit`, on
        the interpret route and with the values of the real route."""
        layer = _layer(CASES["dense"], "gptq_act_order", 128)
        x = _random(CASES["dense"][2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with _routes(real=_real()):
            for route in ("mosaic_gpu_interpret", "real"):
                with kernels.force_route(route):
                    self.assertEqual(_counts(layer, x), {route: 1})
                    self.assertAllEqual(_numpy(layer(x)), expected)
                    y = jax.jit(lambda v: layer(v))(x)
                self.assertAllEqual(_numpy(y), expected)

    @parameterized.named_parameters(
        ("tested", ("0.12.0", "0.12.0"), "gated"),
        ("floor", ("0.11.2", "0.11.2"), "gated"),
        ("unequal_pair", ("0.12.0", "0.11.2"), "reference"),
        ("newer", ("0.12.1", "0.12.1"), "reference"),
        ("older", ("0.11.1", "0.11.1"), "reference"),
    )
    def test_version_gate(self, versions, expected):
        """Under `"auto"`, a row runs the kernel only on an equal (jax,
        jaxlib) pair in `MOSAIC_GPU_TESTED_JAX`."""
        gated = _real(version=lambda: versions)
        layer = _layer(CASES["dense"], "int4", 128)
        row = _row("bfloat16", 3, "gated")
        with _routes(row, gated=gated):
            self.assertEqual(_counts(layer, _random((3, 256))), {expected: 1})

    def test_rows_run_with_their_configuration(self):
        """Each Mosaic GPU row of the allow-list leaves the configuration
        to `MOSAIC_GPU_CONFIGS`, which has a row for the compute
        capability, the compute dtype and the M bucket of the row. With
        that configuration, a call in the bucket of the row runs the
        kernel and passes G1, G2 and G4: on a CPU host, through the values
        of the real route in interpret mode, and on a GPU host, through
        the real route, for the rows of its device class only."""
        rows = [
            (key, routes)
            for key, routes in kernels._ALLOW_LIST.items()
            if any(name == "mosaic_gpu" for name, _ in routes)
        ]
        self.assertNotEqual(rows, [])
        host, route, sources = _row_route()
        for (_, device_class, dtype, bucket), routes in rows:
            with self.subTest(device_class=device_class, dtype=dtype):
                self.assertIn(("mosaic_gpu", None), routes)
                number = device_class.removeprefix("cuda-sm")
                capability = f"{number[:-1]}.{number[-1]}"
                key = (capability, dtype, bucket)
                config = quantized.MOSAIC_GPU_CONFIGS[key]
                if host not in (None, device_class):
                    self.skipTest(f"A row of {device_class} on {host}.")
                m = BUCKET_ROWS[bucket]
                layer = _layer(CASES["dense"], "int4", 128, f"mixed_{dtype}")
                x = _random((m, 256))
                with _routes(_row(dtype, m, route, config), **sources):
                    self.assertEqual(_counts(layer, x), {route: 1})
                    y = layer(x)
                _check_accuracy(self, layer, x, y)

    def test_versions(self):
        source = kernels._ROUTES["mosaic_gpu"]
        self.assertEqual(kernels.MOSAIC_GPU_TESTED_JAX, ("0.11.2", "0.12.0"))
        self.assertEqual(source.tested_versions, kernels.MOSAIC_GPU_TESTED_JAX)
        self.assertEqual(
            source.version(), (jax.__version__, jaxlib.__version__)
        )

    def test_device_values(self):
        """The device gate admits the capabilities of the configuration
        table, and `"require"` runs the route on a JAX GPU."""
        capabilities = {key[0] for key in quantized.MOSAIC_GPU_CONFIGS}
        self.assertEqual(kernels._MOSAIC_GPU_CAPABILITIES, CAPABILITIES)
        self.assertEqual(capabilities, set(CAPABILITIES))
        self.assertEqual(kernels._REQUIRED["jax"]["gpu"], "mosaic_gpu")

    @pytest.mark.skipif(
        testing.jax_uses_gpu(),
        reason="The default JAX device is a GPU, which can pass the gate.",
    )
    def test_device_gate(self):
        """A CPU fails the device gate: an allow-list row runs the
        reference path, and `"require"` raises."""
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random((3, 256))
        with _routes(_row("bfloat16", 3, "mosaic_gpu")):
            self.assertEqual(_counts(layer, x), {"reference": 1})
        required = dict(kernels._REQUIRED)
        kernels._REQUIRED["jax"] = {"cpu": "mosaic_gpu"}
        try:
            with kernels.policy("require"):
                with self.assertRaisesRegex(ValueError, "compute capability"):
                    layer(x)
        finally:
            kernels._REQUIRED.clear()
            kernels._REQUIRED.update(required)

    def test_keras_hub_coverage(self):
        """The alignment gate admits every projection of the presets with
        `G` 128 or 256 and per channel, and none with `G` 32 or 64. The
        report gives the share of projection weights per preset."""
        groups = {"G32": 32, "G64": 64, "G128": 128, "G256": 256}
        groups["per_channel"] = None
        report = [" ".join(["preset".ljust(12), *groups])]
        for preset, rows in PRESETS.items():
            total = sum(count * k * n for _, count, k, n in rows)
            shares = []
            for name, group in groups.items():
                admitted = sum(
                    count * k * n
                    for _, count, k, n in rows
                    if kernels._mosaic_gpu_aligned(k, n, group)
                )
                shares.append(admitted / total)
                expected = 0.0 if name in ("G32", "G64") else 1.0
                self.assertEqual(admitted / total, expected)
            report.append(
                " ".join([preset.ljust(12), *(f"{s:.0%}" for s in shares)])
            )
        print("\n".join(report))


class ExportTest(testing.TestCase):
    def test_export_traces_the_reference_path(self):
        """The export entry points trace under the policy `"off"`, so the
        exported module holds no Mosaic GPU call, and the reloaded model
        gives the reference path's output.

        The batch size is static: a dynamic one adds shape assertions,
        whose custom call TensorFlow 2.21 cannot read from JAX 0.11.2
        and newer.
        """
        inputs = layers.Input((256,))
        outputs = layers.Dense(128, dtype="mixed_bfloat16")(inputs)
        model = models.Model(inputs, outputs)
        model.quantize("int4")
        x = _random((2, 256))
        path = os.path.join(self.get_temp_dir(), "model")
        kwargs = {}
        if testing.jax_uses_gpu():
            # A CPU lowering too, for a TensorFlow without a GPU.
            kwargs["jax2tf_kwargs"] = {
                "native_serialization_platforms": ("cpu", "cuda")
            }
        with kernels.force_route("mosaic_gpu_interpret"):
            kernels.route_counts.clear()
            model.export(
                path,
                format="tf_saved_model",
                verbose=False,
                input_signature=[tf.TensorSpec((2, 256), "float32")],
                **kwargs,
            )
            counts = dict(kernels.route_counts)
            self.assertEqual(_counts(model, x), {"mosaic_gpu_interpret": 1})
        self.assertNotIn("mosaic_gpu_interpret", counts)
        self.assertGreater(counts["reference"], 0)
        with open(os.path.join(path, "saved_model.pb"), "rb") as f:
            self.assertNotIn(b"mosaic_gpu", f.read())
        with kernels.policy("off"):
            expected = _numpy(model(x))
        reloaded = tf.saved_model.load(path)
        self.assertAllClose(_numpy(reloaded.serve(x)), expected)
