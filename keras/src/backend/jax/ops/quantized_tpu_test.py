"""Pallas TPU K1 (`pallas_tpu_w4a16_matmul`).

Most tests need no TPU. `kernels.pallas_tpu_lowering` lowers the kernel
for a TPU v5e and a TPU v6e on a CPU host. On a host without a TPU, the
route `"pallas_interpret"` runs the kernel in TPU interpret mode through
real layers, and on a TPU host the same tests run the real route
`"pallas_tpu"`. The weights come from real quantized layers: int4 (int8
storage), and GPTQ and AWQ (uint8 storage).
"""

import contextlib
import dataclasses
import os

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized
from jax import lax

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.jax.ops import quantized
from keras.src.distribution import distribution_lib
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
    backend.backend() != "jax", reason="Pallas TPU K1 is a JAX kernel."
)

ON_TPU = testing.jax_uses_tpu()
# The route of the layer tests: the real kernel on a TPU, else the kernel
# in TPU interpret mode.
ROUTE = "pallas_tpu" if ON_TPU else "pallas_interpret"
DEVICE_KINDS = ("TPU v5 lite", "TPU v6 lite")
# name -> (K, N, BNH) at keras-hub shapes. Gemma3-1B has `N / 2 = 576`,
# which no block of 128 columns divides, so its block is the full extent.
HUB_SHAPES = {
    "gemma2_2b": (2304, 2304, 128),
    "gemma3_1b_full_extent": (1024, 1152, 576),
    "gemma3_1b_key_value": (1152, 256, 128),
    "gemma3_1b_gate_up": (1152, 6912, 128),
    "llama2_7b_gate_up": (4096, 11008, 128),
    "llama3_8b_gate_up": (4096, 14336, 512),
}
# The global precision values of the precision rule, and their in-kernel
# dot for float32 compute.
PRECISIONS = {
    "none": (None, "DEFAULT"),
    "default": ("default", "DEFAULT"),
    "bfloat16": ("bfloat16", "DEFAULT"),
    "high": ("high", "HIGHEST"),
    "tensorfloat32": ("tensorfloat32", "HIGHEST"),
    "highest": ("highest", "HIGHEST"),
    "float32": ("float32", "HIGHEST"),
    "preset": ("BF16_BF16_F32", "HIGHEST"),
}
# name -> (mode, group size, K, N). For int4, `N / 2 = 384` spans three
# blocks of 128 columns. With `K = 1024`, `BK = 512` holds four groups of
# 128 rows over two K steps. With `K = 768` and `G = 256`, `BK = 256` holds
# one group per step over three steps. In groups of 32 and 64, `BK = 512`
# holds 16 and 8 groups, so the groups slice the activation block at 32
# and 64 lanes. `N = 1152` takes a full-extent block of 576 columns. GPTQ
# compiles its solver once per `N`, so each GPTQ and AWQ layer of this
# file has `N = 256`: one block of 128 columns.
ORACLE_CASES = {
    "int4_g32": ("int4", 32, 1024, 768),
    "int4_g64": ("int4", 64, 1024, 768),
    "int4_g128": ("int4", 128, 1024, 768),
    "int4_g256": ("int4", 256, 768, 768),
    "int4_per_channel": ("int4", -1, 1024, 768),
    "int4_full_extent": ("int4", 128, 256, 1152),
    "gptq_g128": ("gptq", 128, 1024, 256),
    "gptq_one_group": ("gptq", -1, 1024, 256),
    "awq_g128": ("awq", 128, 1024, 256),
}
# The compute dtypes of the interpret suite: dtype policy -> ((global
# precision, M values), ...). bfloat16, and float32 at the default
# precision (one bfloat16 pass) and under `"highest"`. One row takes a
# block of 16 rows, 17 rows a block of 128, and 130 rows two blocks.
REGIMES = {
    "mixed_bfloat16": ((None, (1, 17, 130)),),
    "float32": ((None, (17,)), ("highest", (1, 130))),
}
# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
CASES = {
    "dense": (None, 256, (3, 256)),
    "qkv_dnh": ("btd,dnh->btnh", (None, 4, 64), (2, 5, 256)),
    "query_ndh": ("btd,ndh->btnh", (None, 4, 64), (2, 5, 256)),
    "output_nhd": ("btnh,nhd->btd", (None, 256), (2, 5, 4, 64)),
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


def _tolerance(dtype):
    """The G1 floor and the G2 bound of the in-kernel dot of `dtype`, under
    the global precision: bfloat16, one bfloat16 pass, or float32."""
    if dtype == "bfloat16":
        return 1e-3, 2**-7
    if quantized.pallas_tpu_precision(dtype) == lax.Precision.HIGHEST:
        return 1e-6, 1e-5
    return 2**-8, 2**-7


def _check_accuracy(test, layer, x, y):
    """G1, G2 and G4: the error against the oracle is at most 1.25 times
    that of the reference path plus a floor, and at most a bound."""
    oracle = _oracle(layer, x)
    with kernels.policy("off"):
        reference = _numpy(layer(x))
    floor, bound = _tolerance(layer.compute_dtype)
    y = _numpy(y)
    test.assertTrue(np.all(np.isfinite(y)))
    err = _err(y, oracle)
    test.assertLessEqual(err, 1.25 * _err(reference, oracle) + floor)
    test.assertLessEqual(err, bound)


def _exact_data(mode, dtype, m):
    """G3 data: a layer with small integer codes and zero points and
    power-of-two scales, and `m` inputs that keep every sum exact. `K`
    spans eight groups of 128, or 32 and 16 groups for the modes
    `"int4_g32"` and `"int4_g64"`. For int4, `N / 2` spans three
    blocks."""
    group = {"int4_per_channel": -1, "int4_g32": 32, "int4_g64": 64}
    group = group.get(mode, 128)
    int4 = mode.startswith("int4")
    n = 768 if int4 else 256
    layer = _layer((None, n, (1, 1024)), mode.split("_")[0], group, dtype)
    view = layer._quantized_weight()
    rng = np.random.default_rng(2)
    shape = (1024, n)
    if int4:
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
        low = -1 if int4 else 7
        zero = rng.integers(low, low + 3, layer.kernel_zero.shape)
        layer.kernel_zero.assign(zero.astype(codes.dtype))
    if mode == "awq":
        layer.awq_scales.assign(2.0 ** rng.integers(0, 2, 1024))
    # Eight nonzero inputs of 1 or -1 per row keep every sum exact.
    x = np.zeros((m, 1024), "float32")
    for row in x:
        row[rng.choice(1024, 8, replace=False)] = rng.choice([-1, 1], 8)
    return layer, x


@contextlib.contextmanager
def _routes(rows=None, **sources):
    """Adds `sources` to the route table and sets the allow-list to
    `rows`."""
    allow_list = dict(kernels._ALLOW_LIST)
    kernels._ROUTES.update(sources)
    kernels._ALLOW_LIST.clear()
    kernels._ALLOW_LIST.update(rows or {})
    try:
        yield
    finally:
        for name in sources:
            del kernels._ROUTES[name]
        kernels._ALLOW_LIST.clear()
        kernels._ALLOW_LIST.update(allow_list)


def _row(dtype, m, route):
    """An allow-list row of the default device that runs `route`."""
    key = ("jax", kernels._device_class(None), dtype, kernels._m_bucket(m))
    return {key: ((route, None),)}


def _real(**changes):
    """The route `"pallas_tpu"`. On a CPU host it takes the kernel and the
    gate 5 of the interpret route, so that its other values (the dtypes,
    the group guard and the version range) apply there."""
    if not ON_TPU:
        interpret = kernels._ROUTES["pallas_interpret"]
        changes = {
            "fn": interpret.fn,
            "available": interpret.available,
            **changes,
        }
    return dataclasses.replace(kernels._ROUTES["pallas_tpu"], **changes)


def _lower(device_kind, m, k, n, group_size, codes_dtype, **kwargs):
    """The text of the module that `pallas_tpu_w4a16_matmul` lowers to for
    a TPU of `device_kind`."""
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
        return quantized.pallas_tpu_w4a16_matmul(
            x,
            codes,
            scale,
            zero,
            group_size=group_size,
            signed=codes_dtype == "int8",
            per_channel=per_channel,
            config=quantized.pallas_tpu_blocks(m, k, n, group_size),
            **kwargs,
        )

    with kernels.pallas_tpu_lowering(device_kind):
        exported = jax.export.export(jax.jit(call), platforms=["tpu"])(*args)
    return exported.mlir_module()


@contextlib.contextmanager
def _precision(value, setter="context"):
    """Sets the global `jax_default_matmul_precision` to `value`, with
    `jax.config.update` or the `jax.default_matmul_precision` context
    manager."""
    if setter == "context":
        with jax.default_matmul_precision(value):
            yield
        return
    previous = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", value)
    try:
        yield
    finally:
        jax.config.update("jax_default_matmul_precision", previous)


class InterpretTestCase(testing.TestCase):
    """Skips on a host without a TPU where TPU interpret mode does not
    work."""

    def setUp(self):
        super().setUp()
        if ON_TPU:
            return
        try:
            with kernels.pallas_tpu_interpret():
                pass
        except NotImplementedError as e:
            self.skipTest(str(e))


class StaticGateTest(testing.TestCase):
    """The gates that need no kernel call."""

    @parameterized.named_parameters(
        [(name, *shape) for name, shape in HUB_SHAPES.items()]
    )
    def test_keras_hub_blocks(self, k, n, block_n):
        """`BNH` at the keras-hub shapes, the full extent included, and
        `BM` 16 at decode and 128 above."""
        for m, block_m in ((1, 16), (16, 16), (17, 128), (300, 128)):
            blocks = quantized.pallas_tpu_blocks(m, k, n, 128)
            self.assertEqual((blocks[0], blocks[2]), (block_m, block_n))

    @parameterized.named_parameters(
        ("g128_k1024", 1024, 128, 512),
        ("g256_k1024", 1024, 256, 512),
        ("g128_k384", 384, 128, 128),
        ("g32_k1024", 1024, 32, 512),
        ("g256_k2304", 2304, 256, 256),
        ("one_group", 384, 384, 128),
        ("per_channel", 1152, None, 128),
        ("g256_k384", 384, 256, None),
        ("g256_k1152", 1152, 256, None),
        ("g512_k1536", 1536, 512, 512),
        ("k192", 192, None, None),
        ("k200", 200, 200, None),
    )
    def test_block_k(self, k, group_size, block_k):
        """`BK` is the largest of 512, 256 and 128 that divides `K` and
        that the group size divides, and `None` without one."""
        blocks = quantized.pallas_tpu_blocks(1, k, 256, group_size)
        self.assertEqual(blocks and blocks[1], block_k)

    def test_odd_columns(self):
        self.assertIsNone(quantized.pallas_tpu_blocks(1, 256, 127, 128))

    @parameterized.named_parameters(named_product(setter=["config", "context"]))
    def test_precision_rule(self, setter):
        """The global precision maps through `lax.Precision` to `DEFAULT`
        or `HIGHEST` for float32 compute. bfloat16 compute takes
        `DEFAULT`."""
        for value, expected in PRECISIONS.values():
            with _precision(value, setter):
                self.assertEqual(
                    quantized.pallas_tpu_precision("float32"),
                    getattr(lax.Precision, expected),
                )
                self.assertEqual(
                    quantized.pallas_tpu_precision("bfloat16"),
                    lax.Precision.DEFAULT,
                )

    def test_versions(self):
        source = kernels._ROUTES["pallas_tpu"]
        self.assertEqual(kernels.PALLAS_TPU_TESTED_JAX, ())
        self.assertEqual(source.tested_versions, kernels.PALLAS_TPU_TESTED_JAX)
        self.assertEqual(
            source.version(), (jax.__version__, jaxlib.__version__)
        )
        self.assertEqual(source.dtypes, ("bfloat16", "float32"))
        self.assertTrue(source.contiguous_groups)
        self.assertEqual(kernels._REQUIRED["jax"]["tpu"], "pallas_tpu")

    @pytest.mark.skipif(ON_TPU, reason="A TPU passes the device gate.")
    def test_device_gate(self):
        """A device that is not a TPU fails the device gate: an allow-list
        row runs the reference path, and `"require"` raises."""
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random((3, 256))
        with _routes(_row("bfloat16", 3, "pallas_tpu")):
            with kernels.policy("auto"):
                self.assertEqual(_counts(layer, x), {"reference": 1})
        required = dict(kernels._REQUIRED)
        device_type = distribution_lib.list_devices()[0].split(":")[0]
        kernels._REQUIRED["jax"] = {device_type: "pallas_tpu"}
        try:
            with kernels.policy("require"):
                with self.assertRaisesRegex(ValueError, "needs a TPU"):
                    layer(x)
        finally:
            kernels._REQUIRED.clear()
            kernels._REQUIRED.update(required)

    def test_keras_hub_coverage(self):
        """The share of the projection weights of each preset that the
        alignment gate admits. It admits every weight with `G` 32, 64 and
        128 and per channel. With `G = 256`, the `K = 1152` weights of
        Gemma3-1B have no `BK` that 256 divides."""
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
                    if kernels._pallas_tpu_aligned(1, k, n, group)
                )
                shares.append(admitted / total)
                if preset == "gemma3_1b" and name == "G256":
                    # `attention_output` (K = 1024) and `down` (K = 6912).
                    expected = (1024 * 1152 + 6912 * 1152) / total
                else:
                    expected = 1.0
                self.assertAlmostEqual(admitted / total, expected)
            report.append(
                " ".join([preset.ljust(12), *(f"{s:.0%}" for s in shares)])
            )
        print("\n".join(report))


class LoweringTest(testing.TestCase):
    """The lowering to a `tpu_custom_call` for a TPU v5e and a TPU v6e."""

    def setUp(self):
        super().setUp()
        try:
            with kernels.pallas_tpu_lowering(DEVICE_KINDS[0]):
                pass
        except NotImplementedError as e:
            self.skipTest(str(e))

    @parameterized.named_parameters(
        ("v5e", DEVICE_KINDS[0]), ("v6e", DEVICE_KINDS[1])
    )
    def test_keras_hub_shapes(self, device_kind):
        """Each keras-hub shape in three forms: int4 in groups of 128 at
        decode in bfloat16, AWQ (uint8 codes and input scales) at `M = 130`
        in float32, and per-channel int4 at decode in bfloat16."""
        for k, n, _ in HUB_SHAPES.values():
            for m, group, codes, kwargs in (
                (1, 128, "int8", {}),
                (
                    130,
                    128,
                    "uint8",
                    {
                        "dtype": "float32",
                        "input_scales": jnp.ones((k,), "float32"),
                    },
                ),
                (1, None, "int8", {}),
            ):
                text = _lower(device_kind, m, k, n, group, codes, **kwargs)
                self.assertEqual(text.count("tpu_custom_call"), 1)

    @parameterized.named_parameters(
        named_product(device_kind=DEVICE_KINDS, group=[32, 64])
    )
    def test_small_groups(self, device_kind, group):
        """The gate admits groups of 32 and 64 rows. Their groups slice
        the activation block at 32 and 64 lanes, 2 to 16 groups per K
        step. Each keras-hub shape lowers as int4 at decode in bfloat16,
        and as AWQ at `M = 130` in float32."""
        for k, n, _ in HUB_SHAPES.values():
            for m, codes, kwargs in (
                (1, "int8", {}),
                (
                    130,
                    "uint8",
                    {
                        "dtype": "float32",
                        "input_scales": jnp.ones((k,), "float32"),
                    },
                ),
            ):
                text = _lower(device_kind, m, k, n, group, codes, **kwargs)
                self.assertEqual(text.count("tpu_custom_call"), 1)

    @parameterized.named_parameters(
        named_product(
            precision=sorted(PRECISIONS),
            setter=["config", "context"],
            dtype=["bfloat16", "float32"],
        )
    )
    def test_global_precision(self, precision, setter, dtype):
        """Each in-kernel dot passes an explicit precision, so the kernel
        lowers under each global precision value. A dot without one takes
        the global value, and fails to lower under each `DotAlgorithmPreset`
        name, and on JAX 0.10.2 also under `"high"` and
        `"tensorfloat32"`."""
        value = PRECISIONS[precision][0]
        with _precision(value, setter):
            text = _lower(
                DEVICE_KINDS[0], 1, 1024, 512, 128, "int8", dtype=dtype
            )
        self.assertEqual(text.count("tpu_custom_call"), 1)

    def test_layer_lowers_the_kernel(self):
        """Through a layer and the real route, the module holds one
        `tpu_custom_call`. Under the policy `"off"` it holds none."""
        layer = _layer(CASES["dense"], "int4", 128)
        args = jax.ShapeDtypeStruct(CASES["dense"][2], "float32")
        real = dataclasses.replace(
            kernels._ROUTES["pallas_tpu"],
            available=kernels._ROUTES["pallas_interpret"].available,
            version=None,
        )
        for policy, count in (("auto", 1), ("off", 0)):
            with _routes(_row("bfloat16", 3, "real"), real=real):
                with kernels.policy(policy):
                    # A new function each time: a traced one keeps its route.
                    call = jax.jit(lambda v: layer(v))
                    with kernels.pallas_tpu_lowering(DEVICE_KINDS[1]):
                        exported = jax.export.export(call, platforms=["tpu"])(
                            args
                        )
            self.assertEqual(
                exported.mlir_module().count("tpu_custom_call"), count
            )


class InterpretTest(InterpretTestCase):
    """The kernel through real layers, in TPU interpret mode on a host
    without a TPU."""

    @pytest.mark.skipif(ON_TPU, reason="A TPU needs no interpret helper.")
    def test_helper_reads_the_default_device(self):
        """Inside `pallas_tpu_interpret`, the TPU information that a
        pipeline step reads is a TPU v5e for the device kind of the
        default device (`"cpu"`, or a GPU kind on a GPU host). On exit,
        the block restores the chip registry and clears the cache of
        `get_tpu_info`, so the TPU v5e does not stay."""
        tpu_info = getattr(jax._src, "tpu_info", None)
        if tpu_info is None:
            self.skipTest(f"JAX {jax.__version__} reads an abstract mesh.")
        registry = dict(tpu_info.registry)
        with kernels.pallas_tpu_interpret():
            info = tpu_info.get_tpu_info()
        self.assertEqual(info.chip_version, tpu_info.ChipVersion.TPU_V5E)
        self.assertEqual(tpu_info.registry, registry)
        with self.assertRaisesRegex(ValueError, "Unsupported TPU device"):
            tpu_info.get_tpu_info()

    @parameterized.named_parameters([(name, name) for name in ORACLE_CASES])
    def test_against_the_oracle(self, case):
        """G1, G2, G4 and G5 in bfloat16, and in float32 at the default
        precision and under `"highest"`, with `M` padded to the block
        rows."""
        mode, group, k, n = ORACLE_CASES[case]
        for dtype, regimes in REGIMES.items():
            layer = _layer((None, n, (1, k)), mode, group, dtype)
            for precision, rows in regimes:
                for m in rows:
                    x = _random((m, k), seed=m)
                    kernels.route_counts.clear()
                    with _precision(precision), kernels.force_route(ROUTE):
                        y, again = layer(x), layer(x)
                        counts = dict(kernels.route_counts)
                        self.assertEqual(counts, {ROUTE: 2})
                        self.assertAllEqual(_numpy(again), _numpy(y))
                        _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        named_product(
            mode=[
                "int4",
                "int4_g32",
                "int4_g64",
                "int4_per_channel",
                "gptq",
                "awq",
            ],
            dtype=["mixed_bfloat16", "float32"],
        )
    )
    def test_exact_data(self, mode, dtype):
        """G3: small integers and power-of-two scales give the oracle bit
        for bit, also with one bfloat16 pass. `K` spans eight groups of
        128, four per K step. In groups of 32 and 64, a K step holds 16
        and 8 groups. `M = 3` pads one block of rows, and `M = 130` takes
        two."""
        layer, x = _exact_data(mode, dtype, 130)
        for m in (3, 130):
            oracle = _oracle(layer, x[:m]).astype("float32")
            with kernels.policy("off"):
                self.assertAllEqual(_numpy(layer(x[:m])), oracle)
            with kernels.force_route(ROUTE):
                self.assertEqual(_counts(layer, x[:m]), {ROUTE: 1})
                self.assertAllEqual(_numpy(layer(x[:m])), oracle)

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
        with kernels.force_route(ROUTE):
            self.assertEqual(_counts(layer, x), {ROUTE: 1})
            y = _numpy(layer(x))
            self.assertAllEqual(_numpy(_direct(layer, x)), y)
            self.assertAllEqual(_numpy(jax.jit(lambda v: layer(v))(x)), y)
            self.assertAllEqual(input_gradient(layer, x), expected)
            _check_accuracy(self, layer, x, y)

    def test_activation_order_takes_the_reference_branch(self):
        """GPTQ's activation order scatters the groups, so the guard of
        `run_route` runs the reference path, eager and under `jax.jit`, on
        the route and with the values of the real route."""
        layer = _layer(CASES["dense"], "gptq_act_order", 128)
        x = _random(CASES["dense"][2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with _routes(real=_real()):
            for route in (ROUTE, "real"):
                with kernels.force_route(route):
                    self.assertEqual(_counts(layer, x), {route: 1})
                    self.assertAllEqual(_numpy(layer(x)), expected)
                    y = jax.jit(lambda v: layer(v))(x)
                self.assertAllEqual(_numpy(y), expected)

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
        layer = _layer((None, 256, (1, 1024)), mode, group)
        view = layer._quantized_weight()
        with AutocastScope(layer.compute_dtype):
            view = view.with_tensors(view.read_tensors())
        grouped = view.g_idx is not None
        group_size = view.scheme.group_size if grouped else None
        args = (
            ops.cast(_random((3, 1024)), layer.compute_dtype),
            view.codes,
            view.scale,
            view.zero_point,
        )
        options = {
            "group_size": group_size,
            "signed": mode == "int4",
            "per_channel": not grouped,
            "config": quantized.pallas_tpu_blocks(3, 1024, 256, group_size),
            "input_scales": view.input_scales,
            "interpret": None if ON_TPU else quantized.pltpu.InterpretParams(),
        }
        compiles = []

        def count(event, duration_secs, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles.append(duration_secs)

        # On a TPU, the real kernel runs.
        scope = kernels.pallas_tpu_interpret()
        with contextlib.nullcontext() if ON_TPU else scope:
            first = quantized.pallas_tpu_w4a16_matmul(*args, **options)
            jax.monitoring.register_event_duration_secs_listener(count)
            try:
                second = quantized.pallas_tpu_w4a16_matmul(*args, **options)
            finally:
                jax.monitoring.unregister_event_duration_listener(count)
            body = quantized.pallas_tpu_w4a16_matmul.__wrapped__(
                *args, **options
            )
        self.assertEqual(compiles, [])
        self.assertAllEqual(_numpy(second), _numpy(first))
        self.assertAllEqual(_numpy(first), _numpy(body))

    @pytest.mark.skipif(not ON_TPU, reason="Runs the kernel on a TPU.")
    def test_require_runs_the_kernel(self):
        """G8 on hardware: `"require"` runs Pallas TPU K1, and the lowered
        module holds its `tpu_custom_call`."""
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random(CASES["dense"][2])
        call = jax.jit(lambda v: layer(v))
        with kernels.policy("require"):
            self.assertEqual(_counts(layer, x), {"pallas_tpu": 1})
            self.assertAllEqual(_numpy(layer(x)), _numpy(_direct(layer, x)))
            text = call.lower(x).as_text()
        self.assertEqual(text.count("tpu_custom_call"), 1)


class GateTest(InterpretTestCase):
    """The gate refusals, the inputs of another dtype and the version gate,
    on the route and on the real route."""

    def assertReference(self, layer, x, route=ROUTE):
        """`force_route(route)` runs the reference path."""
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route(route):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters(
        ("odd_columns", (None, 127, (2, 256)), 128, "mixed_bfloat16"),
        ("rows_192", (None, 128, (2, 192)), -1, "mixed_bfloat16"),
        ("group_256_rows_384", (None, 128, (2, 384)), 256, "mixed_bfloat16"),
        ("partial_group", (None, 128, (2, 1152)), 256, "mixed_bfloat16"),
        ("float16", (None, 128, (2, 256)), 128, "mixed_float16"),
        ("no_rows", (None, 128, (0, 256)), 128, "mixed_bfloat16"),
    )
    def test_refusals(self, case, group, dtype):
        """Odd `N`, a `K` with no `BK`, a group size that does not divide
        `BK` (and a partial last group), float16 compute and `M = 0` run
        the reference path, on the route and with the values of the real
        route."""
        layer = _layer(case, "int4", group, dtype)
        with _routes(real=_real()):
            for route in (ROUTE, "real"):
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
        with kernels.force_route(ROUTE):
            self.assertEqual(_counts(layer, x), {ROUTE: 1})
            self.assertDType(layer(x), expected.dtype)

    @parameterized.named_parameters(
        ("tested", ("0.12.0", "0.12.0"), ("0.12.0",), "gated"),
        ("unequal_pair", ("0.12.0", "0.11.2"), ("0.12.0",), "reference"),
        ("outside", ("0.12.1", "0.12.1"), ("0.12.0",), "reference"),
        ("no_tested_version", ("0.12.0", "0.12.0"), (), "reference"),
    )
    def test_version_gate(self, versions, tested_versions, expected):
        """Under `"auto"`, a row runs the kernel only on an equal (jax,
        jaxlib) pair in the tested versions. `PALLAS_TPU_TESTED_JAX` is
        empty, so no version runs it yet."""
        gated = _real(version=lambda: versions, tested_versions=tested_versions)
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random((3, 256))
        with _routes(_row("bfloat16", 3, "gated"), gated=gated):
            with kernels.policy("auto"):
                self.assertEqual(_counts(layer, x), {expected: 1})


class ExportTest(InterpretTestCase):
    def test_export_traces_the_reference_path(self):
        """The export entry points trace under the policy `"off"`, so the
        exported module holds no `tpu_custom_call`, and the reloaded model
        gives the reference path's output."""
        inputs = layers.Input((256,))
        outputs = layers.Dense(128, dtype="mixed_bfloat16")(inputs)
        model = models.Model(inputs, outputs)
        model.quantize("int4")
        x = _random((2, 256))
        path = os.path.join(self.get_temp_dir(), "model")
        kwargs = {}
        if ON_TPU:
            # A CPU lowering too, for a TensorFlow without a TPU.
            kwargs["jax2tf_kwargs"] = {
                "native_serialization_platforms": ("cpu", "tpu")
            }
        with kernels.force_route(ROUTE):
            kernels.route_counts.clear()
            model.export(
                path,
                format="tf_saved_model",
                verbose=False,
                input_signature=[tf.TensorSpec((2, 256), "float32")],
                **kwargs,
            )
            counts = dict(kernels.route_counts)
            self.assertEqual(_counts(model, x), {ROUTE: 1})
        self.assertNotIn(ROUTE, counts)
        self.assertGreater(counts["reference"], 0)
        with open(os.path.join(path, "saved_model.pb"), "rb") as f:
            self.assertNotIn(b"tpu_custom_call", f.read())
        with kernels.policy("off"):
            expected = _numpy(model(x))
        reloaded = tf.saved_model.load(path)
        self.assertAllClose(_numpy(reloaded.serve(x)), expected)
