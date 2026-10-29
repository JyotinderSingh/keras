"""The Pallas TPU matmul kernel (`pallas_tpu_w4a16_matmul`) as an op.

Most tests need no TPU. `pallas_tpu_lowering` (`quantized_test_utils`)
lowers the kernel for a TPU v5e and a TPU v6e on a CPU host. On a host
without a TPU, `InterpretTest` runs the kernel in TPU interpret mode
through real layers, and on a TPU host the same tests run the kernel. The
route suites of `kernels_test.py` run the interpret route
`"pallas_interpret"` (from `kernels_test_utils`) through real layers. The
weights come from real quantized layers: int4 (int8 storage), and GPTQ and
AWQ (uint8 storage). On JAX 0.9, the lowering and interpret tests skip.
The labels G1 to G8 are those of `kernels_test_utils`.
"""

import contextlib

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized
from jax import lax

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.jax.ops import quantized
from keras.src.backend.jax.ops.quantized_test_utils import kernel_record
from keras.src.backend.jax.ops.quantized_test_utils import pallas_tpu_interpret
from keras.src.backend.jax.ops.quantized_test_utils import (
    pallas_tpu_interpret_available,
)
from keras.src.backend.jax.ops.quantized_test_utils import pallas_tpu_lowering
from keras.src.backend.jax.ops.quantized_test_utils import pallas_tpu_route
from keras.src.quantizers import kernels
from keras.src.quantizers.kernels_test_utils import counts
from keras.src.quantizers.kernels_test_utils import direct
from keras.src.quantizers.kernels_test_utils import route_tables
from keras.src.quantizers.kernels_test_utils import row
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.testing.test_utils import named_product

pytestmark = pytest.mark.skipif(
    backend.backend() != "jax", reason="The Pallas TPU kernel is a JAX kernel."
)

ON_TPU = testing.jax_uses_tpu()
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
# The compute dtypes of the interpret tests: dtype policy -> ((global
# precision, M values), ...). bfloat16, and float32 at the default
# precision (one bfloat16 pass) and under `"highest"`. One row takes a
# block of 16 rows, 17 rows a block of 128, and 130 rows two blocks.
REGIMES = {
    "mixed_bfloat16": ((None, (1, 17, 130)),),
    "float32": ((None, (17,)), ("highest", (1, 130))),
}
# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
CASES = {"dense": (None, 256, (3, 256))}
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


def _float_layer(case, dtype):
    """A built float layer of `case` with a random kernel."""
    equation, output, input_shape = case
    if equation is None:
        layer = layers.Dense(output, use_bias=False, dtype=dtype)
    else:
        layer = layers.EinsumDense(equation, output, dtype=dtype)
    layer.build(input_shape)
    rng = np.random.default_rng(0)
    layer._kernel.assign(rng.standard_normal(layer._kernel.shape))
    return layer


def _layer(case, mode, group_size, dtype="mixed_bfloat16"):
    """A layer of `case` in `mode`: `"int4"` (int8 storage, per-channel
    with a `group_size` of -1), or `"gptq"` or `"awq"` (uint8 storage)."""
    layer = _float_layer(case, dtype)
    if mode == "int4":
        config = Int4QuantizationConfig(block_size=group_size)
        layer.quantize("int4", config=config)
        return layer
    config = calibration_config(mode, group_size=group_size)
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
    for inputs in x:
        inputs[rng.choice(1024, 8, replace=False)] = rng.choice([-1, 1], 8)
    return layer, x


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
            **kwargs,
        )

    with pallas_tpu_lowering(device_kind):
        exported = jax.export.export(jax.jit(call), platforms=["tpu"])(*args)
    return exported.mlir_module()


def _lower_layer(layer, device_kind, policy="auto"):
    """The text of the module that a call of `layer` lowers to for a TPU of
    `device_kind`, through a measured row of the kernel route `"real"`."""
    real = pallas_tpu_route(interpret=False, tested=None)
    args = jax.ShapeDtypeStruct(CASES["dense"][2], "float32")
    with route_tables(row("bfloat16", 3, "real"), routes={"real": real}):
        with kernels.policy(policy):
            # A new function each time: a traced one keeps its route.
            call = jax.jit(lambda v: layer(v))
            with pallas_tpu_lowering(device_kind):
                exported = jax.export.export(call, platforms=["tpu"])(args)
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


class BlocksTest(testing.TestCase):
    """The blocks and the precision of a call, which need no kernel
    call."""

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


class LoweringTest(testing.TestCase):
    """The lowering to a `tpu_custom_call` for a TPU v5e and a TPU v6e."""

    def setUp(self):
        super().setUp()
        try:
            with pallas_tpu_lowering(DEVICE_KINDS[0]):
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
        """Through a layer and the kernel route, the module holds one
        `tpu_custom_call`. Under the policy `"off"` it holds none."""
        layer = _layer(CASES["dense"], "int4", 128)
        for policy, count in (("auto", 1), ("off", 0)):
            text = _lower_layer(layer, DEVICE_KINDS[1], policy)
            self.assertEqual(text.count("tpu_custom_call"), count)

    @parameterized.named_parameters(
        ("v5e", DEVICE_KINDS[0]), ("v6e", DEVICE_KINDS[1])
    )
    def test_guard_only_where_g_idx_is_read(self, device_kind):
        """Through a layer and the kernel route: int4 and AWQ in groups of
        128 and GPTQ with one group fix their groups, so the module holds
        the kernel and no guard (`stablehlo.case`) or reference
        contraction. GPTQ in groups of 128 holds the guard, whose
        reference branch contracts the dequantized weight."""
        for mode, group, guarded in (
            ("int4", 128, False),
            ("awq", 128, False),
            ("gptq", -1, False),
            ("gptq", 128, True),
        ):
            layer = _layer(CASES["dense"], mode, group)
            text = _lower_layer(layer, device_kind)
            self.assertEqual(text.count("tpu_custom_call"), 1, mode)
            self.assertEqual(text.count("stablehlo.case"), guarded, mode)
            self.assertEqual("stablehlo.dot_general" in text, guarded, mode)


class InterpretTest(testing.TestCase):
    """The kernel through real layers: in TPU interpret mode on a host
    without a TPU, and the kernel itself on a TPU host. The route
    `"kernel"` runs it with no version range."""

    def setUp(self):
        super().setUp()
        if not ON_TPU and not pallas_tpu_interpret_available():
            self.skipTest(
                f"TPU interpret mode needs a TPU that JAX {jax.__version__} "
                "can register on a host without a TPU."
            )

    @contextlib.contextmanager
    def forced(self):
        """Forces the route `"kernel"`."""
        kernel = pallas_tpu_route(interpret=not ON_TPU, tested=None)
        with route_tables(routes={"kernel": kernel}):
            with kernels.force_route("kernel"):
                yield

    @parameterized.named_parameters([(name, name) for name in ORACLE_CASES])
    def test_against_the_oracle(self, case):
        """G1, G2, G4 and G5 in bfloat16, and in float32 at the default
        precision and under `"highest"`, with `M` padded to the block
        rows. The layer output equals a direct call of the route (G8)."""
        mode, group, k, n = ORACLE_CASES[case]
        for dtype, regimes in REGIMES.items():
            layer = _layer((None, n, (1, k)), mode, group, dtype)
            for precision, rows in regimes:
                for m in rows:
                    x = _random((m, k), seed=m)
                    kernels.route_counts.clear()
                    with _precision(precision), self.forced():
                        y, again = layer(x), layer(x)
                        called = direct(layer, x)
                        self.assertEqual(
                            dict(kernels.route_counts), {"kernel": 3}
                        )
                        self.assertAllEqual(_numpy(again), _numpy(y))
                        self.assertAllEqual(_numpy(called), _numpy(y))
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
            with self.forced():
                self.assertEqual(counts(layer, x[:m]), {"kernel": 1})
                self.assertAllEqual(_numpy(layer(x[:m])), oracle)

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
        grouped = view.scheme.group_size is not None
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

        # On a TPU, the kernel itself runs.
        scope = contextlib.nullcontext() if ON_TPU else pallas_tpu_interpret()
        with scope:
            first = quantized.pallas_tpu_w4a16_matmul(*args, **options)
            jax.monitoring.register_event_duration_secs_listener(count)
            try:
                second = quantized.pallas_tpu_w4a16_matmul(*args, **options)
            finally:
                jax.monitoring.unregister_event_duration_listener(count)
            body = quantized._pallas_tpu_w4a16_matmul.__wrapped__(
                *args, **options
            )
        self.assertEqual(compiles, [])
        self.assertAllEqual(_numpy(second), _numpy(first))
        self.assertAllEqual(_numpy(first), _numpy(body))

    def test_auto_needs_a_tested_version(self):
        """Under `"auto"`, a measured row runs the kernel only with an equal
        jax and jaxlib version in `PALLAS_TPU_TESTED_JAX`. The range is
        empty, so the row runs the reference path, and `force_route`,
        which skips the range, runs the kernel."""
        gated = pallas_tpu_route(interpret=not ON_TPU)
        self.assertIs(gated.tested, quantized.pallas_tpu_tested)
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random(CASES["dense"][2])
        rows = row("bfloat16", 3, "gated")
        with route_tables(rows, routes={"gated": gated}):
            self.assertEqual(counts(layer, x), {"reference": 1})
            with kernels.force_route("gated"):
                self.assertEqual(counts(layer, x), {"gated": 1})

    @pytest.mark.skipif(not ON_TPU, reason="Runs the kernel on a TPU.")
    def test_require_runs_the_kernel(self):
        """G8 on hardware: `"require"` runs the Pallas TPU kernel, and the
        lowered module holds its `tpu_custom_call`."""
        layer = _layer(CASES["dense"], "int4", 128)
        x = _random(CASES["dense"][2])
        call = jax.jit(lambda v: layer(v))
        with kernels.policy("require"):
            self.assertEqual(counts(layer, x), {"pallas_tpu": 1})
            self.assertAllEqual(_numpy(layer(x)), _numpy(direct(layer, x)))
            text = call.lower(x).as_text()
        self.assertEqual(text.count("tpu_custom_call"), 1)


class HelperTest(testing.TestCase):
    """The interpret helpers of `quantized_test_utils`."""

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
        with pallas_tpu_interpret():
            info = tpu_info.get_tpu_info()
        self.assertEqual(info.chip_version, tpu_info.ChipVersion.TPU_V5E)
        self.assertEqual(tpu_info.registry, registry)
        with self.assertRaisesRegex(ValueError, "Unsupported TPU device"):
            tpu_info.get_tpu_info()

    def test_interpret_route_values(self):
        """The interpret route of the route suites has the dtypes and the
        group guard of the kernel route, no version range and no kernel,
        so `"require"` does not run it. Its device check is whether TPU
        interpret mode works with the installed JAX."""
        source = kernels._ROUTES["pallas_tpu"]
        interpret = kernels._ROUTES["pallas_interpret"]
        self.assertEqual(interpret.dtypes, source.dtypes)
        self.assertTrue(interpret.contiguous_groups)
        self.assertIsNone(interpret.tested)
        self.assertIsNone(interpret.kernel)
        available = interpret.available(
            kernels._device_class(),
            3,
            256,
            128,
            _layer(CASES["dense"], "int4", 128)._quantized_weight(),
            "bfloat16",
        )
        self.assertEqual(available, pallas_tpu_interpret_available())


class GateTest(testing.TestCase):
    """The checks of the `Kernel` record, as tables. The route suites of
    `kernels_test.py` run them through real layers."""

    @parameterized.named_parameters(
        ("aligned", 256, 128, 128, True),
        ("groups_of_32", 1024, 256, 32, True),
        ("per_channel", 1152, 128, None, True),
        ("one_group", 384, 128, 384, True),
        ("group_above_k", 256, 128, 1024, True),
        ("full_extent", 256, 1152, 128, True),
        ("odd_columns", 256, 127, 128, False),
        ("rows_192", 192, 128, None, False),
        ("group_256_rows_384", 384, 128, 256, False),
        ("partial_group", 1152, 128, 256, False),
    )
    def test_matmul_gate(self, k, n, group, expected):
        """The shape check: an even `N`, and a `BK` that divides `K` and
        that the group size divides. The device check is separate
        (`supports_device`)."""
        kernel = kernel_record("pallas_tpu")
        available = kernel.available("tpu-TPU v6 lite", 3, k, n, group, "")
        self.assertEqual(available, expected)

    def test_device_values(self):
        """The device check admits a TPU of each kind only, the kernel
        takes bfloat16 and float32 compute only, and the kernel route reads
        the group of row `k` as `k // G`."""
        kernel = kernel_record("pallas_tpu")
        self.assertIs(kernels._ROUTES["pallas_tpu"].kernel, kernel)
        self.assertEqual(kernel.dtypes, ("bfloat16", "float32"))
        self.assertTrue(kernel.contiguous_groups)
        self.assertIsNone(kernel.logits)
        self.assertEqual(kernel.requirements, "a TPU")
        for device_class, expected in (
            ("tpu-TPU v5 lite", True),
            ("tpu-TPU v6 lite", True),
            ("cuda-sm89", False),
            ("cpu-arm64", False),
            ("mps", False),
            ("gpu", False),
        ):
            self.assertEqual(kernel.supports_device(device_class), expected)

    def test_versions(self):
        """The version range is empty, so no installed version is in it,
        and no measured row names the kernel."""
        self.assertEqual(quantized.PALLAS_TPU_TESTED_JAX, ())
        source = kernels._ROUTES["pallas_tpu"]
        self.assertIs(source.tested, quantized.pallas_tpu_tested)
        self.assertFalse(source.tested())
        self.assertFalse(quantized.pallas_tpu_tested(("0.12.0", "0.12.0")))
        for table in (kernels._MEASURED_ROWS, kernels._MEASURED_LOGITS_ROWS):
            for routes in table.values():
                self.assertNotIn("pallas_tpu", routes)

    def test_keras_hub_coverage(self):
        """The share of the projection weights of each preset that the
        alignment gate admits. It admits every weight with `G` 32, 64 and
        128 and per channel. With `G = 256`, the `K = 1152` weights of
        Gemma3-1B have no `BK` that 256 divides."""
        kernel = kernel_record("pallas_tpu")
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
                    if kernel.available("tpu-TPU v6 lite", 1, k, n, group, "")
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
