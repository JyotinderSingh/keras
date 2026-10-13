"""The Mosaic GPU logits kernel (`mosaic_gpu_w4a16_logits`) as an op.

The kernel of the int4 logits of `ReversibleEmbedding(reverse=True)`,
for tied and untied tables. Most tests need no GPU, as for the matmul
kernel (`quantized_test.py`). `mosaic_gpu_lowering` (`quantized_test_utils`)
lowers the kernel for each compute capability, with a check of every
untiled GMEM copy. The interpret route `"mosaic_gpu_interpret"` (from
`kernels_test_utils`) runs the kernel in GPU interpret mode on CPU, with
`jnp.dot` in place of `plgpu.mma`, through a layer with a configuration of
the test. `GPUTest` runs the real route on a CUDA GPU that passes the
device check, and the same cases in interpret mode elsewhere. The route
suites of `lookup_test.py` run the route through real layers. The module
needs JAX 0.11.2 or newer, which has `plgpu.mma`. JAX 0.11.2 needs Python
3.12, so a CI job on Python 3.11 skips it: `kernels_test.py` names the
Python version where a measured row has the device. The labels G1 to G8
are those of `kernels_test_utils`.
"""

import functools
import itertools
import json
import os

import jax
import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.jax.ops import quantized
from keras.src.backend.jax.ops.quantized_test_utils import kernel_record
from keras.src.backend.jax.ops.quantized_test_utils import mosaic_gpu_lowering
from keras.src.backend.jax.ops.quantized_test_utils import mosaic_gpu_routes
from keras.src.quantizers import kernels
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.kernels_test_utils import counts
from keras.src.quantizers.kernels_test_utils import direct_logits
from keras.src.quantizers.kernels_test_utils import route_tables
from keras.src.quantizers.kernels_test_utils import row
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.testing.test_utils import named_product

pytestmark = pytest.mark.skipif(
    backend.backend() != "jax" or not quantized.plgpu.available,
    reason=(
        "The Mosaic GPU kernels need the JAX backend and JAX 0.11.2 or newer, "
        f"which has `plgpu.mma`. Installed: JAX {jax.__version__}."
    ),
)

CAPABILITIES = ("8.0", "8.6", "8.9", "9.0")
ORIENTATIONS = ("tied", "untied")
# An `M` in each bucket.
BUCKET_ROWS = {"1-16": 4, "17-64": 33, "65-256": 130, "257+": 300}
# The logits shapes of keras-hub presets: name -> (D, V, orientation).
SHAPES = {
    "gemma2_2b": (2304, 256000, "tied"),
    "gemma3_1b": (1152, 262144, "tied"),
    "llama3_2_1b": (2048, 128256, "tied"),
    "llama3_8b": (4096, 128256, "untied"),
    "mistral_7b": (4096, 32768, "untied"),
}
# The table of the interpret cases: `D` spans eight K steps, and `V` four
# blocks of 128 columns.
DIM, VOCAB = 1024, 512
# The compute contexts: name -> (dtype policy, reverse dtype). A layer
# under a mixed policy reads its scale in the compute dtype. A float32
# layer with a 16-bit reverse dtype reads a float32 scale.
CONTEXTS = {
    "bfloat16": ("mixed_bfloat16", None),
    "bfloat16_float32_scale": ("float32", "bfloat16"),
    "float16": ("mixed_float16", None),
}
# name -> (BM, BV, stages, split-K) of a `GPUTest` case. The decode
# configuration needs about 71 KB of SMEM and the prefill one about 82 KB,
# so both need the opt-in above 48 KB.
GPU_CONFIGS = {"decode": (16, 128, 3, 2), "prefill": (64, 128, 2, 1)}
# The blocks in an error map: the worst ones.
MAP_BLOCKS = 8


def _layer(
    orientation,
    group_size,
    context="bfloat16",
    extra="plain",
    dim=DIM,
    vocab=VOCAB,
    config=None,
):
    """A quantized `ReversibleEmbedding` from random float tables, int4
    by default (per-channel with a `group_size` of -1). `context` is a
    name of `CONTEXTS` or a pair (dtype policy, reverse dtype). `extra`
    adds the soft cap or the tied LoRA term."""
    policy, reverse_dtype = CONTEXTS.get(context, context)
    rng = np.random.default_rng(0)
    layer = layers.ReversibleEmbedding(
        vocab,
        dim,
        tie_weights=orientation == "tied",
        reverse_dtype=reverse_dtype,
        logit_soft_cap=30.0 if extra == "soft_cap" else None,
        dtype=policy,
    )
    layer.build()
    layer._embeddings.assign(rng.normal(size=(vocab, dim)))
    if orientation == "untied":
        layer.reverse_embeddings.assign(rng.normal(size=(dim, vocab)))
    if config is None:
        config = Int4QuantizationConfig(block_size=group_size)
    layer.quantize(config.mode, config=config)
    if extra == "lora":
        layer.enable_lora(4)
        layer.lora_embeddings_b.assign(
            rng.normal(size=layer.lora_embeddings_b.shape)
        )
    return layer


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _dtype(layer):
    return layer.reverse_dtype or layer.compute_dtype


def _table(layer):
    """The stored table of the reverse lookup and its orientation, over
    the tensors that the layer reads."""
    strategy = strategy_registry.get_strategy(layer.quantization_mode)
    if layer.tie_weights:
        table = strategy._get_lookup_quantized_weight(layer, None)
    else:
        table = strategy._get_reverse_lookup_quantized_weight(layer, None)
    with AutocastScope(layer.compute_dtype):
        table = table.with_tensors(table.read_tensors())
    return table, "tied" if layer.tie_weights else "untied"


def _oracle(layer, x):
    """The float64 oracle of the logits, on the stored table and the scale
    that the layer reads. It shares no code with Keras: it unpacks the
    codes along `D`, the low nibble first, and sign-extends them."""
    table, orientation = _table(layer)
    packed = ops.convert_to_numpy(table.codes).astype(np.int64)
    if orientation == "tied":
        packed = packed.T
    # `(D, V)`: row `2j` holds the low nibble of byte `j`.
    q = np.stack([packed & 15, (packed >> 4) & 15], axis=1)
    q = q.reshape(-1, packed.shape[1])
    q = np.where(q > 7, q - 16, q)
    scale = _numpy(table.scale).astype(np.float64)
    if table.zero_point is None:
        weight = q / scale
    else:
        zero = ops.convert_to_numpy(table.zero_point).astype(np.int64)
        if orientation == "tied":
            scale, zero = scale.T, zero.T
        # Row `d` of the `(D, V)` table is in group `d // G`.
        groups = np.arange(q.shape[0]) // table.scheme.group_size
        weight = (q - zero[groups]) * scale[groups]
    dtype = _dtype(layer)
    rounding = ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype
    inputs = np.asarray(x, np.float32).astype(rounding).astype(np.float64)
    return inputs @ weight


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _limit(layer, x, oracle):
    """The largest error that G1 and G2 allow: 1.25 times the reference
    path's plus a floor, and at most an absolute bound."""
    floor, bound = (1e-3, 2**-7)
    if _dtype(layer) == "float16":
        floor, bound = (2e-4, 2**-9)
    with kernels.policy("off"):
        reference = _numpy(layer(x, reverse=True))
    return min(1.25 * _err(reference, oracle) + floor, bound)


def _check_accuracy(test, layer, x, y):
    """G1, G2 and G4: the error against the oracle is at most `_limit`."""
    oracle = _oracle(layer, x)
    y = _numpy(y)
    test.assertTrue(np.all(np.isfinite(y)))
    test.assertLessEqual(_err(y, oracle), _limit(layer, x, oracle))


def _interpret(config):
    """The interpret logits route with the configuration `config`."""
    return mosaic_gpu_routes(config=config, tested=None)[1]


def _config(m, v, split, block_m=16, stages=3):
    """A configuration with `BM = block_m`, by default a decode one, whose
    split-K rule gives `split` for `M = m` and `V = v`."""
    blocks = -(-m // block_m) * (v // 128)
    return (block_m, 128, stages, blocks * split)


def _exact_data(orientation, group_size, dtype, m):
    """G3 data: a table with small integer codes and zero points and
    power-of-two scales, and `m` inputs that keep every sum exact. `D`
    spans eight K steps, and `V` four blocks."""
    layer = _layer(orientation, group_size, dtype)
    rng = np.random.default_rng(2)
    codes = rng.integers(-1, 2, (DIM, VOCAB)).astype("int8")
    tied = orientation == "tied"
    if tied:
        variables = (layer._embeddings, layer.embeddings_scale)
        variables += (layer.embeddings_zero,)
        variables[0].assign(pack_int4(codes.T, axis=-1)[0])
    else:
        variables = (layer.reverse_embeddings, layer.reverse_embeddings_scale)
        variables += (layer.reverse_embeddings_zero,)
        variables[0].assign(pack_int4(codes, axis=0)[0])
    powers = rng.integers(1, 4, variables[1].shape)
    # A grouped scale multiplies and a per-channel scale divides.
    sign = 1 if group_size == -1 else -1
    variables[1].assign(2.0 ** (sign * powers))
    if variables[2] is not None:
        zero = rng.integers(-1, 2, variables[2].shape)
        variables[2].assign(zero.astype("int8"))
    # Eight nonzero inputs per row keep every sum exact. Seven random
    # signs have an odd sum, and the eighth sign repeats its sign, so
    # each row sums to at least 2 in magnitude. Then a per-channel
    # kernel that does not subtract `XOR` fails.
    x = np.zeros((m, DIM), "float32")
    for inputs in x:
        signs = rng.choice([-1, 1], 7)
        signs = np.append(signs, np.sign(signs.sum()))
        inputs[rng.choice(DIM, 8, replace=False)] = signs
    return layer, x


def _lower(
    capability,
    m,
    d,
    v,
    group_size,
    orientation,
    config,
    dtype="bfloat16",
    scale_dtype="float32",
):
    """The text of the module that `mosaic_gpu_w4a16_logits` lowers to for
    `capability`."""
    per_channel = group_size is None
    rows = -(-d // (group_size or d))
    codes, scale = (d // 2, v), (rows, v)
    if orientation == "tied":
        codes, scale = codes[::-1], scale[::-1]
    args = (
        jax.ShapeDtypeStruct((m, d), dtype),
        jax.ShapeDtypeStruct(codes, "int8"),
        jax.ShapeDtypeStruct((v,) if per_channel else scale, scale_dtype),
        None if per_channel else jax.ShapeDtypeStruct(scale, "int8"),
    )
    call = functools.partial(
        quantized.mosaic_gpu_w4a16_logits,
        group_size=None if per_channel else min(group_size, d),
        per_channel=per_channel,
        orientation=orientation,
        config=config,
    )
    with mosaic_gpu_lowering(capability):
        exported = jax.export.export(jax.jit(call), platforms=["cuda"])(*args)
    return exported.mlir_module()


class ConfigTest(testing.TestCase):
    def test_configuration_rows(self):
        """Each row has blocks that the MMA tiles in both orientations,
        and fits in the SMEM of sm_86 and sm_89 (99 KB per block)."""
        self.assertLen(quantized.MOSAIC_GPU_LOGITS_CONFIGS, 32)
        for key, config in quantized.MOSAIC_GPU_LOGITS_CONFIGS.items():
            block_m, block_v, stages, min_blocks = config
            self.assertEqual(quantized.MOSAIC_GPU_CONFIGS[key][0], block_m)
            self.assertIn(block_m, (16, 32, 64))
            # A row block holds 128 to 256 elements. The tied accumulator
            # `(BV, BM)` splits `BV` over four warps.
            self.assertIn(block_v, (128, 256))
            # On JAX 0.12.0, a tied table with more than 16 rows needs `BM`
            # 32 or 64.
            self.assertEqual(block_m == 16, key[2] == "1-16")
            stage = 2 * block_m * 128 + 64 * block_v + 2 * 4 * block_v
            smem = stages * stage + 2 * 128 * block_v
            self.assertLessEqual(smem, 99 * 1024)
            self.assertGreaterEqual(min_blocks, 1)


class LoweringTest(testing.TestCase):
    """The lowering for compute capability 8.0 to 9.0, with no untiled
    GMEM copy of a block that is not contiguous."""

    def setUp(self):
        super().setUp()
        try:
            with mosaic_gpu_lowering("8.9"):
                pass
        except NotImplementedError as e:
            self.skipTest(str(e))

    @parameterized.named_parameters(
        named_product(capability=CAPABILITIES, dtype=["bfloat16", "float16"])
    )
    def test_configuration_rows(self, capability, dtype):
        """Each row of the capability, for both orientations, in groups of
        128 and per channel. `V` spans several blocks, and `D` eight
        groups."""
        for bucket, m in BUCKET_ROWS.items():
            key = (capability, dtype, bucket)
            config = quantized.MOSAIC_GPU_LOGITS_CONFIGS[key]
            for orientation, group_size in itertools.product(
                ORIENTATIONS, (128, None)
            ):
                text = _lower(
                    capability,
                    m,
                    DIM,
                    VOCAB,
                    group_size,
                    orientation,
                    config,
                    dtype=dtype,
                )
                self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(named_product(capability=CAPABILITIES))
    def test_logits_shapes(self, capability):
        """The five logits shapes in their orientation: in groups of 128
        at each row of the capability, and at decode in groups of 256
        (Gemma3-1B has 4.5 groups) with a bfloat16 scale, and per
        channel."""
        for d, v, orientation in SHAPES.values():
            for bucket, m in BUCKET_ROWS.items():
                key = (capability, "bfloat16", bucket)
                config = quantized.MOSAIC_GPU_LOGITS_CONFIGS[key]
                text = _lower(capability, m, d, v, 128, orientation, config)
                self.assertIn("mosaic_gpu_v2", text)
            config = quantized.MOSAIC_GPU_LOGITS_CONFIGS[
                (capability, "bfloat16", "1-16")
            ]
            for group_size, scale_dtype in (
                (256, "bfloat16"),
                (None, "float32"),
            ):
                text = _lower(
                    capability,
                    1,
                    d,
                    v,
                    group_size,
                    orientation,
                    config,
                    scale_dtype=scale_dtype,
                )
                self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(
        named_product(
            capability=CAPABILITIES,
            orientation=ORIENTATIONS,
            split=[1, 2, 4, 8],
        )
    )
    def test_split_k(self, capability, orientation, split):
        config = _config(1, VOCAB, split)
        text = _lower(capability, 1, DIM, VOCAB, 256, orientation, config)
        self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(named_product(capability=CAPABILITIES))
    def test_256_columns_per_block(self, capability):
        """`BV = 256` with one stage, for both orientations, in groups of
        128 and per channel, at `BM` 16, 32 and 64. `V` spans two blocks.
        One stage needs 86 KB to 98 KB of SMEM, so it fits in the 99 KB of
        sm_86 and sm_89."""
        for (block_m, m), orientation, group_size in itertools.product(
            ((16, 4), (32, 33), (64, 130)), ORIENTATIONS, (128, None)
        ):
            stage = 2 * block_m * 128 + 64 * 256 + 2 * 4 * 256
            self.assertLessEqual(stage + 2 * 128 * 256, 99 * 1024)
            config = (block_m, 256, 1, 1)
            text = _lower(
                capability, m, DIM, VOCAB, group_size, orientation, config
            )
            self.assertIn("mosaic_gpu_v2", text)

    @parameterized.named_parameters(
        named_product(capability=["8.9", "9.0"], orientation=ORIENTATIONS)
    )
    def test_layer_lowers_no_guard(self, capability, orientation):
        """Through a layer and the real route, an int4 table in groups of
        128 and per channel lowers to one Mosaic GPU call, with no guard
        (`stablehlo.case`): column `d` of a table is in group `d // G`."""
        key = (capability, "bfloat16", "1-16")
        config = quantized.MOSAIC_GPU_LOGITS_CONFIGS[key]
        real = mosaic_gpu_routes(False, config, tested=None)[1]
        args = jax.ShapeDtypeStruct((4, DIM), "float32")
        for group in (128, -1):
            layer = _layer(orientation, group)
            self.assertIsNone(_table(layer)[0].g_idx)
            rows = row("bfloat16", 4, "real")
            with route_tables(logits_rows=rows, logits_routes={"real": real}):
                call = jax.jit(lambda v: layer(v, reverse=True))
                with mosaic_gpu_lowering(capability):
                    exported = jax.export.export(call, platforms=["cuda"])(args)
            text = exported.mlir_module()
            self.assertEqual(text.count("mosaic_gpu_v2"), 1, group)
            self.assertNotIn("stablehlo.case", text)

    def test_each_capability_traces_again(self):
        """The pipeline reads the capability at trace time: `cp.async`
        below 9.0, and TMA with barriers from 9.0. So each lowering of one
        signature traces the logits kernel again, and only the lowering
        for 9.0 holds barriers."""
        config = _config(1, VOCAB, 1)
        for capability in ("8.9", "9.0", "8.9"):
            text = _lower(capability, 1, DIM, VOCAB, 128, "untied", config)
            self.assertEqual("nvvm.mbarrier" in text, capability == "9.0")


class InterpretTest(testing.TestCase):
    """The kernel in GPU interpret mode, through real layers."""

    @parameterized.named_parameters(
        named_product(
            orientation=ORIENTATIONS,
            group=[128, 256, -1],
            context=sorted(CONTEXTS),
        )
    )
    def test_against_the_oracle(self, orientation, group, context):
        """G1, G2, G4 and G5 at `M` 1, 3 and 17, with split-K 4, 2 and 1,
        each value twice. `V` spans four blocks, and `D` eight K steps.
        The frontend pads `M` to `BM`: 16, or 32 above 16 rows."""
        layer = _layer(orientation, group, context)
        for m, block_m, split in ((1, 16, 4), (3, 16, 2), (17, 32, 1)):
            x = _random((m, DIM), seed=m)
            config = _config(m, VOCAB, split, block_m)
            rows = row(_dtype(layer), m, "mosaic_gpu_interpret")
            interpret = {"mosaic_gpu_interpret": _interpret(config)}
            kernels.route_counts.clear()
            with route_tables(logits_rows=rows, logits_routes=interpret):
                y = layer(x, reverse=True)
                again = layer(x, reverse=True)
            self.assertEqual(
                dict(kernels.route_counts), {"mosaic_gpu_interpret": 2}
            )
            self.assertAllEqual(_numpy(again), _numpy(y))
            _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        named_product(orientation=ORIENTATIONS, group=[128, -1])
    )
    def test_configuration_rows(self, orientation, group):
        """`force_route` takes the rows of compute capability 8.9: `BM` 16
        and 32, with two blocks of rows at `M = 33` and five at
        `M = 130`. The layer output equals a direct call of the route
        (G8)."""
        layer = _layer(orientation, group)
        for m in (4, 33, 130):
            x = _random((m, DIM), seed=m)
            with kernels.force_route("mosaic_gpu_interpret"):
                selected = counts(layer, x, reverse=True)
                self.assertEqual(selected, {"mosaic_gpu_interpret": 1})
                y, called = layer(x, reverse=True), direct_logits(layer, x)
            self.assertAllEqual(_numpy(called), _numpy(y))
            _check_accuracy(self, layer, x, y)

    @parameterized.named_parameters(
        named_product(
            orientation=ORIENTATIONS,
            group=[128, 256, -1],
            dtype=["bfloat16", "float16"],
        )
    )
    def test_exact_data(self, orientation, group, dtype):
        """G3: small integers and power-of-two scales give the oracle bit
        for bit. `V` spans four blocks and `D` eight K steps, so a wrong
        nibble, column order, group or block shows."""
        layer, x = _exact_data(orientation, group, dtype, 3)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x, reverse=True)), oracle)
        config = _config(3, VOCAB, 2)
        rows = row(_dtype(layer), 3, "mosaic_gpu_interpret")
        interpret = {"mosaic_gpu_interpret": _interpret(config)}
        with route_tables(logits_rows=rows, logits_routes=interpret):
            selected = counts(layer, x, reverse=True)
            self.assertEqual(selected, {"mosaic_gpu_interpret": 1})
            self.assertAllEqual(_numpy(layer(x, reverse=True)), oracle)

    @parameterized.named_parameters(
        named_product(orientation=ORIENTATIONS, group=[128, -1])
    )
    def test_eager_call_compiles_once(self, orientation, group):
        """A second eager call with the same shapes, dtypes and static
        arguments compiles nothing. Its output equals that of the function
        body run op by op bit for bit."""
        layer = _layer(orientation, group)
        table, _ = _table(layer)
        grouped = table.zero_point is not None
        args = (
            ops.cast(_random((3, DIM)), "bfloat16"),
            table.codes,
            table.scale,
            table.zero_point,
        )
        options = {
            "group_size": table.scheme.group_size if grouped else None,
            "per_channel": not grouped,
            "orientation": orientation,
            "config": _config(3, VOCAB, 2),
            "interpret": True,
        }
        compiles = []

        def count(event, duration_secs, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles.append(duration_secs)

        first = quantized.mosaic_gpu_w4a16_logits(*args, **options)
        jax.monitoring.register_event_duration_secs_listener(count)
        try:
            second = quantized.mosaic_gpu_w4a16_logits(*args, **options)
        finally:
            jax.monitoring.unregister_event_duration_listener(count)
        self.assertEqual(compiles, [])
        self.assertAllEqual(_numpy(second), _numpy(first))
        body = quantized._mosaic_gpu_w4a16_logits.__wrapped__(*args, **options)
        self.assertAllEqual(_numpy(first), _numpy(body))


class GPUTest(testing.TestCase):
    """The kernel on the default device: the real route `"mosaic_gpu"` on
    a CUDA GPU that passes the device gate, else the interpret route. A
    measured row gives the configuration, so on a GPU the real route
    needs an equal jax and jaxlib version in `MOSAIC_GPU_TESTED_JAX`. With
    `KSTAGE_OUT` set, a case that fails writes its error map there."""

    def setUp(self):
        super().setUp()
        capability = getattr(jax.devices()[0], "compute_capability", None)
        gpu = capability in quantized._MOSAIC_GPU_CAPABILITIES
        self.route = "mosaic_gpu" if gpu else "mosaic_gpu_interpret"

    def assertMatches(self, layer, x, oracle, config, limit):
        """Through a row of the route with the configuration `config`, the
        error of the logits against `oracle` is at most `limit`."""
        m, v = oracle.shape
        block_m, block_v, stages, split = GPU_CONFIGS[config]
        config_row = _config(m, v, split, block_m, stages)
        # The route with the configuration `config_row`: the real kernel
        # on a GPU, with its version range, else the interpret route.
        interpret = self.route == "mosaic_gpu_interpret"
        changes = {"tested": None} if interpret else {}
        source = mosaic_gpu_routes(interpret, config_row, **changes)[1]
        kernels.route_counts.clear()
        rows = row(_dtype(layer), m, self.route)
        with route_tables(logits_rows=rows, logits_routes={self.route: source}):
            y = _numpy(layer(x, reverse=True))
            self.assertEqual(dict(kernels.route_counts), {self.route: 1})
            err = _err(y, oracle)
            if not err <= limit and os.environ.get("KSTAGE_OUT"):
                self._write_error_map(layer, x, y, oracle, config, limit)
        self.assertLessEqual(err, limit)

    def _write_error_map(self, layer, x, y, oracle, config, limit):
        """Writes `mosaic_logits_error_map_<case>.json` to `$KSTAGE_OUT`,
        with the worst blocks `(m, v, s)`. The reference of split `s` is
        the oracle on the inputs of its K steps and zeros elsewhere: the
        logits of those inputs are the partial sum of `s`."""
        block_m, block_v, stages, split = GPU_CONFIGS[config]
        (m, d), v = x.shape, oracle.shape[1]
        tolerance = limit * np.max(np.abs(oracle))
        blocks = []
        for s in range(split):
            got, want = y, oracle
            if split > 1:
                part = np.zeros_like(x)
                steps = slice(s * d // split, (s + 1) * d // split)
                part[:, steps] = x[:, steps]
                got = _numpy(layer(part, reverse=True))
                want = _oracle(layer, part)
            for i, j in itertools.product(
                range(-(-m // block_m)), range(v // block_v)
            ):
                rows = slice(i * block_m, (i + 1) * block_m)
                columns = slice(j * block_v, (j + 1) * block_v)
                diff = np.abs(got[rows, columns] - want[rows, columns])
                wrong = np.argwhere(~(diff <= tolerance))
                first = None
                if len(wrong):
                    r, c = wrong[0] + (i * block_m, j * block_v)
                    first = [
                        int(r),
                        int(c),
                        float(got[r, c]),
                        float(want[r, c]),
                    ]
                error = float(np.nan_to_num(np.max(diff), nan=np.inf))
                blocks.append(
                    {"m": i, "v": j, "s": s, "max_abs_err": error}
                    | {"first_wrong": first}
                )
        case = self._testMethodName.removeprefix("test_")
        table, orientation = _table(layer)
        error_map = {
            "case": case,
            "dtype": _dtype(layer),
            "jax": jax.__version__,
            "orientation": orientation,
            "shape": {"M": m, "D": d, "V": v, "G": table.scheme.group_size},
            "config": {
                "BM": block_m,
                "BV": block_v,
                "STAGES": stages,
                "SPLIT_K": split,
            },
            "wrong_blocks": sum(b["first_wrong"] is not None for b in blocks),
            "blocks": sorted(
                blocks, key=lambda b: b["max_abs_err"], reverse=True
            )[:MAP_BLOCKS],
        }
        name = f"mosaic_logits_error_map_{case}.json"
        with open(os.path.join(os.environ["KSTAGE_OUT"], name), "w") as f:
            json.dump(error_map, f, indent=1)

    @parameterized.named_parameters(
        named_product(
            config=sorted(GPU_CONFIGS),
            orientation=ORIENTATIONS,
            group=[128, -1],
            dtype=["bfloat16", "float16"],
        )
    )
    def test_exact_data(self, config, orientation, group, dtype):
        """G3 with `V` over four blocks and `D` over eight K steps: `M = 3`
        with the decode configuration, and `M = 80`, two blocks of rows,
        with the prefill one."""
        m = 3 if config == "decode" else 80
        layer, x = _exact_data(orientation, group, dtype, m)
        self.assertMatches(layer, x, _oracle(layer, x), config, 0.0)

    @parameterized.named_parameters(
        named_product(
            config=sorted(GPU_CONFIGS),
            orientation=ORIENTATIONS,
            group=[128, -1],
            dtype=["bfloat16", "float16"],
        )
    )
    def test_against_the_oracle(self, config, orientation, group, dtype):
        """G1 and G2 on random data with `D = V = 1024`: `M = 1` with the
        decode configuration, and `M = 80` with the prefill one."""
        layer = _layer(orientation, group, dtype, vocab=1024)
        x = _random((1 if config == "decode" else 80, DIM))
        oracle = _oracle(layer, x)
        self.assertMatches(layer, x, oracle, config, _limit(layer, x, oracle))

    @parameterized.named_parameters(("tied", "tied"), ("untied", "untied"))
    @pytest.mark.skipif(
        not testing.jax_uses_gpu(), reason="Runs the kernel on a GPU."
    )
    def test_require_runs_the_kernel(self, orientation):
        """G8 on hardware: `"require"` runs the Mosaic GPU logits kernel,
        with the configuration of the device. `M = 4` takes the `BM = 16`
        row of the bucket `"1-16"`, which serves a tied table too."""
        layer = _layer(orientation, 128)
        x = _random((4, DIM))
        with kernels.policy("require"):
            self.assertEqual(counts(layer, x, reverse=True), {"mosaic_gpu": 1})
            y = layer(x, reverse=True)
            self.assertAllEqual(_numpy(y), _numpy(direct_logits(layer, x)))


class GateTest(testing.TestCase):
    """The checks of the `Kernel` record, as tables, and the configurations
    that the kernel refuses. The route suites of `lookup_test.py` run them
    through real layers."""

    def test_rows_have_a_configuration(self):
        """Each Mosaic GPU measured row of the reverse lookup names a device
        class and a compute dtype that the kernel supports, and
        `MOSAIC_GPU_LOGITS_CONFIGS` has a row for its compute capability,
        compute dtype and M bucket."""
        kernel = kernel_record("mosaic_gpu")
        rows = [
            key
            for key, routes in kernels._MEASURED_LOGITS_ROWS.items()
            if "mosaic_gpu" in routes
        ]
        self.assertNotEqual(rows, [])
        for key in rows:
            _, device_class, dtype, bucket = key
            self.assertTrue(kernel.supports_device(device_class), key)
            self.assertIn(dtype, kernel.dtypes, key)
            number = device_class.removeprefix("cuda-sm")
            capability = f"{number[:-1]}.{number[-1]}"
            config_key = (capability, dtype, bucket)
            self.assertIn(config_key, quantized.MOSAIC_GPU_LOGITS_CONFIGS, key)

    def test_route_values(self):
        """The logits routes have the dtypes, the version range and the
        kernel of the matmul route. The interpret route has no version
        range and no kernel, so `"require"` does not run it."""
        matmul = kernels._ROUTES["mosaic_gpu"]
        source = kernels._LOGITS_ROUTES["mosaic_gpu"]
        interpret = kernels._LOGITS_ROUTES["mosaic_gpu_interpret"]
        self.assertEqual(source.dtypes, matmul.dtypes)
        self.assertEqual(interpret.dtypes, matmul.dtypes)
        self.assertIs(source.tested, quantized.mosaic_gpu_tested)
        self.assertIs(source.kernel, matmul.kernel)
        self.assertIsNone(interpret.tested)
        self.assertIsNone(interpret.kernel)

    @parameterized.named_parameters(
        ("aligned", 256, 256, 128, True),
        ("per_channel", 256, 256, None, True),
        ("one_group", 256, 256, 512, True),
        ("vocab_192", 256, 192, 128, False),
        ("dim_192", 192, 256, None, False),
        ("group_64", 256, 256, 64, False),
    )
    def test_logits_gate(self, d, v, group, expected):
        """The shape check of the logits: blocks of 128 columns of `V`,
        then the alignment of the matmul for `K = D` and `N = V`. The
        device check is separate (`supports_device`)."""
        kernel = kernel_record("mosaic_gpu")
        available = kernel.logits_available("cuda-sm89", 3, d, v, group, "")
        self.assertEqual(available, expected)

    @parameterized.named_parameters(
        ("tied_33_rows_bm16", "tied", 33, 16, False),
        ("untied_33_rows_bm16", "untied", 33, 16, True),
        ("tied_16_rows_bm16", "tied", 16, 16, True),
        ("tied_33_rows_bm32", "tied", 33, 32, True),
    )
    def test_tied_rows_of_16(self, orientation, m, block_m, runs):
        """A configuration with `BM = 16` serves a tied table only up to 16
        rows: on JAX 0.12.0, more blocks of 16 rows do not lower. The
        kernel raises for such a tied call, and runs the untied one and
        the other configurations. The rows of
        `MOSAIC_GPU_LOGITS_CONFIGS` give `BM = 16` only up to 16 rows
        (`ConfigTest`)."""
        layer = _layer(orientation, 128, dim=256, vocab=256)
        x = _random((m, 256))
        table, _ = _table(layer)

        def call():
            return quantized.mosaic_gpu_w4a16_logits(
                ops.cast(x, _dtype(layer)),
                table.codes,
                table.scale,
                table.zero_point,
                group_size=128,
                per_channel=False,
                orientation=orientation,
                config=(block_m, 128, 3, 1),
                interpret=True,
            )

        if not runs:
            with self.assertRaisesRegex(ValueError, "BM` 32 or 64"):
                call()
            return
        _check_accuracy(self, layer, x, call())
