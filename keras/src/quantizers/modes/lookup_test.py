"""int8 and int4 store an embeddings table through one `LookupHandlers`.

The int4 reverse lookup selects a route of `kernels.py`. The tests run the
reference contraction as a route (`force_route("reference_wrapped")`),
and `matrix`, a route of this file. `quantized_logits_test.py` of the
torch backend tests the Triton route.
"""

import contextlib
import functools
import itertools
import warnings

import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import distribution
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers import kernels
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.modes.int8 import Int8Strategy
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_test_utils import reordered_sum_bound
from keras.src.quantizers.quantized_weight import WeightScheme
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing.test_utils import named_product

if backend.backend() == "jax":
    import jax
elif backend.backend() == "torch":
    import torch
elif backend.backend() == "tensorflow":
    import tensorflow as tf

VOCAB, DIM = 96, 256
# The int4 reverse lookups through a route: a tied or untied table, a
# group size, a policy and a reverse dtype, the layer on its own or in a
# `mixed_bfloat16` model, and the soft cap or the tied LoRA term. On its
# own, a `mixed_bfloat16` layer opens a bfloat16 autocast scope, so its
# scale reads bfloat16. In the model, the `autocast=False` of the layer
# clears the scope of the model, so its scale reads float32. A float32
# layer with a bfloat16 reverse dtype also reads a float32 scale. Only in
# these two contexts does a grouped table with a bfloat16 reverse dtype
# dequantize in float32.
ROUTED_CASES = [
    {
        "testcase_name": f"{tie}_{block}_{policy}_{reverse}_{context}_{extra}",
        "tie_weights": tie == "tied",
        "block_size": block,
        "policy": policy,
        "reverse_dtype": reverse,
        "in_model": context == "model",
        "extra": extra,
    }
    for tie in ("tied", "untied")
    for block in (64, 128, -1)
    for policy, reverse, context in (
        ("float32", None, "layer"),
        ("float32", "bfloat16", "layer"),
        ("mixed_bfloat16", None, "layer"),
        ("mixed_bfloat16", "float32", "layer"),
        ("mixed_bfloat16", None, "model"),
    )
    for extra in ("plain", "soft_cap", "lora")
    if tie == "tied" or extra != "lora"
]


class ToyModeConfig(Int8QuantizationConfig):
    """The int8 config under the name of a toy mode."""

    def __init__(self, mode):
        super().__init__()
        self._mode = mode

    @property
    def mode(self):
        return self._mode


class MultiplierScaleInt8(Int8Strategy):
    """int8 with an ungrouped multiplier scale for the table."""

    name = "demo_multiplier"

    def _get_lookup_scheme(self, layer, config):
        return WeightScheme(code_range=(-127, 127), scale_form="multiplier")


class NoTableViewInt8(Int8Strategy):
    """int8 without a `QuantizedWeight` view of the table."""

    name = "demo_no_view"

    def _get_lookup_quantized_weight(self, layer, geometry):
        return None


class ToyModesEmbedding(layers.Embedding):
    """An `Embedding` that stores the toy modes as int8 does."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        for mode in (MultiplierScaleInt8.name, NoTableViewInt8.name):
            spec[mode] = spec["int8"]
        return spec


class LookupHandlersTest(testing.TestCase):
    @parameterized.named_parameters(
        ("multiplier_scale", MultiplierScaleInt8, "does not implement"),
        ("no_table_view", NoTableViewInt8, "cannot write its codes"),
    )
    def test_quantize_refuses_a_mode_the_handlers_cannot_serve(
        self, strategy_cls, message
    ):
        strategy_registry.register_quantization_strategy(strategy_cls)
        self.addCleanup(
            strategy_registry.unregister_quantization_strategy,
            strategy_cls.name,
        )
        layer = ToyModesEmbedding(10, 4)
        layer.build()
        before = ops.convert_to_numpy(layer.embeddings)
        with self.assertRaisesRegex(NotImplementedError, message):
            layer.quantize(
                config=ToyModeConfig(strategy_cls.name), type_check=False
            )
        self.assertIsNone(layer.quantization_mode)
        self.assertAllEqual(layer.embeddings, before)

    @parameterized.named_parameters(
        named_product(
            tie_weights=(True, False),
            reverse_dtype=("bfloat16", "float16"),
        )
    )
    def test_grouped_reverse_keeps_the_table_in_float32(
        self, tie_weights, reverse_dtype
    ):
        # Below the compute dtype, only the inputs and the logits are cast
        # to `reverse_dtype`: the dequantized table keeps the precision of
        # the float32 scale.
        rng = np.random.default_rng(0)
        layer = layers.ReversibleEmbedding(
            37, 13, tie_weights=tie_weights, reverse_dtype=reverse_dtype
        )
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if not tie_weights:
            layer.reverse_embeddings.assign(rng.normal(size=(13, 37)))
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=4))
        views = strategy_registry.get_strategy("int4").quantized_weights(layer)
        if tie_weights:
            table = ops.transpose(views[0].dequantize("float32"))
        else:
            table = views[1].dequantize("float32")
        inputs = rng.normal(size=(3, 13)).astype("float32") * 4.0

        logits = layer(inputs, reverse=True)

        inputs = ops.cast(ops.cast(inputs, reverse_dtype), "float32")
        expected = ops.cast(ops.matmul(inputs, table), reverse_dtype)
        self.assertDType(logits, reverse_dtype)
        self.assertAllEqual(logits, expected)

    @parameterized.named_parameters(
        named_product(
            block_size=(None, -1, 4),
            ids_shape=((), (5,), (2, 3), (2, 2, 2)),
            policy=("float32", "mixed_bfloat16"),
        )
    )
    def test_forward_is_the_rows_of_the_dequantized_table(
        self, block_size, ids_shape, policy
    ):
        # The lookup gathers the stored rows and dequantizes only those,
        # which gives the rows of the dequantized table bit for bit.
        rng = np.random.default_rng(0)
        layer = layers.Embedding(37, 13, dtype=policy)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if block_size is None:
            mode, config = "int8", None
        else:
            mode = "int4"
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        ids = rng.integers(0, 37, size=ids_shape)

        outputs = layer(ids)

        (view,) = strategy_registry.get_strategy(mode).quantized_weights(layer)
        with AutocastScope(layer.compute_dtype):
            table = view.dequantize(layer.compute_dtype)
        self.assertAllEqual(outputs, ops.take(table, ids, axis=0))

    @parameterized.named_parameters(
        ("tied", True),
        ("untied", False),
    )
    def test_weight_only_reverse_projects_through_the_dequantized_table(
        self, tie_weights
    ):
        # Per-channel int4 contracts with the codes and applies the scale
        # to the logits.
        rng = np.random.default_rng(0)
        layer = layers.ReversibleEmbedding(37, 13, tie_weights=tie_weights)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if not tie_weights:
            layer.reverse_embeddings.assign(rng.normal(size=(13, 37)))
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=-1))
        strategy = strategy_registry.get_strategy("int4")
        table = strategy._get_reverse_lookup_quantized_weight(layer, None)
        inputs = rng.normal(size=(3, 13)).astype("float32")

        logits = layer(inputs, reverse=True)

        self.assertAllClose(
            logits,
            ops.matmul(inputs, table.dequantize("float32")),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.named_parameters(
        named_product(block_size=(None, -1, 4), dtype=(None, "bfloat16"))
    )
    def test_forward_in_a_mixed_precision_model_reads_the_scale_dtype(
        self, block_size, dtype
    ):
        # An `Embedding` does not autocast, so inside a model under
        # `mixed_bfloat16` its scale reads in its own dtype (float32 for
        # the global policy, bfloat16 for a `bfloat16` layer), although
        # the variable reports `floatx()` there, and the rows are
        # dequantized in that dtype.
        dtype_policies.dtype_policy.set_dtype_policy("mixed_bfloat16")
        self.addCleanup(dtype_policies.dtype_policy.set_dtype_policy, "float32")
        rng = np.random.default_rng(0)
        layer = layers.Embedding(37, 13, dtype=dtype)
        layer.build()
        layer._embeddings.assign(rng.normal(size=(37, 13)))
        if block_size is None:
            mode, config = "int8", None
        else:
            mode = "int4"
            config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize(mode, config=config)
        inputs = layers.Input(shape=(None,), dtype="int32")
        model = models.Model(inputs, layer(inputs))
        ids = rng.integers(0, 37, size=(2, 3))

        outputs = model(ids)

        (view,) = strategy_registry.get_strategy(mode).quantized_weights(layer)
        scale_dtype = layer.variable_dtype
        self.assertDType(outputs, scale_dtype)
        self.assertAllEqual(
            outputs, ops.take(view.dequantize(scale_dtype), ids, axis=0)
        )


def _reversible(
    tie_weights,
    block_size,
    policy="float32",
    reverse_dtype=None,
    extra="plain",
    config=None,
    mode="int4",
):
    """A quantized `ReversibleEmbedding` from random float tables."""
    rng = np.random.default_rng(0)
    layer = layers.ReversibleEmbedding(
        VOCAB,
        DIM,
        tie_weights=tie_weights,
        reverse_dtype=reverse_dtype,
        logit_soft_cap=30.0 if extra == "soft_cap" else None,
        dtype=policy,
    )
    layer.build()
    layer._embeddings.assign(rng.normal(size=(VOCAB, DIM)))
    if not tie_weights:
        layer.reverse_embeddings.assign(rng.normal(size=(DIM, VOCAB)))
    if config is None and mode == "int4":
        config = Int4QuantizationConfig(block_size=block_size)
    layer.quantize(mode, config=config)
    if extra == "lora":
        layer.enable_lora(4)
        layer.lora_embeddings_b.assign(
            rng.normal(size=layer.lora_embeddings_b.shape)
        )
    return layer


class ReverseLookupModel(models.Model):
    """A model that runs the reverse lookup of `inner`."""

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, inputs):
        return self.inner(inputs, reverse=True)


def _random(shape, seed):
    return np.random.default_rng(seed).normal(size=shape).astype("float32")


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _bits(x):
    """The bits of `x` as float32, so that signed zeros differ too."""
    return _numpy(x).view("int32")


def _logits_and_gradient(call, x, upstream, compiled=False):
    """The logits `call(x)`, and their input gradient for `upstream`.

    `compiled` runs `jax.jit`, `tf.function` or `torch.compile`. NumPy
    gives no gradient.
    """
    if backend.backend() == "jax":

        def run(v, u):
            y, vjp = jax.vjp(call, v)
            return y, vjp(ops.cast(u, y.dtype))[0]

        return (jax.jit(run) if compiled else run)(x, upstream)
    if backend.backend() == "torch":
        if compiled:
            torch._dynamo.reset()
            call = torch.compile(call)
        v = torch.tensor(x, requires_grad=True)
        y = call(v)
        y.backward(ops.cast(upstream, y.dtype))
        return y, v.grad
    if backend.backend() == "tensorflow":

        def run(v, u):
            with tf.GradientTape() as tape:
                tape.watch(v)
                y = call(v)
            u = tf.cast(u, y.dtype)
            return y, tape.gradient(y, v, output_gradients=u)

        run = tf.function(run) if compiled else run
        return run(tf.constant(x), tf.constant(upstream))
    return call(x), None


def _gradient_bound(layer, upstream, gradient):
    """How far an input gradient of the custom gradient can be from
    autodiff's `gradient`.

    Each entry sums `V` products of the output gradient with one row of
    the `(D, V)` table, in float32: `reordered_sum_bound`. Then each side
    rounds twice in the reverse dtype: the cast of the sum, and the sum
    with the gradient of the tied LoRA term. Each rounding is at most
    half an ulp of the largest entry.
    """
    dtype = layer.reverse_dtype or layer.compute_dtype
    stored, orientation = _stored(layer)
    table = kernels.reverse_lookup_view(stored, orientation)
    rows = np.abs(_numpy(table.dequantize("float32")))
    upstream = np.abs(_numpy(ops.cast(upstream, dtype)))
    magnitude = np.max(upstream) * np.max(np.sum(rows, axis=1))
    peak = np.max(np.abs(_numpy(gradient)))
    ulp = 2.0 ** (np.floor(np.log2(peak)) - ml_dtypes.finfo(dtype).nmant)
    return reordered_sum_bound(magnitude, rows.shape[1]) + 2 * ulp


def _stored(layer):
    """The stored table of the reverse lookup, and its orientation."""
    strategy = strategy_registry.get_strategy(layer.quantization_mode)
    if layer.tie_weights:
        return strategy._get_lookup_quantized_weight(layer, None), "tied"
    return strategy._get_reverse_lookup_quantized_weight(layer, None), "untied"


def _matrix_logits(layer, table, orientation, inputs, config):
    """A `matrix` route: the `(M, D)` inputs times the table as stored."""
    weight = table.dequantize(layer.reverse_dtype or layer.compute_dtype)
    if orientation == "tied":
        weight = ops.transpose(weight)
    return ops.matmul(inputs, weight)


@contextlib.contextmanager
def _lookup_routes(rows, **sources):
    """Adds `sources` to the reverse lookup routes and sets their
    allow-list to `rows`. The tables are as before when the block ends."""
    allow_list = dict(kernels._LOOKUP_ALLOW_LIST)
    kernels._LOOKUP_ROUTES.update(sources)
    kernels._LOOKUP_ALLOW_LIST.clear()
    kernels._LOOKUP_ALLOW_LIST.update(rows)
    try:
        yield
    finally:
        for name in sources:
            del kernels._LOOKUP_ROUTES[name]
        kernels._LOOKUP_ALLOW_LIST.clear()
        kernels._LOOKUP_ALLOW_LIST.update(allow_list)


def _row_key(dtype, bucket):
    """The allow-list key of this backend and device."""
    device = None
    if backend.backend() == "torch":
        device = str(ops.convert_to_tensor(np.ones(1)).device)
    return (backend.backend(), kernels._device_class(device), dtype, bucket)


def _counts(layer, x):
    """The routes that one reverse call of `layer` selects."""
    kernels.route_counts.clear()
    layer(x, reverse=True)
    return dict(kernels.route_counts)


def _first_tested_lookup_route(routes):
    """The first route of a reverse lookup row that this process has, with
    its library version in its tested range, else `"reference"`."""
    for name, _ in routes:
        source = kernels._LOOKUP_ROUTES.get(name)
        if source is None:
            continue
        if source.version is None or kernels.tested(
            source.version(), source.tested_versions
        ):
            return name
    return "reference"


class ReverseLookupRouteTest(testing.TestCase):
    """The int4 reverse lookup selects a route and keeps its gradient."""

    @parameterized.named_parameters(ROUTED_CASES)
    def test_reference_wrapped_is_the_reference(
        self, tie_weights, block_size, policy, reverse_dtype, in_model, extra
    ):
        """Through the custom gradient, the logits are the reference's bit
        for bit, eagerly and compiled. The input gradient repeats the
        arithmetic of autodiff, but its matmul can sum in another order:
        `_gradient_bound`."""
        layer = _reversible(
            tie_weights, block_size, policy, reverse_dtype, extra
        )
        x = _random((2, 3, DIM), 1)
        upstream = _random((2, 3, VOCAB), 2)
        call = functools.partial(layer, reverse=True)
        if in_model:
            call = ReverseLookupModel(layer, dtype=policy)
        compiled = (False, True)
        if backend.backend() in ("numpy", "torch"):
            # NumPy has no gradients. `test_torch_compile` compiles.
            compiled = (False,)
        for compile_ in compiled:
            with kernels.policy("off"):
                expected = _logits_and_gradient(call, x, upstream, compile_)
            kernels.route_counts.clear()
            with kernels.force_route("reference_wrapped"):
                actual = _logits_and_gradient(call, x, upstream, compile_)
            self.assertEqual(
                dict(kernels.route_counts), {"reference_wrapped": 1}
            )
            self.assertAllEqual(_bits(actual[0]), _bits(expected[0]))
            if expected[1] is not None:
                bound = _gradient_bound(layer, upstream, expected[1])
                self.assertAllClose(actual[1], expected[1], atol=bound, rtol=0)

    @parameterized.named_parameters(
        named_product(tie_weights=(True, False), block_size=(64, -1))
    )
    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Compiles with `torch.compile`."
    )
    def test_torch_compile(self, tie_weights, block_size):
        # float32 only: Dynamo breaks the graph at `CustomGradientFunction`,
        # and inductor skips the rounding of a bfloat16 cast that it fuses,
        # so in bfloat16 the two compiled forwards round at other places.
        layer = _reversible(tie_weights, block_size, extra="soft_cap")
        call = functools.partial(layer, reverse=True)
        x = _random((2, 3, DIM), 1)
        upstream = _random((2, 3, VOCAB), 2)
        with kernels.policy("off"):
            expected = _logits_and_gradient(call, x, upstream, True)
        with kernels.force_route("reference_wrapped"):
            actual = _logits_and_gradient(call, x, upstream, True)
        for a, b in zip(actual, expected):
            self.assertAllEqual(_bits(a), _bits(b))

    @parameterized.named_parameters(
        named_product(
            tie_weights=(True, False),
            block_size=(64, -1),
            policy=("float32", "mixed_bfloat16"),
        )
    )
    def test_route_replaces_the_forward_only(
        self, tie_weights, block_size, policy
    ):
        """An allow-list row runs its route in its M bucket. The logits
        are the route's, and the input gradient is the reference's, within
        `_gradient_bound`."""
        layer = _reversible(tie_weights, block_size, policy)
        call = functools.partial(layer, reverse=True)
        x = _random((2, 3, DIM), 1)
        upstream = _random((2, 3, VOCAB), 2)
        with kernels.policy("off"):
            expected_logits, expected_grad = _logits_and_gradient(
                call, x, upstream
            )
        rows = {_row_key(layer.compute_dtype, "1-16"): (("matrix", None),)}
        matrix = kernels.RouteSource(_matrix_logits)
        # The allow-list applies under `"auto"` only.
        with kernels.policy("auto"), _lookup_routes(rows, matrix=matrix):
            self.assertEqual(_counts(layer, x), {"matrix": 1})
            self.assertEqual(
                _counts(layer, np.ones((17, DIM))), {"reference": 1}
            )
            logits, grad = _logits_and_gradient(call, x, upstream)
            stored, orientation = _stored(layer)
            with AutocastScope(layer.compute_dtype):
                stored = stored.with_tensors(stored.read_tensors())
                inputs = ops.cast(x, layer.compute_dtype)
                route = kernels.select_lookup_route(
                    layer, stored, orientation, inputs
                )
                direct = kernels.run_lookup_route(
                    route, layer, stored, orientation, inputs
                )
        self.assertAllEqual(_bits(logits), _bits(direct))
        # The route dequantizes in the compute dtype: the max-norm error
        # bound of `kernels_test.py`.
        bound = 1e-5 if policy == "float32" else 2**-7
        expected_logits = _numpy(expected_logits)
        self.assertLessEqual(
            np.max(np.abs(_numpy(logits) - expected_logits)),
            bound * np.max(np.abs(expected_logits)),
        )
        if expected_grad is not None:
            bound = _gradient_bound(layer, upstream, expected_grad)
            self.assertAllClose(grad, expected_grad, atol=bound, rtol=0)

    def test_first_route_whose_gates_pass(self):
        """A row runs its first reverse lookup route whose gates pass.

        `available` gets `M`, `D` and `V`.
        """
        layer = _reversible(True, 64)
        row = (
            ("bfloat16_only", None),
            ("projection_only", None),
            ("unavailable", None),
            ("matrix", None),
        )
        with _lookup_routes(
            {_row_key("float32", "1-16"): row},
            bfloat16_only=kernels.RouteSource(
                _matrix_logits, dtypes=("bfloat16",)
            ),
            unavailable=kernels.RouteSource(
                _matrix_logits, available=lambda m, k, n, view: False
            ),
            matrix=kernels.RouteSource(
                _matrix_logits,
                available=lambda m, k, n, view: (m, k, n) == (2, DIM, VOCAB),
            ),
        ):
            with kernels.policy("auto"):
                counts = _counts(layer, _random((2, DIM), 1))
        self.assertEqual(counts, {"matrix": 1})

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="A one-device `DataParallel`."
    )
    def test_distribution_runs_the_reference_path(self):
        layer = _reversible(True, 64)
        devices = distribution.list_devices("cpu")[:1]
        data_parallel = distribution.DataParallel(devices=devices)
        rows = {_row_key("float32", "1-16"): (("matrix", None),)}
        matrix = kernels.RouteSource(_matrix_logits)
        with (
            kernels.policy("auto"),
            _lookup_routes(rows, matrix=matrix),
            data_parallel.scope(),
        ):
            counts = _counts(layer, _random((2, DIM), 1))
        self.assertEqual(counts, {"reference": 1})

    @parameterized.named_parameters(
        named_product(
            tie_weights=(True, False),
            block_size=(64, -1),
            policy=("float32", "mixed_bfloat16"),
        )
    )
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Reads the JAX residuals."
    )
    def test_residuals_hold_the_stored_tensors(
        self, tie_weights, block_size, policy
    ):
        layer = _reversible(tie_weights, block_size, policy)
        x = _random((2, 3, DIM), 1)

        def residuals(scope):
            with scope:
                _, vjp = jax.vjp(lambda v: layer(v, reverse=True), x)
            return [
                leaf
                for leaf in jax.tree_util.tree_leaves(vjp)
                if hasattr(leaf, "dtype")
            ]

        today = residuals(kernels.policy("off"))
        wrapped = residuals(kernels.force_route("reference_wrapped"))
        stored, _ = _stored(layer)
        self.assertIn(
            (tuple(stored.codes.shape), "int8"),
            [(leaf.shape, standardize_dtype(leaf.dtype)) for leaf in wrapped],
        )
        # Today keeps the `(D, V)` table, dequantized or unpacked.
        self.assertTrue(any(leaf.size == VOCAB * DIM for leaf in today))
        self.assertTrue(all(leaf.size < VOCAB * DIM for leaf in wrapped))
        self.assertLess(
            sum(leaf.nbytes for leaf in wrapped),
            sum(leaf.nbytes for leaf in today),
        )

    @parameterized.named_parameters(
        (
            "int8_weight_only",
            "int8",
            Int8QuantizationConfig(activation_quantizer=None),
            (2, DIM),
        ),
        ("empty_inputs", "int4", None, (0, DIM)),
    )
    def test_refusal_runs_the_reference(self, mode, config, input_shape):
        layer = _reversible(True, 64, config=config, mode=mode)
        x = _random(input_shape, 1)
        with kernels.policy("off"):
            expected = layer(x, reverse=True)
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(layer(x, reverse=True), expected)

    @parameterized.named_parameters(
        ("int8", "int8", None),
        (
            "int4_activation_quantizer",
            "int4",
            Int4QuantizationConfig(
                block_size=-1, activation_quantizer=AbsMaxQuantizer()
            ),
        ),
    )
    def test_activation_quantizer_keeps_its_path(self, mode, config):
        layer = _reversible(True, -1, config=config, mode=mode)
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(_counts(layer, _random((2, DIM), 1)), {})

    @parameterized.named_parameters(
        ("tied", True),
        ("untied", False),
    )
    def test_auto_without_rows_is_the_reference_path(self, tie_weights):
        layer = _reversible(tie_weights, 64)
        x = _random((2, DIM), 1)
        with kernels.policy("off"):
            expected = layer(x, reverse=True)
        with kernels.policy("auto"):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(layer(x, reverse=True), expected)

    def test_rows_of_this_platform(self):
        """Under `"auto"`, each reverse lookup row of this backend and
        device runs its first route that this process has in its tested
        version range, for a tied and an untied table, and the logits
        equal a direct call of the route (G8). With no such route, the
        reference path runs. The table passes the shape gates of every
        route: `D = V = 256` and groups of 128."""
        platform_key = _row_key("float32", "1-16")[:2]
        rows = {
            key: routes
            for key, routes in kernels._LOOKUP_ALLOW_LIST.items()
            if key[:2] == platform_key
        }
        if not rows:
            self.skipTest("The allow-list has no row for this platform.")
        policy = {"bfloat16": "mixed_bfloat16", "float16": "mixed_float16"}
        m = {"1-16": 4, "17-64": 33, "65-256": 130, "257+": 300}
        rng = np.random.default_rng(0)
        for ((_, _, dtype, bucket), routes), tie_weights in itertools.product(
            rows.items(), (True, False)
        ):
            with self.subTest(dtype=dtype, bucket=bucket, tied=tie_weights):
                layer = layers.ReversibleEmbedding(
                    256,
                    256,
                    tie_weights=tie_weights,
                    dtype=policy.get(dtype, dtype),
                )
                layer.build()
                layer._embeddings.assign(rng.normal(size=(256, 256)))
                if not tie_weights:
                    layer.reverse_embeddings.assign(rng.normal(size=(256, 256)))
                config = Int4QuantizationConfig(block_size=128)
                layer.quantize("int4", config=config)
                x = _random((m[bucket], 256), 1)
                expected = _first_tested_lookup_route(routes)
                # Only `"auto"` reads the allow-list.
                with kernels.policy("auto"):
                    self.assertEqual(_counts(layer, x), {expected: 1})
                    logits = layer(x, reverse=True)
                    if expected != "reference":
                        stored, orientation = _stored(layer)
                        with AutocastScope(layer.compute_dtype):
                            stored = stored.with_tensors(stored.read_tensors())
                            inputs = ops.cast(x, layer.compute_dtype)
                            route = kernels.select_lookup_route(
                                layer, stored, orientation, inputs
                            )
                            direct = kernels.run_lookup_route(
                                route, layer, stored, orientation, inputs
                            )
                        self.assertAllEqual(_bits(logits), _bits(direct))
                if expected == "reference":
                    with kernels.policy("off"):
                        expected_logits = layer(x, reverse=True)
                    self.assertAllEqual(_bits(logits), _bits(expected_logits))

    @pytest.mark.skipif(
        backend.backend() != "numpy",
        reason="Only NumPy warns on a custom gradient.",
    )
    def test_numpy_reference_path_does_not_warn(self):
        layer = _reversible(True, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            layer(_random((2, DIM), 1), reverse=True)
