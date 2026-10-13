import numpy as np
from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.geometry import LookupGeometry
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.calibration import GPTQStrategy
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantization_test_utils import Pointwise1D


class StrategyRegistryTest(testing.TestCase):
    def test_builtin_modes_match_public_tuple(self):
        # The registration order is observable (validation error messages
        # render the registered-names tuple), so it must stay identical to
        # the public QUANTIZATION_MODES constant.
        self.assertEqual(
            strategy_registry.registered_modes(), QUANTIZATION_MODES
        )
        for name in QUANTIZATION_MODES:
            self.assertIsNotNone(strategy_registry.get_strategy(name))

    def test_unknown_mode(self):
        self.assertIsNone(strategy_registry.get_strategy("bogus"))
        self.assertFalse(strategy_registry.is_registered("bogus"))

    def test_register_requires_name(self):
        class Nameless(strategy_registry.QuantizationStrategy):
            pass

        with self.assertRaisesRegex(ValueError, "non-empty string `name`"):
            strategy_registry.register_quantization_strategy(Nameless)

    def test_register_rejects_duplicates(self):
        class Duplicate(strategy_registry.QuantizationStrategy):
            name = "int8"

        with self.assertRaisesRegex(ValueError, "already registered"):
            strategy_registry.register_quantization_strategy(Duplicate)

    @parameterized.named_parameters(
        ("existing_builtin_is_prefix", "int42"),
        ("new_is_prefix_of_builtin", "in"),
    )
    def test_register_rejects_builtin_prefix_collisions(self, name):
        # Built-in mode names are routed by `str.startswith` over policy
        # strings, so no mode name may share a prefix with a built-in.
        colliding_name = name

        class Colliding(strategy_registry.QuantizationStrategy):
            name = colliding_name

        with self.assertRaisesRegex(ValueError, "collides"):
            strategy_registry.register_quantization_strategy(Colliding)

    def test_register_allows_custom_prefix_overlap(self):
        # Externally registered modes match only their exact grammar
        # (name, name + "/", name + "_from_"), so two custom modes may
        # share a prefix without ambiguity.
        class Custom(ToyHalfStrategy):
            name = "custom"

        class CustomTwo(ToyHalfStrategy):
            name = "custom2"

        strategy_registry.register_quantization_strategy(Custom)
        try:
            strategy_registry.register_quantization_strategy(CustomTwo)
            policy = dtype_policies.get("custom2_from_float32")
            self.assertEqual(policy.quantization_mode, "custom2")
        finally:
            strategy_registry.unregister_quantization_strategy("custom")
            strategy_registry.unregister_quantization_strategy("custom2")

    @parameterized.named_parameters(
        ("slash", "my/mode", "must not contain"),
        ("from_separator", "my_from_mode", "must not contain"),
        ("standard_dtype", "float32", "conflicts with a standard dtype"),
        ("mixed_policy", "mixed_custom", "conflicts with a standard dtype"),
    )
    def test_register_rejects_reserved_names(self, name, error):
        # Names containing the policy-grammar separators or shadowing a
        # standard dtype / mixed-precision policy would break ordinary
        # policy-string parsing.
        reserved_name = name

        class Reserved(strategy_registry.QuantizationStrategy):
            name = reserved_name

        with self.assertRaisesRegex(ValueError, error):
            strategy_registry.register_quantization_strategy(Reserved)

    def test_registered_name_does_not_capture_ordinary_policies(self):
        # Policy strings are routed by mode name, but only through the
        # quantized grammar (bare name, name + "/", name + "_from_"). A
        # registered mode whose name prefixes ordinary policy strings (like
        # "mixed" prefixing "mixed_bfloat16") must not hijack them.
        class MixedMode(ToyHalfStrategy):
            name = "mixed"

        strategy_registry.register_quantization_strategy(MixedMode)
        try:
            policy = dtype_policies.get("mixed_bfloat16")
            self.assertIsNone(policy.quantization_mode)
            self.assertEqual(policy.compute_dtype, "bfloat16")
        finally:
            strategy_registry.unregister_quantization_strategy("mixed")

    def test_register_as_decorator_keeps_the_class(self):
        # Registering returns its argument, so a decorated strategy stays
        # a class and can still be subclassed.
        @strategy_registry.register_quantization_strategy
        class Decorated(ToyHalfStrategy):
            name = "decorated"

        try:
            self.assertIsInstance(Decorated, type)
            self.assertIsNotNone(strategy_registry.get_strategy("decorated"))

            class Sub(Decorated):
                name = "decorated_sub"

            self.assertIsInstance(Sub, type)
        finally:
            strategy_registry.unregister_quantization_strategy("decorated")

    @parameterized.named_parameters(
        ("config_cls", {"config_cls": None}, "`config_cls`"),
        ("geometry_families", {"geometry_families": ()}, "`geometry_families`"),
        (
            "build",
            {"build": strategy_registry.QuantizationStrategy.build},
            "`build`",
        ),
        (
            "call",
            {"call": strategy_registry.QuantizationStrategy.call},
            "`call`",
        ),
        (
            "quantize",
            {"quantize": strategy_registry.QuantizationStrategy.quantize},
            "`quantize`",
        ),
    )
    def test_register_refuses_an_incomplete_strategy(self, attributes, missing):
        incomplete = type(
            "Incomplete",
            (ToyHalfStrategy,),
            {"name": "incomplete", **attributes},
        )
        self.addCleanup(
            strategy_registry.unregister_quantization_strategy, "incomplete"
        )
        with self.assertRaisesRegex(ValueError, f"does not define {missing}"):
            strategy_registry.register_quantization_strategy(incomplete)
        self.assertFalse(strategy_registry.is_registered("incomplete"))

    @parameterized.named_parameters(
        ("policy_cls", "policy_cls"), ("calibrator_cls", "calibrator_cls")
    )
    def test_register_refuses_a_calibration_mode_without(self, attribute):
        incomplete = type(
            "Incomplete",
            (GPTQStrategy,),
            {"name": "incomplete", attribute: None},
        )
        self.addCleanup(
            strategy_registry.unregister_quantization_strategy, "incomplete"
        )
        with self.assertRaisesRegex(
            ValueError, f"does not define `{attribute}`"
        ):
            strategy_registry.register_quantization_strategy(incomplete)
        self.assertFalse(strategy_registry.is_registered("incomplete"))

    def test_register_refuses_a_lora_mode_with_a_view_and_no_encode(self):
        class ViewNoEncode(ToyHalfStrategy):
            name = "view_no_encode"

            def quantized_weight(self, layer):
                return None

        self.addCleanup(
            strategy_registry.unregister_quantization_strategy,
            "view_no_encode",
        )
        with self.assertRaisesRegex(ValueError, "does not define `encode`"):
            strategy_registry.register_quantization_strategy(ViewNoEncode)
        self.assertFalse(strategy_registry.is_registered("view_no_encode"))

        # Without LoRA, the save never merges, so `encode` is not needed.
        class ViewNoLoRA(ViewNoEncode):
            supports_lora = False

        strategy_registry.register_quantization_strategy(ViewNoLoRA)
        self.assertTrue(strategy_registry.is_registered("view_no_encode"))


class PolicyCodecCorpusTest(testing.TestCase):
    """Every historical policy-string form parses and round-trips."""

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", "int8_from_float32", "int8", {}),
        (
            "int8_mixed",
            "int8_from_mixed_bfloat16",
            "int8_from_mixed_bfloat16",
            "int8",
            {},
        ),
        (
            "int4_legacy_bare",
            "int4_from_float32",
            "int4_from_float32",
            "int4",
            {},
        ),
        (
            "int4_grouped",
            "int4/128_from_float32",
            "int4/128_from_float32",
            "int4",
            {"block_size": 128},
        ),
        (
            "int4_per_channel",
            "int4/-1_from_float32",
            "int4/-1_from_float32",
            "int4",
            {"block_size": -1},
        ),
        (
            "int4_legacy_none_block",
            "int4/None_from_float32",
            "int4/-1_from_float32",
            "int4",
            {"block_size": -1},
        ),
        (
            "float8",
            "float8_from_float32",
            "float8_from_float32",
            "float8",
            {},
        ),
        (
            "ternary",
            "ternary_from_float32",
            "ternary_from_float32",
            "ternary",
            {},
        ),
        (
            "gptq",
            "gptq/4/128_from_float32",
            "gptq/4/128_from_float32",
            "gptq",
            {"weight_bits": 4, "group_size": 128},
        ),
        (
            "gptq_whole_tensor",
            "gptq/2/-1_from_bfloat16",
            "gptq/2/-1_from_bfloat16",
            "gptq",
            {"weight_bits": 2, "group_size": -1},
        ),
        (
            "gptq_mixed",
            "gptq/8/32_from_mixed_bfloat16",
            "gptq/8/32_from_mixed_bfloat16",
            "gptq",
            {"weight_bits": 8, "group_size": 32},
        ),
        (
            "awq",
            "awq/4/128_from_float32",
            "awq/4/128_from_float32",
            "awq",
            {"weight_bits": 4, "group_size": 128},
        ),
        (
            "awq_per_channel",
            "awq/4/-1_from_float32",
            "awq/4/-1_from_float32",
            "awq",
            {"weight_bits": 4, "group_size": -1},
        ),
        (
            "gptq_corrupted_source",
            "gptq/4/128_from_None",
            "gptq/4/128_from_float32",
            "gptq",
            {"weight_bits": 4, "group_size": 128},
        ),
    )
    def test_policy_string_corpus(
        self, policy_str, expected_name, expected_mode, expected_params
    ):
        policy = dtype_policies.get(policy_str)
        self.assertEqual(policy.name, expected_name)
        self.assertEqual(policy.quantization_mode, expected_mode)
        for attr, value in expected_params.items():
            self.assertEqual(getattr(policy, attr), value)
        # Serialization round-trip preserves the resolved policy.
        revived = dtype_policies.deserialize(dtype_policies.serialize(policy))
        self.assertEqual(revived.name, expected_name)
        for attr, value in expected_params.items():
            self.assertEqual(getattr(revived, attr), value)

    @parameterized.named_parameters(
        ("no_source", "int8"),
        ("int4_zero_block", "int4/0_from_float32"),
        ("int4_garbage_block", "int4/abc_from_float32"),
        ("gptq_bad_bits", "gptq/5/128_from_float32"),
        ("gptq_missing_group", "gptq/4_from_float32"),
        ("awq_bad_bits", "awq/8/128_from_float32"),
        ("unknown_mode", "int7_from_float32"),
    )
    def test_invalid_policy_strings(self, policy_str):
        with self.assertRaises(ValueError):
            dtype_policies.get(policy_str)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", "QuantizedDTypePolicy"),
        ("int4_legacy_bare", "int4_from_float32", "QuantizedDTypePolicy"),
        ("int4_grouped", "int4/128_from_float32", "Int4DTypePolicy"),
        ("float8", "float8_from_float32", "QuantizedFloat8DTypePolicy"),
        ("ternary", "ternary_from_float32", "QuantizedDTypePolicy"),
        ("gptq", "gptq/4/128_from_float32", "GPTQDTypePolicy"),
        ("awq", "awq/4/128_from_float32", "AWQDTypePolicy"),
    )
    def test_policy_string_class(self, policy_str, class_name):
        policy = dtype_policies.get(policy_str)
        self.assertEqual(type(policy).__name__, class_name)
        self.assertEqual(
            dtype_policies.serialize(policy)["class_name"], class_name
        )


@saving.register_keras_serializable(package="strategy_registry_test")
class ToyModeConfig(QuantizationConfig):
    """Config for the toy float16-storage mode used in the test below."""

    def __init__(self):
        super().__init__(None, None)

    @property
    def mode(self):
        return "demo_half"

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


class ToyHalfStrategy(strategy_registry.QuantizationStrategy):
    """A toy quantization mode: store the kernel in float16.

    The strategy implements build/call/quantize directly against the
    layer's geometry (`_quantization_geometry()`); `ToyDense` declares the
    mode in its `variable_serialization_spec`.
    """

    name = "demo_half"
    config_cls = ToyModeConfig
    geometry_families = ("projection",)

    def build(self, layer, input_shape, config):
        del config
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=input_shape,
            initializer="zeros",
            dtype="float16",
            trainable=False,
        )

    def quantize(self, layer, config):
        kernel_shape = layer._quantization_geometry().weight_shape
        kernel_value = ops.cast(layer._kernel, "float16")
        del layer._kernel
        layer.quantized_build(kernel_shape, self.name, config)
        layer._kernel.assign(kernel_value)

    def call(self, layer, inputs, training=None):
        x = ops.matmul(inputs, ops.cast(layer._kernel, layer.compute_dtype))
        if layer.bias is not None:
            x = ops.add(x, layer.bias)
        if layer.activation is not None:
            x = layer.activation(x)
        return x


@saving.register_keras_serializable(package="strategy_registry_test")
class ToyDense(layers.Dense):
    """A `Dense` that lists the toy mode among the modes it supports."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        spec["demo_half"] = ["kernel", "bias"]
        return spec

    def _quantization_geometry(self):
        return ProjectionGeometry(self)


class ToyModeRegistrationTest(testing.TestCase):
    """Registry dispatch, through a toy strategy and a spec entry.

    The registration mechanism accepts the toy strategy, but Keras neither
    supports nor documents modes beyond the built-in ones.
    """

    def setUp(self):
        super().setUp()
        strategy_registry.register_quantization_strategy(ToyHalfStrategy)

    def tearDown(self):
        strategy_registry.unregister_quantization_strategy("demo_half")
        super().tearDown()

    def test_toy_mode_end_to_end(self):
        layer = ToyDense(units=3)
        layer.build((None, 4))
        reference_kernel = ops.convert_to_numpy(layer._kernel)

        layer.quantize("demo_half")

        # The kernel is now stored in float16 and the policy is named after
        # the mode, all through the generic machinery.
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "float16")
        self.assertEqual(layer.dtype_policy.name, "demo_half_from_float32")
        self.assertEqual(layer.quantization_mode, "demo_half")
        self.assertTrue(layer._is_quantized)

        # The quantized forward pass dispatches through the strategy.
        x = np.random.uniform(-1, 1, size=(2, 4)).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        expected = x @ reference_kernel.astype("float16").astype(
            "float32"
        ) + ops.convert_to_numpy(layer.bias)
        self.assertAllClose(
            y, expected, atol=1e-3, tpu_atol=1e-2, tpu_rtol=5e-2
        )

    def test_toy_mode_through_model_quantize(self):
        model = models.Sequential(
            [layers.Input((4,)), ToyDense(3, name="target")]
        )
        report = model.quantize("demo_half", verbose=False)
        self.assertEqual(
            model.get_layer("target").dtype_policy.name,
            "demo_half_from_float32",
        )
        self.assertIn(
            "target", "".join(path for path, _, _ in report.quantized)
        )

    def test_toy_mode_saves_and_loads(self):
        model = models.Sequential(
            [layers.Input((4,)), ToyDense(3, name="target")]
        )
        model.quantize("demo_half", verbose=False)
        x = np.random.uniform(-1, 1, size=(2, 4)).astype("float32")
        path = self.get_temp_dir() + "/toy.keras"
        model.save(path)
        revived = saving.load_model(path)
        self.assertEqual(
            revived.get_layer("target").dtype_policy.name,
            "demo_half_from_float32",
        )
        self.assertAllClose(revived(x), model(x))

    def test_toy_mode_rejects_a_layer_that_does_not_list_it(self):
        for layer in (layers.Dense(3), layers.Embedding(5, 3)):
            layer.build((None, 4))
            with self.assertRaises(NotImplementedError):
                layer.quantize("demo_half")
            self.assertIsNone(layer.quantization_mode)

    def test_toy_mode_compute_dtype_rule_reaches_layer_and_policy(self):
        class NoBFloat16(ToyHalfStrategy):
            def check_compute_dtype(self, compute_dtype):
                if compute_dtype == "bfloat16":
                    raise NotImplementedError("demo_half refuses bfloat16")

        strategy_registry.unregister_quantization_strategy("demo_half")
        strategy_registry.register_quantization_strategy(NoBFloat16)

        layer = ToyDense(3, dtype="bfloat16")
        layer.build((None, 4))
        with self.assertRaisesRegex(
            NotImplementedError, "^demo_half refuses bfloat16$"
        ):
            layer.quantize("demo_half")
        self.assertIsNone(layer.quantization_mode)
        self.assertEqual(standardize_dtype(layer._kernel.dtype), "bfloat16")
        with self.assertRaisesRegex(ValueError, "^demo_half refuses bfloat16$"):
            dtype_policies.get("demo_half_from_mixed_bfloat16")

        # Another compute dtype still quantizes.
        layer = ToyDense(3)
        layer.build((None, 4))
        layer.quantize("demo_half")
        self.assertEqual(layer.quantization_mode, "demo_half")


class ToyNoneConfig(ToyModeConfig):
    @property
    def mode(self):
        return "demo_none"


class ToyUnsupportedStrategy(ToyHalfStrategy):
    """A registered mode that no layer lists."""

    name = "demo_none"
    config_cls = ToyNoneConfig


class QuantizeTransactionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        strategy_registry.register_quantization_strategy(ToyUnsupportedStrategy)

    def tearDown(self):
        strategy_registry.unregister_quantization_strategy("demo_none")
        super().tearDown()

    def test_unsupported_mode_leaves_layer_untouched(self):
        # An unsupported (layer, mode) pair must be rejected before any
        # state is mutated, so `Model.quantize` records the layer as
        # skipped and the layer stays fully usable and quantizable.
        layer = layers.Dense(4)
        layer.build((None, 8))
        with self.assertRaises(NotImplementedError):
            layer.quantize("demo_none", config=ToyNoneConfig())
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(getattr(layer, "_is_quantized", False))
        self.assertIsNone(layer.quantization_mode)
        # The layer is still float and still quantizable.
        layer.quantize("int8")
        self.assertEqual(layer.quantization_mode, "int8")


class ListsEveryModeEmbedding(layers.Embedding):
    """An `Embedding` that lists modes a lookup geometry cannot carry."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        for mode in ("float8", "ternary", "gptq", "awq"):
            spec[mode] = ["embeddings"]
        return spec

    def _quantization_geometry(self):
        return LookupGeometry(self)


class ListsTernaryEinsumDense(layers.EinsumDense):
    """An `EinsumDense` that lists the ternary mode."""

    @property
    def variable_serialization_spec(self):
        spec = super().variable_serialization_spec
        spec["ternary"] = ["kernel", "bias", "kernel_scale"]
        return spec

    def _quantization_geometry(self):
        return super()._quantization_geometry()


class ProjectionOnlyModeTest(testing.TestCase):
    """A mode written for projections refuses a lookup layer that lists it."""

    @parameterized.named_parameters(
        ("float8", "float8"),
        ("ternary", "ternary"),
        ("gptq", "gptq"),
        ("awq", "awq"),
    )
    def test_listing_lookup_layer_is_refused(self, mode):
        config = None
        if mode == "gptq":
            config = GPTQConfig(dataset=None, tokenizer=None)
        elif mode == "awq":
            config = AWQConfig(dataset=None, tokenizer=None)
        layer = ListsEveryModeEmbedding(10, 8)
        layer.build()
        with self.assertRaisesRegex(
            NotImplementedError, "'lookup' quantization geometry"
        ):
            layer.quantize(mode, config=config)
        # Refused before the layer changes.
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(layer._is_quantized)

    def test_ternary_refuses_a_listing_einsum_layer(self):
        layer = ListsTernaryEinsumDense("ab,bc->ac", output_shape=4)
        layer.build((None, 3))
        with self.assertRaisesRegex(
            NotImplementedError, "only a `Dense` kernel"
        ):
            layer.quantize("ternary")
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)

    def test_float8_build_refuses_a_listing_lookup_layer(self):
        # A layer built from a float8 policy, as on load.
        layer = ListsEveryModeEmbedding(10, 8, dtype="float8_from_float32")
        with self.assertRaisesRegex(
            NotImplementedError, "'lookup' quantization geometry"
        ):
            layer.build()

    def test_model_quantize_skips_listing_lookup_layer(self):
        model = models.Sequential(
            [
                layers.Input((3,), dtype="int32"),
                ListsEveryModeEmbedding(10, 8, name="emb"),
                layers.Dense(4, name="proj"),
            ]
        )
        model.quantize("float8")
        self.assertIsNone(model.get_layer("emb").quantization_mode)
        self.assertEqual(model.get_layer("proj").quantization_mode, "float8")
        model(np.array([[1, 2, 3]]))


RECIPE_PROJECTION_SPEC = {
    None: ["kernel"],
    "int8": ["kernel", "kernel_scale"],
    "int4": ["kernel", "kernel_scale", "kernel_zero", "g_idx"],
    "float8": [
        "kernel",
        "inputs_scale",
        "inputs_amax_history",
        "kernel_scale",
        "kernel_amax_history",
        "outputs_grad_scale",
        "outputs_grad_amax_history",
    ],
    "ternary": ["kernel", "kernel_scale"],
}


class RecipeProjection(layers.Layer):
    """A third-party 2-D projection written per the `geometry.py` recipe.

    It has no `units` attribute, and its `bias` and `activation` are `None`.
    """

    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.activation = None

    def build(self, input_shape):
        self.kernel_shape = (input_shape[-1], self.features)
        if self.quantization_mode:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if not self._strategy_owns_weight_storage():
            self._kernel = self.add_weight(
                name="kernel", shape=self.kernel_shape
            )
        self.bias = None

    @property
    def kernel(self):
        quantized_weight = self._quantized_weight()
        if quantized_weight is None:
            return self._kernel
        return quantized_weight.unpack()

    def call(self, inputs):
        return ops.matmul(inputs, self._kernel)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        return RECIPE_PROJECTION_SPEC

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")


class WithoutActivation(RecipeProjection):
    """A `RecipeProjection` that does not set `activation`.

    It defines `_quantization_geometry()`, so the type check of `quantize()`
    accepts it.
    """

    def __init__(self, **kwargs):
        super().__init__(4, **kwargs)
        del self.activation

    def _quantization_geometry(self):
        return ProjectionGeometry(self)


class CustomProjectionTest(testing.TestCase):
    """Third-party projections stay quantizable with the built-in modes."""

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
        ("float8", "float8"),
        ("ternary", "ternary"),
    )
    def test_quantizes_saves_and_loads(self, mode):
        layer = RecipeProjection(4)
        layer.build((None, 6))
        x = np.random.uniform(-1, 1, size=(3, 6)).astype("float32")
        layer.quantize(mode)
        self.assertEqual(layer.quantization_mode, mode)
        y = layer(x)
        store = {}
        layer.save_own_variables(store)
        restored = RecipeProjection(4, dtype=layer.dtype_policy.name)
        restored.build((None, 6))
        restored.load_own_variables(store)
        self.assertAllClose(restored(x), y)

    def test_ternary_refuses_a_kernel_that_is_not_2d(self):
        layer = Pointwise1D(4)
        layer.build((None, 5, 6))
        kernel = ops.convert_to_numpy(layer._kernel)
        with self.assertRaisesRegex(NotImplementedError, "only a 2-D kernel"):
            layer.quantize("ternary")
        # Refused before the layer changes.
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(layer._is_quantized)
        self.assertAllEqual(layer._kernel, kernel)

    def test_ternary_build_refuses_a_kernel_that_is_not_2d(self):
        # A layer built from a ternary policy, as on load.
        layer = Pointwise1D(4, dtype="ternary_from_float32")
        with self.assertRaisesRegex(NotImplementedError, "only a 2-D kernel"):
            layer.build((None, 5, 6))

    def test_model_quantize_skips_a_kernel_that_is_not_2d(self):
        model = models.Sequential(
            [
                layers.Input((5, 6)),
                Pointwise1D(4, name="pointwise"),
                layers.Dense(3, name="dense"),
            ]
        )
        pointwise = model.get_layer("pointwise")
        dense = model.get_layer("dense")
        report = model.quantize("ternary", verbose=False)
        self.assertIn((pointwise.path, report.SKIP_NO_SUPPORT), report.skipped)
        self.assertIn(dense.path, [path for path, *_ in report.quantized])
        self.assertIsNone(pointwise.quantization_mode)
        self.assertEqual(dense.quantization_mode, "ternary")
        model(np.ones((2, 5, 6), "float32"))

    def test_missing_build_attributes_are_refused_before_the_change(self):
        layer = WithoutActivation()
        layer.build((None, 6))
        with self.assertRaisesRegex(ValueError, "did not set activation"):
            layer.quantize("int8")
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(layer._is_quantized)
        self.assertEqual(tuple(layer._kernel.shape), (6, 4))

    def test_model_quantize_raises_on_missing_build_attributes(self):
        model = models.Sequential([layers.Input((6,)), WithoutActivation()])
        with self.assertRaisesRegex(ValueError, "did not set activation"):
            model.quantize("int8", verbose=False)
