import math
import os
import warnings
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import activations
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.calibration_run import CalibrationRun
from keras.src.quantizers.calibration_run import _execution_stages
from keras.src.quantizers.calibration_run import get_dataloader
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import calibration_statistic
from keras.src.quantizers.quantization_test_utils import tiny_calibration_model
from keras.src.quantizers.quantization_test_utils import token_dataset
from keras.src.quantizers.report import QuantizationReport
from keras.src.utils.rng_utils import set_random_seed

VOCAB_SIZE = 100


class MockTokenizer:
    """A mock tokenizer that mimics the real API for testing."""

    def tokenize(self, text):
        return [ord(c) % VOCAB_SIZE for c in "".join(text)]

    def __call__(self, text):
        return self.tokenize(text)


class EmptyBlock(layers.Layer):
    """A block that contains no quantizable layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()

    def call(self, inputs):
        return self.ln(inputs)


class TupleBlock(layers.Layer):
    """A block that returns its hidden states and a second output."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units)

    def call(self, inputs):
        hidden = self.dense(inputs)
        return hidden, ops.sum(hidden)


class TransformerBlock(layers.Layer):
    """A toy transformer block with a quantizable Dense layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(128)

    def call(self, inputs):
        return self.dense(inputs)


def build_all_tokens_strings(dataset, tokenizer):
    pieces = [
        np.asarray(tokenizer.tokenize(s), dtype=np.int32).reshape(-1)
        for s in dataset
    ]
    return np.concatenate(pieces, axis=0).astype(np.int32, copy=False)


@pytest.mark.requires_trainable_backend
class TestCalibrationCore(testing.TestCase):
    def test_shape_and_dtype_strings(self):
        """Test the shape and dtype of the output for string inputs."""
        tok = MockTokenizer()
        dataset = ["a b c d e f g", "h i j k"]
        seq_len, n = 5, 7

        out = get_dataloader(tok, seq_len, dataset, num_samples=n)
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_shape_and_dtype_pretokenized(self):
        """Test the shape and dtype of the output for pre-tokenized inputs."""
        tok = MockTokenizer()
        # Pre-tokenized inputs; mixed shapes (1, L) and (L,)
        seqs = [
            np.array([[1, 2, 3, 4]], dtype=np.int64),
            np.array([5, 6], dtype=np.int64),
        ]
        tok = MockTokenizer()
        seq_len, n = 3, 4

        out = get_dataloader(tok, seq_len, seqs, num_samples=n)
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_strided_is_deterministic_for_same_args(self):
        tok = MockTokenizer()
        dataset = ["a b c d e", "f g h i j k"]
        out1 = get_dataloader(tok, 4, dataset, num_samples=6)
        out2 = get_dataloader(tok, 4, dataset, num_samples=6)
        self.assertTrue(ops.all(ops.equal(out1, out2)))

    def test_windows_are_contiguous_runs_of_the_stream(self):
        tok = MockTokenizer()
        dataset = [" ".join([f"t{i}" for i in range(20)])]
        seq_len, n = 4, 5

        out = get_dataloader(tok, seq_len, dataset, num_samples=n)

        # Validate that each sample is a contiguous run
        # of length seq_len from the flattened stream
        flat = build_all_tokens_strings(dataset, tok)
        for s in out[:, 0, :]:
            # Each window should appear as a slice in the flat stream
            # (This is a soft check; exact start positions depend on offset.)
            joined = " ".join(map(str, s.tolist()))
            self.assertIn(joined, " ".join(map(str, flat.tolist())))

    def test_get_dataloader_error_scenarios(self):
        """Tests error cases for get_dataloader."""
        with pytest.raises(ValueError, match="Provided dataset is empty"):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset=[],
                num_samples=10,
            )
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be an iterable.*Got type: str.*"
            "Please pass the loaded dataset directly.",
        ):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset="wikitext2",
                num_samples=10,
            )

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_batching_matches_one_sample_at_a_time(self, mode):
        """The statistics are means over the observed rows, so batching the
        sweep changes only the order they accumulate in."""
        seq_len, d_model, num_samples = 8, 16, 8
        block = TransformerBlock()
        block(ops.zeros((1, seq_len, d_model)))
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((num_samples, seq_len, d_model))
        samples = ops.convert_to_tensor(samples.astype("float32"))

        def accumulate(batch_size):
            calibrator = calibrate_layer(
                block.dense, calibration_config(mode), solve=False
            )
            with calibration_scope({block.dense: calibrator.observe}):
                for start in range(0, num_samples, batch_size):
                    block(samples[start : start + batch_size])
            return calibration_statistic(calibrator), calibrator.num_samples

        statistic, rows = accumulate(batch_size=1)
        batched_statistic, batched_rows = accumulate(batch_size=4)
        self.assertEqual(rows, batched_rows)
        self.assertAllClose(statistic, batched_statistic, rtol=1e-5, atol=1e-5)

    def test_apply_gptq_on_multi_block_model(self):
        """Tests quantization on a model with multiple blocks."""
        model = models.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, 128),
                TransformerBlock(),
                TransformerBlock(),
            ]
        )
        model.build(input_shape=(None, 10))

        layer_structure = {
            "pre_block_layers": [model.layers[0]],
            "sequential_blocks": [model.layers[1], model.layers[2]],
        }

        config = GPTQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(),
            group_size=32,
            quantization_layer_structure=layer_structure,
        )
        model.quantize("gptq", config=config)


class TrainingOnlyBlock(layers.Layer):
    """A block whose second layer runs only in training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observed = layers.Dense(8, name="observed")
        self.training_only = layers.Dense(8, name="training_only")

    def build(self, input_shape):
        self.observed.build(input_shape)
        self.training_only.build(input_shape[:-1] + (8,))

    def call(self, inputs, training=False):
        hidden = self.observed(inputs)
        if training:
            hidden = self.training_only(hidden)
        return hidden


class RecipeOnlyProjection(layers.Layer):
    """A projection layer written to the documented recipe only.

    It is not a `Dense`: it defines a geometry, lists the modes in its
    `variable_serialization_spec`, and runs the quantized build sequence.
    It has none of the extras of `quantization_test_utils`'s third-party
    layers (LoRA, a `kernel` property, a config).
    """

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel_shape = (input_shape[-1], self.units)
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
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros"
        )

    def call(self, inputs):
        outputs = ops.add(ops.matmul(inputs, self._kernel), self.bias)
        return self.activation(outputs)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        return {
            None: ["kernel", "bias"],
            "gptq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "g_idx",
            ],
            "awq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "awq_scales",
                "g_idx",
            ],
        }

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")


class AttentionLikeBlock(layers.Layer):
    """Attention-shaped stages, a gated MLP, and a layer that never runs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query = layers.Dense(4)
        self.key = layers.Dense(4)
        self.value = layers.Dense(4)
        self.attention_output = layers.Dense(4)
        self.gate = layers.Dense(8)
        self.up = layers.Dense(8)
        self.down = layers.Dense(4)
        self.unused = layers.Dense(4)

    def call(self, inputs):
        scores = self.query(inputs) * self.key(inputs) + self.value(inputs)
        hidden = self.attention_output(scores)
        return self.down(self.gate(hidden) * self.up(hidden))


class TestExecutionStages(testing.TestCase):
    def test_stages_group_by_shared_input_and_order(self):
        block = AttentionLikeBlock()
        x = ops.ones((2, 4))
        block(x)
        stages = _execution_stages(
            block,
            [
                block.down,
                block.up,
                block.gate,
                block.attention_output,
                block.value,
                block.key,
                block.query,
            ],
            x,
        )
        self.assertEqual(
            stages,
            [
                [block.query, block.key, block.value],
                [block.attention_output],
                [block.gate, block.up],
                [block.down],
            ],
        )

    def test_untraced_layers_form_the_last_stage(self):
        block = AttentionLikeBlock()
        x = ops.ones((2, 4))
        block(x)
        stages = _execution_stages(
            block, [block.unused, block.attention_output, block.query], x
        )
        self.assertEqual(
            stages,
            [[block.query], [block.attention_output], [block.unused]],
        )


class TestDataloaderReproducibility(testing.TestCase):
    def test_strided_offset_is_process_stable(self):
        """The strided sampling offset must not depend on PYTHONHASHSEED.

        The offset was previously derived via `hash(("gptq-calib", seed))`;
        Python randomizes string hashing per process, so calibration
        windows - and therefore every quantization result - silently
        differed between runs despite the fixed seed. These golden values
        pin the numpy-based derivation (seed=42, 1000 tokens, seq len 8,
        4 samples); the old hash-based offset only reproduces them under
        one specific PYTHONHASHSEED by coincidence.
        """

        class PassthroughTokenizer:
            def tokenize(self, x):
                return np.asarray(x, dtype=np.int32)

        out = get_dataloader(
            PassthroughTokenizer(),
            8,
            [np.arange(1000, dtype=np.int32)],
            num_samples=4,
        )
        self.assertEqual(out.shape, (4, 1, 8))
        self.assertAllClose(out[:, 0, 0], np.array([88, 336, 584, 832]))
        self.assertAllClose(out[0, 0], np.arange(88, 96))


def _assert_calibrated(test, layer, mode):
    """`layer` holds `mode`'s codes, no float kernel and no config."""
    test.assertEqual(layer.quantization_mode, mode)
    test.assertFalse(hasattr(layer, "_kernel"))
    test.assertIsNone(layer.quantization_config)


def _assert_float(test, layer):
    """`layer` is float, with its kernel and no config."""
    test.assertIsNone(layer.quantization_mode)
    test.assertFalse(layer._is_quantized)
    test.assertTrue(hasattr(layer, "_kernel"))
    test.assertIsNone(layer.quantization_config)


def _tiny_model(mode, num_samples=4, dtype=None, block_layers=None, **kwargs):
    """Embedding -> block -> pooled head.

    The default block is `[Dense(16, relu), Dense(8)]`; given block layers
    map `(batch, 16, 8)` inputs to `(batch, 16, features)`.
    """
    set_random_seed(123)
    seq_len, vocab_size, embed_dim = 16, 48, 8
    if block_layers is None:
        block_layers = [
            layers.Dense(16, activation="relu", dtype=dtype),
            layers.Dense(embed_dim, dtype=dtype),
        ]
    model, structure = tiny_calibration_model(
        block_layers,
        vocab_size=vocab_size,
        sequence_length=seq_len,
        embed_dim=embed_dim,
        dtype=dtype,
    )
    if mode == "awq":
        kwargs["num_grid_points"] = 5
    rng = np.random.default_rng(7)
    config = calibration_config(
        mode,
        dataset=token_dataset(num_samples, seq_len, vocab_size, rng),
        tokenizer=lambda text: text,
        num_samples=num_samples,
        sequence_length=seq_len,
        group_size=8,
        quantization_layer_structure=structure,
        **kwargs,
    )
    return model, config


def _spy_on_observe(mode):
    """Patches the mode's `observe`; the mock records `(calibrator, x)`."""
    calibrator_cls = strategy_registry.get_strategy(mode).calibrator_cls
    return mock.patch.object(
        calibrator_cls,
        "observe",
        autospec=True,
        side_effect=calibrator_cls.observe,
    )


class CalibrationRunTest(testing.TestCase):
    """The driver the calibration modes share, once per mode."""

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_resolution_error_names_the_received_policy(self, mode):
        layer = layers.Dense(4)
        layer.build((None, 3))
        strategy = strategy_registry.get_strategy(mode)
        with self.assertRaisesRegex(
            ValueError,
            f"{mode.upper()} quantization.*Received: dtype_policy=<.*float32",
        ):
            strategy.resolve_group_size(layer, None)

    def test_requires_sequential_blocks(self):
        strategy = strategy_registry.get_strategy("gptq")
        with self.assertRaisesRegex(ValueError, "No sequential blocks"):
            CalibrationRun(
                strategy, calibration_config("gptq"), {"pre_block_layers": []}
            )

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_requires_a_dataset(self, mode):
        strategy = strategy_registry.get_strategy(mode)
        with self.assertRaisesRegex(
            ValueError, f"{mode.upper()} quantization requires a dataset"
        ):
            CalibrationRun(
                strategy, calibration_config(mode), {"sequential_blocks": [1]}
            )

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_calibrator_refuses_unsupported_layers(self, mode):
        ternary = layers.TernaryDense(4)
        ternary.build((None, 3))
        for layer in (layers.Layer(), ternary):
            with self.assertRaisesRegex(
                TypeError, f"Unsupported layer type for {mode.upper()}"
            ):
                calibrate_layer(layer, calibration_config(mode), solve=False)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_equation_without_a_view_is_refused_before_the_layer_changes(
        self, mode
    ):
        layer = layers.EinsumDense("abc,cd->ad", output_shape=(5,))
        layer.build((None, 3, 8))
        weights = [ops.convert_to_numpy(w) for w in layer.weights]
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot derive a contraction view"
        ):
            layer.quantize(mode, config=calibration_config(mode, group_size=-1))
        self.assertIsNone(layer.quantization_mode)
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(len(layer.weights), len(weights))
        for variable, value in zip(layer.weights, weights):
            self.assertAllEqual(variable, value)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_calibrates_every_block_in_order(self, mode):
        vocab_size, seq_len, embed_dim = 32, 8, 4
        embedding = layers.Embedding(vocab_size, embed_dim)
        blocks = [
            models.Sequential(
                [layers.Dense(embed_dim, activation="relu"), layers.Dense(4)]
            )
            for _ in range(2)
        ]
        inputs = layers.Input((seq_len,), dtype="int32")
        x = embedding(inputs)
        for block in blocks:
            x = block(x)
        model = models.Model(inputs, x)
        structure = {
            "pre_block_layers": [embedding],
            "sequential_blocks": blocks,
        }
        kwargs = dict(
            num_samples=3,
            sequence_length=seq_len,
            group_size=-1,
            calibration_batch_size=2,
        )
        if mode == "awq":
            kwargs["num_grid_points"] = 3
        config = calibration_config(mode, **kwargs)
        config.quantization_layer_structure = structure
        rng = np.random.default_rng(0)
        config.dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(4)
        ]
        config.tokenizer = lambda text: text

        strategy = strategy_registry.get_strategy(mode)
        run = CalibrationRun(strategy, config, structure)
        self.assertEqual(run.batch_size, 2)
        denses = [dense for block in blocks for dense in block.layers]
        for dense in denses:
            self.assertTrue(run.covers(dense))
            run.add(dense)
        self.assertFalse(run.covers(embedding))
        # Handing a layer over changes nothing on it.
        for dense in denses:
            self.assertIsNone(dense.quantization_mode)
        run.run()
        self.assertEqual(len(run.dataloader), 3)
        self.assertEqual(run.quantized, denses)
        for dense in denses:
            _assert_calibrated(self, dense, mode)
        outputs = ops.convert_to_numpy(model(config.dataset[0]))
        self.assertTrue(np.isfinite(outputs).all())

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_takes_the_first_output_of_a_block(self, mode):
        # A block that returns `(hidden, cache)` hands its first output on.
        vocab_size, seq_len, embed_dim = 32, 8, 4
        embedding = layers.Embedding(vocab_size, embed_dim)
        blocks = [TupleBlock(embed_dim) for _ in range(2)]
        structure = {
            "pre_block_layers": [embedding],
            "sequential_blocks": blocks,
        }
        config = calibration_config(
            mode, num_samples=2, sequence_length=seq_len, group_size=-1
        )
        if mode == "awq":
            config.num_grid_points = 3
        x = embedding(np.zeros((1, seq_len), "int32"))
        for block in blocks:
            block(x)
        rng = np.random.default_rng(1)
        config.dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(2)
        ]
        config.tokenizer = lambda text: text
        strategy = strategy_registry.get_strategy(mode)
        run = CalibrationRun(strategy, config, structure)
        for block in blocks:
            run.add(block.dense)
        run.run()
        for block in blocks:
            _assert_calibrated(self, block.dense, mode)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_skips_blocks_without_quantizable_layers(self, mode):
        vocab_size, seq_len, embed_dim = 32, 8, 4
        embedding = layers.Embedding(vocab_size, embed_dim)
        empty = EmptyBlock()
        block = models.Sequential([layers.Dense(4)])
        structure = {
            "pre_block_layers": [embedding],
            "sequential_blocks": [empty, block],
        }
        config = calibration_config(
            mode, num_samples=2, sequence_length=seq_len, group_size=-1
        )
        if mode == "awq":
            config.num_grid_points = 3
        x = embedding(np.zeros((1, seq_len), "int32"))
        block(empty(x))
        rng = np.random.default_rng(2)
        config.dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(2)
        ]
        config.tokenizer = lambda text: text
        strategy = strategy_registry.get_strategy(mode)
        run = CalibrationRun(strategy, config, structure)
        self.assertFalse(run.covers(empty.ln))
        run.add(block.layers[0])
        run.run()
        _assert_calibrated(self, block.layers[0], mode)


@pytest.mark.requires_trainable_backend
class CalibrationRunModelTest(testing.TestCase):
    """The run as `model.quantize` drives it, once per mode."""

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_captures_observe_real_inputs(self, mode):
        """Regression test for #23512.

        `model.quantize` switches the layers to their quantized dtype
        policy before calibration, so their forward dispatches to
        `quantized_call`. The hooks patched `call` only and never fired:
        GPTQ's Hessian stayed all-zeros (replaced by the identity, which
        disables the error correction) and AWQ's scale search was
        activation-blind.
        """
        model, config = _tiny_model(mode)
        with _spy_on_observe(mode) as observe:
            model.quantize(mode, config=config)
        calibrator = observe.call_args.args[0]
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        if mode == "gptq":
            # Real activations are correlated across input features.
            statistic = statistic - np.diag(np.diag(statistic))
        self.assertGreater(np.abs(statistic).max(), 0.0)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_each_stage_is_observed_in_its_own_sweep(self, mode):
        """The chained Dense layers form two stages. Each stage observes
        one sweep over the batches, the second after the first is
        quantized."""
        model, config = _tiny_model(
            mode, num_samples=4, calibration_batch_size=3
        )
        block = config.quantization_layer_structure["sequential_blocks"][0]
        with _spy_on_observe(mode) as observe:
            model.quantize(mode, config=config)
        observed = [call.args[0].layer for call in observe.call_args_list]
        sweep = math.ceil(4 / 3)
        self.assertEqual(
            observed, [block.layers[0]] * sweep + [block.layers[1]] * sweep
        )

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_layers_without_observations_are_named_in_one_warning(self, mode):
        block = TrainingOnlyBlock()
        model, config = _tiny_model(mode, block_layers=[block])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.quantize(mode, config=config)
        messages = [str(w.message) for w in caught]
        unreached = [m for m in messages if "observed no input" in m]
        self.assertLen(unreached, 1)
        self.assertIn(block.training_only.path, unreached[0])
        self.assertNotIn(block.observed.path, unreached[0])
        # A layer that observed nothing is not also undersampled.
        self.assertFalse([m for m in messages if "undersampled" in m])
        for layer in (block.observed, block.training_only):
            _assert_calibrated(self, layer, mode)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_third_party_projection_in_a_block_is_calibrated(self, mode):
        """A layer that lists the mode is calibrated, not only `Dense`.

        The third-party layer computes what the `Dense` it replaces
        computes, so the run must store the same values for both.
        """
        reference, reference_config = _tiny_model(mode)
        custom = RecipeOnlyProjection(16, activation="relu")
        model, config = _tiny_model(
            mode, block_layers=[custom, layers.Dense(8)]
        )
        block = config.quantization_layer_structure["sequential_blocks"][0]
        model.set_weights(reference.get_weights())
        report = model.quantize(mode, config=config)
        reference.quantize(mode, config=reference_config)

        self.assertIn(custom.path, [path for path, *_ in report.quantized])
        _assert_calibrated(self, custom, mode)
        reference_block = reference_config.quantization_layer_structure[
            "sequential_blocks"
        ][0]
        for layer, reference_layer in zip(block.layers, reference_block.layers):
            self.assertEqual(len(layer.weights), len(reference_layer.weights))
            for variable, expected in zip(
                layer.weights, reference_layer.weights
            ):
                self.assertAllEqual(variable, expected)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_equation_without_a_view_is_skipped_and_reported(self, mode):
        dense = layers.Dense(8)
        einsum = layers.EinsumDense("abc,bc->abc", output_shape=(16, 8))
        model, config = _tiny_model(mode, block_layers=[dense, einsum])
        kernel = ops.convert_to_numpy(einsum.kernel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = model.quantize(mode, config=config)
        self.assertIn(
            (einsum.path, QuantizationReport.SKIP_NO_SUPPORT), report.skipped
        )
        self.assertIsNone(einsum.quantization_mode)
        self.assertIsNone(einsum.quantization_config)
        self.assertAllEqual(einsum.kernel, kernel)
        _assert_calibrated(self, dense, mode)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_calibration_runs_without_grad_tracking(self, mode):
        """On torch, graphs of the calibration forwards would keep every
        activation of the run alive and exhaust GPU memory."""
        if backend.backend() != "torch":
            self.skipTest("gradient tracking is specific to torch")
        model, config = _tiny_model(mode)
        with _spy_on_observe(mode) as observe:
            model.quantize(mode, config=config)
        self.assertTrue(observe.called)
        for call in observe.call_args_list:
            self.assertIsNone(getattr(call.args[1], "grad_fn", None))

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_hands_a_rank_4_block_output_on(self, mode):
        # Regression test: the handoff split each batch into samples and
        # re-added the batch axis only to rank-2 samples, so a rank-4
        # output reached the next block with its axes merged.
        vocab_size, seq_len, embed_dim = 32, 8, 16
        inputs = layers.Input((seq_len,), dtype="int32")
        embedding = layers.Embedding(vocab_size, embed_dim)
        first = models.Sequential(
            [layers.Dense(embed_dim), layers.Reshape((seq_len, 2, 8))]
        )
        second = models.Sequential(
            [layers.Reshape((seq_len, embed_dim)), layers.Dense(embed_dim)]
        )
        model = models.Model(inputs, second(first(embedding(inputs))))
        kwargs = {"num_grid_points": 3} if mode == "awq" else {}
        config = calibration_config(
            mode,
            num_samples=6,
            calibration_batch_size=4,
            sequence_length=seq_len,
            group_size=-1,
            **kwargs,
        )
        rng = np.random.default_rng(3)
        config.dataset = [
            rng.integers(0, vocab_size, (1, seq_len), dtype=np.int32)
            for _ in range(6)
        ]
        config.tokenizer = lambda text: text
        config.quantization_layer_structure = {
            "pre_block_layers": [embedding],
            "sequential_blocks": [first, second],
        }
        model.quantize(mode, config=config)
        _assert_calibrated(self, first.layers[0], mode)
        _assert_calibrated(self, second.layers[1], mode)
        outputs = ops.convert_to_numpy(model(config.dataset[0]))
        self.assertTrue(np.isfinite(outputs).all())

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_calibrates_only_float_layers(self, mode):
        # A layer quantized in another mode, or already calibrated, is
        # left as it is instead of failing inside the solve.
        model, config = _tiny_model(mode)
        block = config.quantization_layer_structure["sequential_blocks"][0]
        int8_layer, calibrated_layer = block.layers
        int8_layer.quantize("int8")
        int8_codes = ops.convert_to_numpy(int8_layer._kernel)
        report = model.quantize(mode, config=config)
        self.assertEqual(int8_layer.quantization_mode, "int8")
        self.assertAllEqual(int8_layer._kernel, int8_codes)
        _assert_calibrated(self, calibrated_layer, mode)
        self.assertEqual(
            [path for path, *_ in report.quantized], [calibrated_layer.path]
        )

        # A second call finds no float layer and runs no forward pass.
        codes = ops.convert_to_numpy(calibrated_layer.quantized_kernel)
        with mock.patch.object(CalibrationRun, "_prefix_outputs") as prefix:
            report = model.quantize(mode, config=config)
        prefix.assert_not_called()
        self.assertEqual(report.quantized, [])
        self.assertAllEqual(calibrated_layer.quantized_kernel, codes)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_refused_call_leaves_the_model_untouched(self, mode):
        # The run refuses a missing dataset or structure before the walk,
        # so no layer changes and the model still saves.
        for missing in ("dataset", "tokenizer", "quantization_layer_structure"):
            model, config = _tiny_model(mode)
            block = config.quantization_layer_structure["sequential_blocks"][0]
            setattr(config, missing, None)
            with self.assertRaises(ValueError):
                model.quantize(mode, config=config)
            for layer in block.layers:
                _assert_float(self, layer)
            model.save_weights(
                os.path.join(self.get_temp_dir(), "untouched.weights.h5")
            )

    def test_a_layer_refused_on_its_own_is_calibrated_by_the_run(self):
        # `layer.quantize("gptq")` raises and leaves the layer float; the
        # run then solves it with the run's config, like its neighbours.
        model, config = _tiny_model("gptq", weight_bits=8)
        layer = config.quantization_layer_structure["sequential_blocks"][0]
        layer = layer.layers[0]
        kernel = ops.convert_to_numpy(layer.kernel)
        layer_config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=4, group_size=8
        )
        with self.assertRaisesRegex(ValueError, "model.quantize"):
            layer.quantize("gptq", config=layer_config)
        _assert_float(self, layer)
        with _spy_on_observe("gptq") as observe:
            model.quantize("gptq", config=config)
        for call in observe.call_args_list:
            self.assertIs(call.args[0].config, config)
        _assert_calibrated(self, layer, "gptq")
        self.assertEqual(layer.dtype_policy.name, "gptq/8/8_from_float32")
        weight = layer._quantized_weight().dequantize("float32")
        error = np.linalg.norm(weight - kernel) / np.linalg.norm(kernel)
        self.assertLess(error, 0.05)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_a_run_that_raises_keeps_a_valid_model(self, mode):
        # A failure inside the run leaves the layers solved before it
        # quantized and the rest float. The report lists the quantized
        # layers, the model drops its compiled functions and still saves,
        # and a second call completes the run with its own config.
        model, config = _tiny_model(mode)
        block = config.quantization_layer_structure["sequential_blocks"][0]
        first, second = block.layers
        model.predict(np.stack(config.dataset)[:, 0], verbose=0)
        calibrator_cls = strategy_registry.get_strategy(mode).calibrator_cls
        solve = calibrator_cls.quantize

        def quantize(calibrator):
            if calibrator.layer is second:
                raise RuntimeError("solve failed")
            return solve(calibrator)

        with mock.patch.object(
            calibrator_cls, "quantize", autospec=True, side_effect=quantize
        ):
            with self.assertRaisesRegex(RuntimeError, "solve failed"):
                model.quantize(mode, config=config)
        _assert_calibrated(self, first, mode)
        _assert_float(self, second)
        report = model._quantization_report
        self.assertEqual([path for path, *_ in report.quantized], [first.path])
        self.assertIsNone(model.predict_function)
        path = os.path.join(self.get_temp_dir(), "partial.weights.h5")
        model.save_weights(path)
        model.load_weights(path)

        report = model.quantize(mode, config=config)
        self.assertEqual([path for path, *_ in report.quantized], [second.path])
        _assert_calibrated(self, second, mode)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_under_a_bfloat16_policy(self, mode):
        # The solve runs in float32 when the variables are `bfloat16`.
        model, config = _tiny_model(mode, dtype="bfloat16")
        model.quantize(mode, config=config)
        block = config.quantization_layer_structure["sequential_blocks"][0]
        for layer in block.layers:
            _assert_calibrated(self, layer, mode)
        outputs = ops.convert_to_numpy(
            ops.cast(model(config.dataset[0]), "float32")
        )
        self.assertTrue(np.isfinite(outputs).all())
