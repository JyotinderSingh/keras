import math
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.calibration_run import CalibrationRun
from keras.src.quantizers.calibration_run import _execution_stages
from keras.src.quantizers.calibration_run import find_layers_in_block
from keras.src.quantizers.calibration_run import get_dataloader
from keras.src.quantizers.calibration_run import stream_inputs
from keras.src.quantizers.gptq import GPTQCalibrator
from keras.src.quantizers.gptq_config import GPTQConfig
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


def build_all_tokens_strings(dataset, tokenizer, eos_id=None):
    pieces = []
    for i, s in enumerate(dataset):
        toks = np.asarray(tokenizer.tokenize(s), dtype=np.int32).reshape(-1)
        pieces.append(toks)
        if eos_id is not None and i < len(dataset) - 1:
            pieces.append(np.array([eos_id], dtype=np.int32))
    return np.concatenate(pieces, axis=0).astype(np.int32, copy=False)


def sliding_windows(x, L):
    return np.lib.stride_tricks.sliding_window_view(x, L)


@pytest.mark.requires_trainable_backend
class TestCalibrationCore(testing.TestCase):
    @parameterized.named_parameters(
        [("strided", "strided"), ("linspace", "linspace"), ("random", "random")]
    )
    def test_shape_and_dtype_strings(self, sampling):
        """Test the shape and dtype of the output for string inputs."""
        tok = MockTokenizer()
        dataset = ["a b c d e f g", "h i j k"]
        seq_len, n = 5, 7

        out = get_dataloader(
            tok, seq_len, dataset, num_samples=n, sampling=sampling, seed=123
        )
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    @parameterized.named_parameters(
        [("strided", "strided"), ("linspace", "linspace"), ("random", "random")]
    )
    def test_shape_and_dtype_pretokenized(self, sampling):
        """Test the shape and dtype of the output for pre-tokenized inputs."""
        tok = MockTokenizer()
        # Pre-tokenized inputs; mixed shapes (1, L) and (L,)
        seqs = [
            np.array([[1, 2, 3, 4]], dtype=np.int64),
            np.array([5, 6], dtype=np.int64),
        ]
        tok = MockTokenizer()
        seq_len, n = 3, 4

        out = get_dataloader(
            tok, seq_len, seqs, num_samples=n, sampling=sampling, seed=7
        )
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_strided_is_deterministic_for_same_args(self):
        tok = MockTokenizer()
        dataset = ["a b c d e", "f g h i j k"]
        out1 = get_dataloader(
            tok, 4, dataset, num_samples=6, sampling="strided", seed=99
        )
        out2 = get_dataloader(
            tok, 4, dataset, num_samples=6, sampling="strided", seed=99
        )
        self.assertTrue(ops.all(ops.equal(out1, out2)))

    def test_random_reproducibility_by_seed(self):
        tok = MockTokenizer()
        dataset = ["a b c d e", "f g h i j k"]
        a = get_dataloader(
            tok, 4, dataset, num_samples=6, sampling="random", seed=123
        )
        b = get_dataloader(
            tok, 4, dataset, num_samples=6, sampling="random", seed=123
        )
        c = get_dataloader(
            tok, 4, dataset, num_samples=6, sampling="random", seed=124
        )
        self.assertTrue(ops.all(ops.equal(a, b)))
        self.assertFalse(ops.all(ops.equal(a, c)))

    def test_linspace_windows_match_expected(self):
        tok = MockTokenizer()
        dataset = ["aa bb cc dd", "ee ff gg"]
        seq_len, n = 3, 5
        eos_id = None

        all_tokens = build_all_tokens_strings(dataset, tok, eos_id=eos_id)
        max_start = all_tokens.size - seq_len
        expected_starts = np.linspace(0, max_start, n, dtype=np.int64)

        expected = sliding_windows(all_tokens, seq_len)[expected_starts]
        got = get_dataloader(
            tok, seq_len, dataset, num_samples=n, sampling="linspace"
        )
        self.assertTrue(
            ops.all(ops.equal(got[:, 0, :], expected.astype(np.int32)))
        )

    def test_strided_override_respected(self):
        """Tests that strided windows are disjoint and cover the input."""
        tok = MockTokenizer()
        # 20 tokens total
        # with seq_len=4 and stride=4, we expect disjoint chunks
        # in order (modulo offset)
        dataset = [" ".join([f"t{i}" for i in range(20)])]
        seq_len, n, stride = 4, 5, 4

        out = get_dataloader(
            tok,
            seq_len,
            dataset,
            num_samples=n,
            sampling="strided",
            stride=stride,
            seed=0,
        )

        # Validate that each sample is a contiguous run
        # of length seq_len from the flattened stream
        flat = build_all_tokens_strings(dataset, tok)
        for s in out[:, 0, :]:
            # Each window should appear as a slice in the flat stream
            # (This is a soft check; exact start positions depend on offset.)
            joined = " ".join(map(str, s.tolist()))
            self.assertIn(joined, " ".join(map(str, flat.tolist())))

    def test_eos_insertion_is_present_in_some_window_with_linspace(self):
        tok = MockTokenizer()
        dataset = ["aa aa", "bb bb"]  # len = 5 + 1(EOS) + 5 = 11
        eos = 9999
        seq_len = 3
        n = 3

        out = get_dataloader(
            tok,
            seq_len,
            dataset,
            num_samples=n,
            sampling="linspace",
            eos_id=eos,
        )

        # linspace starts -> [0, 4, 8]; the middle window [4:7]
        # includes EOS at 5
        windows = out[:, 0, :]
        self.assertTrue(
            np.any(np.any(windows == eos, axis=1)),
            "Expected EOS to appear in at least one sampled window with "
            "linspace.",
        )

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
        calibrator_cls = strategy_registry.get_strategy(mode).calibrator_cls

        def accumulate(batch_size):
            layers_map = find_layers_in_block(block)
            calibrators = {
                name: calibrator_cls(layer, _config(mode))
                for name, layer in layers_map.items()
            }
            with stream_inputs(layers_map, calibrators):
                for start in range(0, num_samples, batch_size):
                    block(samples[start : start + batch_size])
            (calibrator,) = calibrators.values()
            return _statistic(calibrator), calibrator.num_samples

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


class TestExecutionStages(testing.TestCase):
    def test_stages_group_by_shared_input_and_order(self):
        x_attn, x_out, x_mlp, x_down = (
            object(),
            object(),
            object(),
            object(),
        )
        trace = {
            "query": (0, x_attn),
            "key": (1, x_attn),
            "value": (2, x_attn),
            "attention_output": (3, x_out),
            "gate": (4, x_mlp),
            "up": (5, x_mlp),
            "down": (6, x_down),
        }
        stages = _execution_stages(list(trace), trace)
        self.assertEqual(
            stages,
            [
                ["query", "key", "value"],
                ["attention_output"],
                ["gate", "up"],
                ["down"],
            ],
        )

    def test_untraced_layers_join_first_stage(self):
        x = object()
        trace = {"a": (0, x)}
        stages = _execution_stages(["ghost", "a"], trace)
        self.assertEqual(stages, [["ghost", "a"]])

    def test_no_trace_single_stage(self):
        stages = _execution_stages(["a", "b"], {})
        self.assertEqual(stages, [["a", "b"]])


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


def _config(mode, **kwargs):
    if mode == "gptq":
        return GPTQConfig(dataset=None, tokenizer=None, **kwargs)
    return AWQConfig(dataset=None, tokenizer=None, **kwargs)


def _statistic(calibrator):
    """The statistic a calibrator accumulates from the layer's inputs."""
    if isinstance(calibrator, GPTQCalibrator):
        return calibrator.hessian
    return calibrator.activation_magnitudes


def _tiny_model(mode, num_samples=4, dtype=None, **kwargs):
    """Embedding -> [Dense(16, relu), Dense(8)] block -> pooled head."""
    set_random_seed(123)
    seq_len, vocab_size, embed_dim = 16, 48, 8
    block = models.Sequential(
        [
            layers.Dense(16, activation="relu", dtype=dtype),
            layers.Dense(embed_dim, dtype=dtype),
        ]
    )
    inputs = layers.Input((seq_len,), dtype="int32")
    embedding = layers.Embedding(vocab_size, embed_dim, dtype=dtype)
    x = layers.GlobalAveragePooling1D(dtype=dtype)(block(embedding(inputs)))
    model = models.Model(inputs, layers.Dense(4, dtype=dtype)(x))
    if mode == "awq":
        kwargs["num_grid_points"] = 5
    config = _config(
        mode,
        num_samples=num_samples,
        sequence_length=seq_len,
        group_size=8,
        **kwargs,
    )
    rng = np.random.default_rng(7)
    config.dataset = [
        rng.integers(0, vocab_size, (1, seq_len), dtype=np.int32)
        for _ in range(num_samples)
    ]
    config.tokenizer = lambda text: text
    config.quantization_layer_structure = {
        "pre_block_layers": [embedding],
        "sequential_blocks": [block],
    }
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
            CalibrationRun(strategy, _config("gptq"), {"pre_block_layers": []})

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_calibrate_requires_a_dataset(self, mode):
        strategy = strategy_registry.get_strategy(mode)
        with self.assertRaisesRegex(
            ValueError, f"{mode.upper()} quantization requires a dataset"
        ):
            strategy.calibrate(_config(mode), {"sequential_blocks": [1]})

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_calibrator_refuses_unsupported_layers(self, mode):
        calibrator_cls = strategy_registry.get_strategy(mode).calibrator_cls
        ternary = layers.TernaryDense(4)
        ternary.build((None, 3))
        for layer in (layers.Layer(), ternary):
            with self.assertRaisesRegex(
                TypeError, f"Unsupported layer type for {mode.upper()}"
            ):
                calibrator_cls(layer, _config(mode))

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
        config = _config(mode, **kwargs)
        config.quantization_layer_structure = structure
        rng = np.random.default_rng(0)
        dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(4)
        ]
        for block in blocks:
            for dense in block.layers:
                dense.quantize(mode, config=config)
                self.assertTrue(dense.calibration_pending)

        strategy = strategy_registry.get_strategy(mode)
        run = CalibrationRun(strategy, config, structure)
        self.assertEqual(run.batch_size, 2)
        run.run(dataset)
        self.assertEqual(run.num_samples, 3)
        for block in blocks:
            for dense in block.layers:
                self.assertFalse(dense.calibration_pending)
        outputs = ops.convert_to_numpy(model(dataset[0]))
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
        config = _config(
            mode, num_samples=2, sequence_length=seq_len, group_size=-1
        )
        if mode == "awq":
            config.num_grid_points = 3
        x = embedding(np.zeros((1, seq_len), "int32"))
        for block in blocks:
            block(x)
            block.dense.quantize(mode, config=config)
        rng = np.random.default_rng(1)
        dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(2)
        ]
        strategy = strategy_registry.get_strategy(mode)
        CalibrationRun(strategy, config, structure).run(dataset)
        for block in blocks:
            self.assertFalse(block.dense.calibration_pending)

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
        config = _config(
            mode, num_samples=2, sequence_length=seq_len, group_size=-1
        )
        if mode == "awq":
            config.num_grid_points = 3
        x = embedding(np.zeros((1, seq_len), "int32"))
        block(empty(x))
        block.layers[0].quantize(mode, config=config)
        rng = np.random.default_rng(2)
        dataset = [
            rng.integers(0, vocab_size, (1, seq_len)).astype("int32")
            for _ in range(2)
        ]
        strategy = strategy_registry.get_strategy(mode)
        CalibrationRun(strategy, config, structure).run(dataset)
        self.assertFalse(block.layers[0].calibration_pending)


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
        statistic = ops.convert_to_numpy(_statistic(calibrator))
        if mode == "gptq":
            # Real activations are correlated across input features.
            statistic = statistic - np.diag(np.diag(statistic))
        self.assertGreater(np.abs(statistic).max(), 0.0)

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_later_stages_are_observed_again(self, mode):
        """The chained Dense layers form two stages. The first sweep
        observes both (two calls per forward pass); once the first layer is
        quantized, the second is observed again on its quantized output
        (one more call per forward pass)."""
        model, config = _tiny_model(
            mode, num_samples=4, calibration_batch_size=3
        )
        with _spy_on_observe(mode) as observe:
            model.quantize(mode, config=config)
        self.assertEqual(observe.call_count, 3 * math.ceil(4 / 3))

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
    def test_run_under_a_bfloat16_policy(self, mode):
        # The solve runs in float32 when the variables are `bfloat16`.
        model, config = _tiny_model(mode, dtype="bfloat16")
        model.quantize(mode, config=config)
        block = config.quantization_layer_structure["sequential_blocks"][0]
        for layer in block.layers:
            self.assertFalse(layer.calibration_pending)
        outputs = ops.convert_to_numpy(
            ops.cast(model(config.dataset[0]), "float32")
        )
        self.assertTrue(np.isfinite(outputs).all())
