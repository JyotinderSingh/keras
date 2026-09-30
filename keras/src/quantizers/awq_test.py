"""Tests for AWQ quantization."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers import awq
from keras.src.quantizers.awq import AWQCalibrator
from keras.src.quantizers.awq import _fake_quantize_weights
from keras.src.quantizers.awq import _get_weight_scale
from keras.src.quantizers.awq import awq_quantize_matrix
from keras.src.quantizers.awq import awq_search_best_clip
from keras.src.quantizers.awq import awq_search_optimal_scales
from keras.src.quantizers.awq_config import AWQConfig

# Shared RNG instance for reproducible tests
RNG = np.random.default_rng(seed=42)


def _hessian(x):
    """The Hessian `2 mean(x x^T)` of a sample `x` [rows, in_features]."""
    x = np.asarray(x, "float32")
    return 2.0 * x.T @ x / x.shape[0]


def _random_hessian(weights):
    """A Hessian for `weights` [out, in] from a random sample."""
    in_features = int(weights.shape[1])
    return _hessian(RNG.standard_normal((4 * in_features, in_features)))


class MockTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size=100, seq_len=64):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def tokenize(self, text):
        # Simple character-based tokenization
        tokens = [ord(c) % self.vocab_size for c in str(text)]
        # Pad or truncate to seq_len
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        return ops.array([tokens], dtype="int32")

    def __call__(self, text):
        return self.tokenize(text)


@pytest.mark.requires_trainable_backend
class AWQAlgorithmTest(testing.TestCase):
    """Test AWQ algorithm core functionality."""

    def test_scale_search_returns_valid_scales(self):
        """Test that scale search returns valid positive scales."""
        weights = RNG.standard_normal((32, 16)).astype("float32")
        activations = ops.abs(
            ops.add(RNG.standard_normal((16,)).astype("float32"), 0.1)
        )

        scales = awq_search_optimal_scales(
            weights,
            activations,
            _random_hessian(weights),
            num_grid_points=10,
            group_size=-1,
        )

        self.assertEqual(scales.shape, (16,))
        # All scales should be positive
        self.assertTrue(ops.all(ops.greater(scales, 0)))

    def test_scale_search_with_zero_activations(self):
        """Test scale search handles near-zero activations."""
        weights = ops.array(RNG.standard_normal((32, 16)).astype("float32"))
        # Some activations are very small
        activations = np.abs(RNG.standard_normal((16,)).astype("float32"))
        activations[:5] = 1e-10
        activations = ops.array(activations)

        scales = awq_search_optimal_scales(
            weights,
            activations,
            _random_hessian(weights),
            num_grid_points=10,
            group_size=-1,
        )

        # Should handle gracefully without NaN or Inf
        self.assertFalse(ops.any(ops.isnan(scales)))
        self.assertFalse(ops.any(ops.isinf(scales)))

    def test_get_weight_scale_shape_and_range(self):
        """Weight statistic is per-in-channel and normalized to (0, 1]."""
        weights = RNG.standard_normal((32, 64)).astype("float32")
        for group_size in (-1, 16):
            w_stat = _get_weight_scale(weights, group_size)
            self.assertEqual(w_stat.shape, (64,))
            self.assertTrue(ops.all(ops.greater(w_stat, 0)))
            self.assertTrue(ops.all(ops.less_equal(w_stat, 1.0 + 1e-6)))

    def test_search_best_clip_shapes(self):
        """Clip search returns a per-group bound not exceeding the max."""
        weights = RNG.standard_normal((64, 32)).astype("float32")
        scales = ops.add(
            ops.abs(RNG.standard_normal((32,)).astype("float32")), 0.1
        )
        weights_scaled = ops.multiply(weights, scales)
        sample = RNG.standard_normal((128, 32)).astype("float32")

        best_max = awq_search_best_clip(
            weights_scaled,
            _hessian(sample),
            scales,
            group_size=8,
        )
        self.assertEqual(best_max.shape, (64, 4, 1))
        # Bound must be positive and never exceed the original per-group max.
        w_grouped = ops.reshape(weights_scaled, (64, 4, 8))
        org_max = ops.max(ops.abs(w_grouped), axis=-1, keepdims=True)
        self.assertTrue(ops.all(ops.greater(best_max, 0)))
        self.assertTrue(
            ops.all(ops.less_equal(best_max, ops.add(org_max, 1e-6)))
        )

    @parameterized.named_parameters(("per_channel", -1), ("grouped", 8))
    def test_quantize_matrix_shapes(self, group_size):
        weights = ops.array(RNG.standard_normal((64, 32)).astype("float32"))
        activations = ops.add(
            ops.abs(RNG.standard_normal((32,)).astype("float32")), 0.1
        )
        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights,
            activations,
            _random_hessian(weights),
            num_grid_points=5,
            group_size=group_size,
        )
        n_groups = 1 if group_size == -1 else 32 // group_size
        self.assertEqual(quantized.shape, (64, 32))
        self.assertEqual(scale.shape, (64, n_groups))
        self.assertEqual(zero.shape, (64, n_groups))
        self.assertEqual(awq_scales.shape, (32,))
        self.assertAllEqual(g_idx, np.arange(32) // (32 // n_groups))


@pytest.mark.requires_trainable_backend
class AWQLayerTest(testing.TestCase):
    """Test AWQ class for layer quantization."""

    def test_awq_on_dense_layer(self):
        """Test AWQ on a Dense layer."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            num_grid_points=10,
        )

        layer.quantize(config=config)
        calibrator = AWQCalibrator(layer, config)

        # Simulate activation capture
        calibration_data = RNG.standard_normal((64, 16)).astype("float32")
        calibrator.observe(calibration_data)

        self.assertEqual(calibrator.num_samples, 64)
        # Activation magnitudes should be non-negative
        self.assertTrue(
            ops.all(ops.greater_equal(calibrator.activation_magnitudes, 0))
        )

    def test_awq_activation_accumulation(self):
        """Test that activation magnitudes accumulate as a running mean.

        The reference AWQ statistic is the per-channel mean of |x|. The running
        update must be equivalent to computing that mean over all rows seen so
        far, regardless of how the rows are split into batches.
        """
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)
        calibrator = AWQCalibrator(layer, config)

        # First batch.
        batch1 = RNG.standard_normal((10, 16)).astype("float32")
        calibrator.observe(batch1)

        # Second batch with a different row count to exercise the weighting.
        batch2 = ops.add(RNG.standard_normal((30, 16)).astype("float32"), 1.0)
        calibrator.observe(batch2)

        # Running mean must equal the mean of |x| over all rows.
        combined = ops.concatenate([ops.abs(batch1), ops.abs(batch2)], axis=0)
        expected_mean = ops.mean(combined, axis=0)
        self.assertEqual(calibrator.num_samples, 40)
        self.assertAllClose(
            calibrator.activation_magnitudes, expected_mean, atol=1e-6
        )

    def test_hessian_accumulates_over_batches(self):
        # The running means of `|x|` and of `2 x x^T` do not depend on the
        # batching.
        x = RNG.standard_normal((96, 16)).astype("float32")
        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=5
        )

        def calibrator():
            layer = layers.Dense(32)
            layer.build(input_shape=(None, 16))
            layer.quantize(config=config)
            return AWQCalibrator(layer, config)

        whole = calibrator()
        whole.observe(x)
        batched = calibrator()
        batched.observe(x[:40])
        batched.observe(x[40:])
        self.assertEqual(batched.num_samples, 96)
        self.assertAllClose(whole.hessian, _hessian(x), atol=1e-5)
        self.assertAllClose(batched.hessian, whole.hessian, atol=1e-5)
        self.assertAllClose(
            batched.activation_magnitudes, whole.activation_magnitudes
        )

    def test_query_and_key_layers_skip_the_clipping_search(self):
        # The reference rule: the attention scores depend on the product of
        # the two projections, so one layer's output error is a poor guide
        # for their clip bound. The pattern list is configurable.
        weights = RNG.standard_normal((64, 32)).astype("float32")
        weights[::8] *= 6.0  # outliers, so clipping changes the codes
        x = RNG.standard_normal((128, 32)).astype("float32")

        def codes(name, apply_clip, **kwargs):
            layer = layers.Dense(64, name=name)
            layer.build(input_shape=(None, 32))
            layer.kernel.assign(weights.T)
            config = AWQConfig(
                dataset=None,
                tokenizer=None,
                group_size=8,
                num_grid_points=5,
                apply_clip=apply_clip,
                **kwargs,
            )
            layer.quantize("awq", config=config)
            calibrator = AWQCalibrator(layer, config)
            calibrator.observe(x)
            calibrator.quantize()
            return ops.convert_to_numpy(layer.quantized_kernel)

        unclipped = codes("dense", apply_clip=False)
        clipped = codes("dense", apply_clip=True)
        self.assertFalse(np.array_equal(clipped, unclipped))
        self.assertAllEqual(codes("query", apply_clip=True), unclipped)
        self.assertAllEqual(codes("key_dense", apply_clip=True), unclipped)
        self.assertAllEqual(
            codes("query", apply_clip=True, clip_skip_patterns=()), clipped
        )

    def test_awq_layer_variables_created(self):
        """Test that AWQ layer variables are properly created."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)

        # Check that AWQ-specific variables exist
        self.assertTrue(hasattr(layer, "quantized_kernel"))
        self.assertTrue(hasattr(layer, "kernel_scale"))
        self.assertIsNotNone(layer.kernel_zero)
        self.assertTrue(hasattr(layer, "awq_scales"))
        self.assertIsNotNone(layer.g_idx)
        self.assertTrue(layer.calibration_pending)


@pytest.mark.requires_trainable_backend
class AWQIntegrationTest(testing.TestCase):
    """Integration tests for AWQ quantization."""

    def test_dense_layer_quantize_awq(self):
        """Test Dense layer can be quantized with AWQ."""
        layer = layers.Dense(64)
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=16, num_grid_points=5
        )
        layer.quantize(config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_einsum_dense_layer_quantize_awq(self):
        """Test EinsumDense layer can be quantized with AWQ."""
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(64,))
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=5
        )
        layer.quantize(config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_model_quantize_requires_structure(self):
        """Test model.quantize requires structure for AWQ."""
        model = models.Sequential([layers.Dense(10, input_shape=(5,))])
        model.build()

        config = AWQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(vocab_size=100, seq_len=5),
        )

        with self.assertRaisesRegex(ValueError, "quantization structure"):
            model.quantize(config=config)

    @pytest.mark.requires_trainable_backend
    def test_awq_save_load_round_trip_einsum_dense_block(self):
        """Regression test for a RecursionError when saving an AWQ model.

        Saving an AWQ-quantized model used to raise a RecursionError because
        `AWQConfig.get_config` serialized `quantization_layer_structure`,
        which holds live model layers, forming a reference cycle
        (layer -> config -> layer). This builds a tiny model whose quantized
        block contains both `Dense` and `EinsumDense` (unlike the Dense-only
        round trip in `AWQAccuracyTest`), saves it, reloads it, and checks
        the predictions are preserved exactly.
        """
        vocab_size, seq_len, embed_dim = 32, 8, 4

        inputs = layers.Input(shape=(seq_len,), dtype="int32")
        embedding = layers.Embedding(vocab_size, embed_dim)
        x = embedding(inputs)
        block = models.Sequential(
            [
                layers.Dense(embed_dim, activation="relu"),
                # Gemma's `[heads, d_model, head_dim]` query projection,
                # whose contracted axis does not lead the kernel.
                layers.EinsumDense(
                    "btd,ndh->btnh", output_shape=(seq_len, 2, 2)
                ),
                layers.Reshape((seq_len, embed_dim)),
                layers.EinsumDense(
                    "abc,cd->abd", output_shape=(seq_len, embed_dim)
                ),
            ]
        )
        x = block(x)
        x = layers.GlobalAveragePooling1D()(x)
        head = layers.Dense(2)
        outputs = head(x)
        model = models.Model(inputs, outputs)

        rng = np.random.default_rng(seed=21)
        dataset = [
            rng.integers(0, vocab_size, size=(1, seq_len)).astype("int32")
            for _ in range(3)
        ]
        config = AWQConfig(
            dataset=dataset,
            tokenizer=lambda text: text,
            num_samples=2,
            sequence_length=seq_len,
            group_size=4,
            num_grid_points=5,
            quantization_layer_structure={
                "pre_block_layers": [embedding],
                "sequential_blocks": [block],
            },
        )

        # Layers outside the structure (embedding, pooling, head) are not
        # quantized at all, so the round-trip can be compared exactly.
        model.quantize("awq", config=config)
        self.assertIsNone(getattr(head, "quantization_mode", None))

        # The embedding only supports int8/int4; `quantize` must reject the
        # unsupported mode without stashing a stale AWQ config on it.
        self.assertIsNone(embedding.quantization_config)

        x_eval = rng.integers(0, vocab_size, size=(2, seq_len)).astype("int32")
        y_quantized = model.predict(x_eval)

        # This `save` used to raise a RecursionError.
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        restored = saving.load_model(path)
        y_restored = restored.predict(x_eval)
        self.assertAllClose(y_quantized, y_restored)

        # The quantized block state survives the round-trip.
        restored_block = next(
            l for l in restored.layers if isinstance(l, models.Sequential)
        )
        restored_dense = restored_block.layers[0]
        self.assertEqual(
            getattr(restored_dense, "quantization_mode", None), "awq"
        )
        self.assertTrue(hasattr(restored_dense, "quantized_kernel"))
        self.assertIsNone(
            restored_dense.quantization_config.quantization_layer_structure
        )
        # Stored by the model width: 4 rows of 4 columns packed to 2 bytes.
        self.assertEqual(
            tuple(restored_block.layers[1].quantized_kernel.shape), (4, 2)
        )


# Constants for end-to-end tests
VOCAB_SIZE = 1000
SEQ_LEN = 128
NUM_SAMPLES = 16
NUM_CLASSES = 32

CALIBRATION_TEXT = """
AWQ (Activation-aware Weight Quantization) is an efficient and accurate
low-bit weight quantization method for LLMs. AWQ is based on the observation
that weights are not equally important: protecting only 1% of salient weights
can greatly reduce quantization error. To find salient weights, AWQ looks at
the activation distribution, not weights. Salient weights are those that
correspond to channels with larger activation magnitudes. AWQ then applies
per-channel scaling to protect salient weights during quantization.
The key insight is that for a weight channel, if the corresponding activation
channel has large values, quantizing that weight channel will incur large
error. By scaling up salient weight channels before quantization and scaling
down during inference, AWQ can significantly reduce quantization error
while maintaining the same effective computation.
"""


def _mean_kl(p, q):
    """Compute mean KL divergence between two probability distributions."""
    eps = 1e-8
    p = ops.clip(p, eps, 1.0)
    q = ops.clip(q, eps, 1.0)
    return ops.mean(
        ops.sum(ops.multiply(p, ops.subtract(ops.log(p), ops.log(q))), axis=-1)
    )


def _top1_match_rate(a_logits, b_logits):
    """Calculate top-1 match rate between two sets of logits."""
    return ops.mean(
        ops.equal(ops.argmax(a_logits, axis=-1), ops.argmax(b_logits, axis=-1))
    )


def _get_sequence_classifier():
    """Create a transformer-based sequence classifier for testing."""
    embed_dim = 32
    num_heads = 4
    ff_dim = 32

    class SimpleTransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
            super().__init__(**kwargs)
            self.att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.ffn = models.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            attention_output = self.att(inputs, inputs)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            return self.layernorm2(out1 + ffn_output)

    inputs = layers.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, embed_dim)(inputs)
    x = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return models.Model(inputs, outputs)


def _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Character-based tokenizer for testing."""

    def _pad_or_trim_1d(ids, length):
        ids = ops.ravel(ops.array(ids, "int64"))
        if len(ids) < length:
            ids = ops.concatenate(
                [ids, ops.zeros(length - len(ids), dtype=ids.dtype)]
            )
        else:
            ids = ids[:length]
        return ids

    def _tok(x):
        if isinstance(x, str):
            ids = ops.convert_to_tensor(
                np.fromiter((ord(c) % vocab_size for c in x), dtype=np.int64)
            )
        else:
            ids = np.asarray(x, dtype=np.int64)
        ids = _pad_or_trim_1d(ids, seq_len)
        return ids[None, :]

    _tok.tokenize = _tok
    return _tok


def _string_dataset(
    long_text, num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN
):
    """Yield string slices for calibration."""
    length = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = RNG.integers(0, length) if length > 1 else 0
        yield long_text[start : start + sequence_length]


@pytest.mark.requires_trainable_backend
class AWQAccuracyTest(testing.TestCase):
    """End-to-end accuracy preservation tests for AWQ quantization."""

    @parameterized.named_parameters(
        ("per_channel", -1, 20, 0.5, 0.30),
        ("group_16", 16, 10, 0.4, 0.40),
    )
    def test_awq_transformer_accuracy(
        self, group_size, num_grid_points, min_top1, max_kl
    ):
        """Test that AWQ quantization preserves model accuracy.

        This test:
        1. Creates a transformer-based sequence classifier
        2. Gets baseline (full precision) predictions
        3. Applies AWQ quantization with calibration data
        4. Compares quantized predictions against baseline
        5. Validates top-1 match rate and KL divergence bounds
        """
        keras.utils.set_random_seed(123)

        # Build calibration dataset
        calibration_set = list(_string_dataset(CALIBRATION_TEXT, NUM_SAMPLES))
        self.assertNotEmpty(calibration_set)

        # Build model and tokenizer
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build eval batch from same distribution as calibration
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[RNG.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Get baseline predictions (full precision)
        y_ref = model.predict(x_eval)

        # Define layer structure for AWQ
        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]

        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        # Configure AWQ
        awq_config = AWQConfig(
            dataset=calibration_set,
            tokenizer=tokenizer,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=group_size,
            num_grid_points=num_grid_points,
            quantization_layer_structure=layer_structure,
        )

        # Quantize model with AWQ
        model.quantize(config=awq_config)

        # Get post-quantization predictions
        y_q = model.predict(x_eval)

        # Calculate accuracy metrics
        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref = ops.softmax(y_ref)
        p_q = ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        # Validate accuracy preservation
        self.assertGreaterEqual(
            float(top1_match),
            min_top1,
            f"Top-1 agreement too low for group_size={group_size}: "
            f"{float(top1_match):.3f}",
        )
        self.assertLessEqual(
            float(kl),
            max_kl,
            f"KL divergence too high for group_size={group_size}: "
            f"{float(kl):.3f}",
        )

    @parameterized.named_parameters(
        ("per_channel", -1, 0.35),
        ("group_16", 16, 0.35),
        ("group_32", 32, 0.35),
        ("group_64", 64, 0.35),
        ("group_128", 128, 0.35),
    )
    def test_awq_accuracy_various_group_sizes(
        self, group_size, max_relative_mse
    ):
        """Test AWQ accuracy across various group sizes.

        Verifies that quantizing a single layer maintains reasonable
        output reconstruction error and correct variable shapes.
        """
        in_features = 128
        out_features = 64

        keras.utils.set_random_seed(42)

        # Create fresh layer for each test
        layer = layers.Dense(out_features)
        layer.build(input_shape=(None, in_features))

        # Create data
        calibration_data = RNG.standard_normal((64, in_features)).astype(
            "float32"
        )
        test_data = RNG.standard_normal((16, in_features)).astype("float32")

        # Get original output
        original_output = layer(test_data)

        # Configure and quantize
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=group_size,
            num_grid_points=5,
        )
        layer.quantize(config=config)

        calibrator = AWQCalibrator(layer, config)
        calibrator.observe(calibration_data)
        calibrator.quantize()

        # Verify layer variables have correct shapes for grouped quantization
        if group_size > 0:
            n_groups = in_features // group_size
            self.assertEqual(
                layer.kernel_scale.shape,
                (n_groups, out_features),
                f"kernel_scale shape mismatch for group_size={group_size}",
            )
            self.assertEqual(
                layer.kernel_zero.shape,
                (n_groups, out_features),
                f"kernel_zero shape mismatch for group_size={group_size}",
            )

        # Verify output
        quantized_output = layer(test_data)

        # Should have no NaN/Inf
        self.assertFalse(
            ops.any(ops.isnan(quantized_output)),
            f"NaN in output for group_size={group_size}",
        )
        self.assertFalse(
            ops.any(ops.isinf(quantized_output)),
            f"Inf in output for group_size={group_size}",
        )

        # Should maintain reasonable accuracy
        mse = ops.mean(
            ops.power(ops.subtract(original_output, quantized_output), 2)
        )
        original_var = ops.var(original_output)
        relative_mse = ops.divide(mse, ops.add(original_var, 1e-8))

        self.assertLess(
            relative_mse,
            max_relative_mse,
            f"Accuracy too low for group_size={group_size}: "
            f"relative_mse={relative_mse:.4f}",
        )

    def test_awq_save_load_round_trip(self):
        """Full AWQ quantize -> save -> load round trip.

        Only the Dense layers inside the structure's ``sequential_blocks``
        are quantized; the embedding and classifier head stay untouched,
        and predictions are reproduced after a save/load cycle. Uses small
        local dimensions to keep the grid search fast.
        """
        keras.utils.set_random_seed(123)
        seq_len = 16
        vocab = 48
        num_classes = 4
        embed_dim = 8

        block = models.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        inputs = layers.Input(shape=(seq_len,), dtype="int32")
        embedding = layers.Embedding(vocab, embed_dim)
        x = embedding(inputs)
        x = block(x)
        x = layers.GlobalAveragePooling1D()(x)
        head = layers.Dense(num_classes)
        outputs = head(x)
        model = models.Model(inputs, outputs)

        rng = np.random.default_rng(seed=7)
        dataset = [
            rng.integers(0, vocab, size=(1, seq_len), dtype=np.int32)
            for _ in range(4)
        ]
        tokenizer = _char_tokenizer(vocab_size=vocab, seq_len=seq_len)

        config = AWQConfig(
            dataset=dataset,
            tokenizer=tokenizer,
            group_size=8,
            num_samples=4,
            sequence_length=seq_len,
            num_grid_points=5,
            quantization_layer_structure={
                "pre_block_layers": [embedding],
                "sequential_blocks": [block],
            },
        )

        model.quantize("awq", config=config)

        # In-structure Dense layers are quantized and calibrated.
        for dense in block.layers:
            self.assertEqual(dense.quantization_mode, "awq")
            self.assertFalse(dense.calibration_pending)

        # Out-of-structure layers must stay completely untouched.
        self.assertIsNone(getattr(head, "quantization_mode", None))
        self.assertFalse(hasattr(head, "quantized_kernel"))
        self.assertIsNone(getattr(embedding, "quantization_mode", None))
        self.assertFalse(hasattr(embedding, "quantized_kernel"))

        # Predictions survive a save/load round trip.
        eval_rng = np.random.default_rng(seed=99)
        x_eval = eval_rng.integers(0, vocab, size=(4, seq_len), dtype=np.int32)
        y_before = model.predict(x_eval)

        path = os.path.join(self.get_temp_dir(), "awq_model.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        y_after = reloaded.predict(x_eval)

        self.assertAllClose(y_before, y_after)


def _pad_groups(values, group_size):
    """Zero-pads the last axis to a multiple of `group_size`.

    Keras pads a short last group this way; the references refuse such a
    width.
    """
    padding = -values.shape[-1] % group_size if group_size > 0 else 0
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, padding)])


def _keras_pseudo_quantize(weights, group_size, bits=4):
    """Fake quantization with Keras's group rule, in float64.

    The range is stretched to include zero (GPTQ's rule, which
    `compute_awq_scale_zero` shares); llm-awq's `pseudo_quantize_tensor`
    uses the raw group range. The two agree on every group that spans
    zero.
    """
    out_features, in_features = weights.shape
    group = group_size if group_size > 0 else in_features
    grouped = _pad_groups(weights, group_size).reshape(-1, group)
    low = np.minimum(grouped.min(axis=1, keepdims=True), 0.0)
    high = np.maximum(grouped.max(axis=1, keepdims=True), 0.0)
    flat = low == high
    low = np.where(flat, low - 1.0, low)
    high = np.where(flat, high + 1.0, high)
    maxq = 2**bits - 1
    scale = (high - low) / maxq
    zero = np.clip(np.round(-low / scale), 0, maxq)
    codes = np.clip(np.round(grouped / scale) + zero, 0, maxq)
    dequantized = ((codes - zero) * scale).reshape(out_features, -1)
    return dequantized[:, :in_features]


def _reference_scale_losses(weights, x, group_size, n_grid=20):
    """AutoAWQ `_compute_best_scale` (duo scaling) for one linear layer.

    Returns the candidate scales and the layer's output error on every
    row of `x` for each `ratio = i / n_grid`, in float64.
    """
    weights = np.asarray(weights, np.float64)
    x = np.asarray(x, np.float64)
    out_features, in_features = weights.shape
    x_mean = np.abs(x).mean(axis=0)
    magnitude = np.abs(weights)
    if group_size > 0:
        grouped = _pad_groups(magnitude, group_size)
        grouped = grouped.reshape(out_features, -1, group_size)
        normalized = grouped / (grouped.max(axis=-1, keepdims=True) + 1e-6)
        normalized = normalized.reshape(out_features, -1)[:, :in_features]
        w_mean = normalized.mean(axis=0)
    else:
        w_mean = (
            magnitude / (magnitude.max(axis=1, keepdims=True) + 1e-6)
        ).mean(axis=0)
    reference_output = x @ weights.T
    candidates, losses = [], []
    for i in range(n_grid):
        ratio = i / n_grid
        scales = x_mean**ratio / (w_mean ** (1 - ratio) + 1e-4)
        scales = np.maximum(scales, 1e-4)
        scales = scales / np.sqrt(scales.max() * scales.min())
        reconstructed = _keras_pseudo_quantize(weights * scales, group_size)
        reconstructed = reconstructed / scales
        losses.append(np.mean((reference_output - x @ reconstructed.T) ** 2))
        candidates.append(scales)
    return np.stack(candidates), np.array(losses)


def _reference_clip_errors(weights_scaled, x_scaled, group_size, n_grid=20):
    """llm-awq `auto_clip_layer` on every row of `x_scaled`.

    Returns the candidate bounds `[steps, out, n_groups]` and the
    per-(channel, group) partial-output error of each, in float64. Each
    candidate is quantized with Keras's float32 quantizer, so the test
    checks the error and the grid, not the quantizer: a bound that cuts
    both extremes of a group puts its zero point on a half-way tie, which
    a float64 quantizer can round the other way.
    """
    weights_scaled = _pad_groups(np.asarray(weights_scaled), group_size)
    x_scaled = _pad_groups(np.asarray(x_scaled), group_size)
    out_features, in_features = weights_scaled.shape
    group = group_size if group_size > 0 else in_features
    n_groups = in_features // group
    x_grouped = np.asarray(x_scaled, np.float64).reshape(-1, 1, n_groups, group)
    w_grouped = np.asarray(weights_scaled, "float32").reshape(
        1, out_features, n_groups, group
    )
    group_max = np.abs(w_grouped).max(axis=-1, keepdims=True)
    reference_output = (x_grouped * w_grouped).sum(axis=-1)
    bounds, errors = [], []
    for step in range(10):
        bound = group_max * np.float32(1 - step / n_grid)
        clipped = np.clip(w_grouped, -bound, bound)
        quantized = _fake_quantize_weights(
            clipped.reshape(out_features, in_features),
            group_size if n_groups > 1 else -1,
        )
        quantized = ops.convert_to_numpy(quantized).reshape(w_grouped.shape)
        output = (x_grouped * quantized).sum(axis=-1)
        errors.append(((output - reference_output) ** 2).mean(axis=0))
        bounds.append(bound[0, :, :, 0])
    return np.stack(bounds), np.stack(errors)


class AWQReferenceTest(testing.TestCase):
    @parameterized.named_parameters(
        ("per_channel", -1),
        ("grouped_8", 8),
        ("grouped_32", 32),
        ("ragged_24", 24),
    )
    def test_scale_search_matches_the_reference(self, group_size):
        # The scales come from the reference grid, and the layer's output
        # error of the chosen candidate over every calibration row is the
        # grid's minimum, computed here without the Hessian.
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((24, 64)).astype("float32")
        x = rng.standard_normal((512, 64)).astype("float32")
        x[:, :8] *= 4.0  # a few salient channels
        candidates, losses = _reference_scale_losses(weights, x, group_size)

        scales = ops.convert_to_numpy(
            awq_search_optimal_scales(
                weights,
                np.abs(x).mean(axis=0),
                _hessian(x),
                group_size=group_size,
            )
        )
        distances = np.abs(candidates - scales).max(axis=1)
        chosen = int(np.argmin(distances))
        self.assertLess(distances[chosen], 1e-4)
        self.assertLessEqual(losses[chosen], losses.min() * (1 + 1e-5))

    @parameterized.named_parameters(
        ("per_channel", -1),
        ("grouped_8", 8),
        ("grouped_16", 16),
        ("ragged_24", 24),
    )
    def test_clip_search_matches_the_reference(self, group_size):
        # For every output channel and group, the chosen bound's partial
        # output error over every calibration row is the grid's minimum.
        rng = np.random.default_rng(1)
        weights = rng.standard_normal((32, 64)).astype("float32")
        weights[::4] *= 5.0
        x = rng.standard_normal((512, 64)).astype("float32")
        awq_scales = np.abs(x).mean(axis=0) ** 0.5
        weights_scaled = weights * awq_scales
        bounds, errors = _reference_clip_errors(
            weights_scaled, x / awq_scales, group_size
        )

        clip_bound = awq_search_best_clip(
            weights_scaled, _hessian(x), awq_scales, group_size=group_size
        )
        clip_bound = ops.convert_to_numpy(clip_bound)[:, :, 0]
        n_groups = clip_bound.shape[1]
        self.assertEqual(clip_bound.shape, (32, n_groups))
        # Which grid step each bound is, then that step's error.
        steps = np.argmin(np.abs(bounds - clip_bound[None]), axis=0)
        chosen_bound = np.take_along_axis(bounds, steps[None], axis=0)[0]
        self.assertAllClose(chosen_bound, clip_bound, rtol=1e-6)
        chosen_error = np.take_along_axis(errors, steps[None], axis=0)[0]
        self.assertTrue(
            np.all(chosen_error <= errors.min(axis=0) * (1 + 1e-5) + 1e-12)
        )
        # Keras's grid is the reference's: ten steps of 0.05 from the max.
        self.assertEqual(awq._CLIP_GRID_POINTS, 20)
        self.assertEqual(int(awq._CLIP_MAX_SHRINK * awq._CLIP_GRID_POINTS), 10)
