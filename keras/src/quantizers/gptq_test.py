from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.api import models
from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.gptq import GPTQ
from keras.src.quantizers.gptq import _stable_permutation
from keras.src.quantizers.gptq import gptq_quantize_matrix
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_zero_point
from keras.src.testing.test_utils import named_product

VOCAB_SIZE = 1000
SEQ_LEN = 128
NUM_SAMPLES = 16
W_BITS = 4
NUM_CLASSES = 32

CALIBRATION_TEXT = """
GPTQ (Generative Pre-trained Transformer Quantization) is an advanced 
post-training quantization (PTQ) algorithm designed to compress large 
language models with minimal accuracy degradation. It addresses the 
challenge of reducing model size from high-precision formats like 
FP16 to low-bit integers (e.g., INT4, INT3) without the need for
expensive retraining. The algorithm operates on a layer-by-layer basis, 
treating the quantization of each weight matrix $W$ as a 
reconstruction problem. Its objective is to find a quantized weight 
matrix $\hat{W}$ that minimizes the mean squared error of the layer's 
output, formulated as $\arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$, 
where $X$ is a set of calibration inputs. GPTQ's primary innovation 
is its greedy, error-compensating quantization process, based on the 
Optimal Brain Quantizer (OBQ) framework. It quantizes weights one by 
one (or in small groups). After quantizing a single weight $w_q$ to 
its discrete value $\hat{w}_q$, it introduces a quantization error of 
$\delta = w_q - \hat{w}_q$. This error is then immediately compensated 
for by updating all remaining, unquantized weights in the layer. 
The update step is guided by second-order information, specifically 
the inverse of the Hessian matrix ($\mathbf{H}^{-1}$) of the layer's 
reconstruction loss. This inverse Hessian provides a measure of weight 
saliency and inter-dependencies. The update applied to the remaining 
weights is calculated based on $\delta$ and the corresponding entries 
in $\mathbf{H}^{-1}$, effectively propagating the error to less 
sensitive weights. This sequential compensation minimizes the 
cumulative error across the entire layer, allowing GPTQ to maintain 
high model fidelity, as measured by perplexity, even at aggressive 
bit-rates.
"""


def _get_mock_layer(layer_type, kernel_shape, rng):
    if layer_type == "Dense":
        layer = layers.Dense(units=kernel_shape[1])
        layer.build(input_shape=(None, kernel_shape[0]))
    elif layer_type == "EinsumDense":
        output_shape = (kernel_shape[1], kernel_shape[2])
        layer = layers.EinsumDense(
            equation="...h,hio->...io", output_shape=output_shape
        )
        dummy_input = rng.standard_normal(size=(1, 1, kernel_shape[0]))
        layer(dummy_input)
        layer.kernel.assign(
            rng.standard_normal(size=kernel_shape).astype("float32")
        )
    else:
        layer = layers.Layer()
    return layer


@pytest.mark.requires_trainable_backend
class GPTQTest(testing.TestCase):
    def test_initialization_with_dense_layer(self):
        rng = np.random.default_rng(seed=42)

        mock_layer = _get_mock_layer("Dense", kernel_shape=(64, 128), rng=rng)

        gptq_instance = GPTQ(mock_layer)
        self.assertEqual(gptq_instance.rows, 64)
        self.assertEqual(gptq_instance.columns, 128)
        self.assertEqual(gptq_instance.hessian.shape, (64, 64))

    def test_initialization_with_einsumdense_3d(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer(
            "EinsumDense", kernel_shape=(64, 4, 32), rng=rng
        )
        gptq_instance = GPTQ(mock_layer)
        self.assertEqual(gptq_instance.rows, 64)
        self.assertEqual(gptq_instance.columns, 4 * 32)
        self.assertEqual(gptq_instance.hessian.shape, (64, 64))

    def test_update_hessian(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        gptq_instance = GPTQ(mock_layer)
        batch1 = rng.standard_normal(size=(8, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(batch1)
        self.assertEqual(gptq_instance.num_samples, 8)
        H1 = np.copy(ops.convert_to_numpy(gptq_instance.hessian))
        batch2 = rng.standard_normal(size=(4, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(batch2)
        self.assertEqual(gptq_instance.num_samples, 12)
        H2 = np.copy(ops.convert_to_numpy(gptq_instance.hessian))
        self.assertFalse(np.allclose(H1, H2))

    def test_full_quantization_process(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        original_weights = np.copy(ops.convert_to_numpy(mock_layer.kernel))

        gptq_instance = GPTQ(
            mock_layer,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                weight_bits=4,
                symmetric=False,
                group_size=-1,
            ),
        )
        calibration_data = rng.standard_normal(size=(128, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(calibration_data)
        gptq_instance.quantize_and_correct_layer()

        quantized_weights = ops.convert_to_numpy(mock_layer.kernel)
        self.assertFalse(np.allclose(original_weights, quantized_weights))

        gptq_instance.free()
        self.assertIsNone(gptq_instance.hessian)

    def test_unsupported_layer_error(self):
        rng = np.random.default_rng(seed=42)
        unsupported_layer = _get_mock_layer(
            "Unsupported", kernel_shape=None, rng=rng
        )
        with self.assertRaisesRegex(TypeError, "Unsupported layer type"):
            GPTQ(unsupported_layer)

    def test_update_hessian_invalid_input(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        gptq_instance = GPTQ(mock_layer)
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            gptq_instance.update_hessian_with_batch(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            gptq_instance.update_hessian_with_batch(np.empty((0, 16)))
        with self.assertRaisesRegex(ValueError, "match input features"):
            bad_input = rng.standard_normal(size=(8, 99))
            gptq_instance.update_hessian_with_batch(bad_input)

    def test_streaming_equals_big_batch(self):
        """Tests that streaming updates match big batch updates."""
        # dummy inputs
        x = ops.array(np.random.randn(100, 7), "float32")

        # One-shot hessian update
        layer_1 = layers.Dense(5, use_bias=False)
        layer_1.build(input_shape=(None, 7))

        g1 = GPTQ(layer_1)
        g1.update_hessian_with_batch(x)

        # Streamed hessian update
        layer_2 = layers.Dense(5, use_bias=False)
        layer_2.build(input_shape=(None, 7))
        g2 = GPTQ(layer_2)
        g2.update_hessian_with_batch(x[:50])
        g2.update_hessian_with_batch(x[50:])

        # Both the one-shot and streamed hessian updates should match
        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_hessian_matches_closed_form(self):
        """Tests that the Hessian matches the closed-form solution."""
        x = ops.array(np.random.randn(128, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = GPTQ(layer)
        g.update_hessian_with_batch(x)

        expected = ops.multiply(
            ops.divide(2.0, x.shape[0]), ops.matmul(ops.transpose(x), x)
        )
        self.assertAllClose(g.hessian, expected, rtol=1e-6, atol=1e-6)

    def test_higher_rank_inputs_are_reshaped(self):
        """Tests that higher-rank inputs are reshaped correctly."""
        # x: [batch, time, feat]
        x = ops.array(np.random.randn(10, 4, 7), "float32")
        x_flat = ops.reshape(x, (-1, ops.shape(x)[-1]))

        layer1 = layers.Dense(5, use_bias=False)
        layer1.build((None, 7))
        g1 = GPTQ(layer1)
        g1.update_hessian_with_batch(x)

        layer2 = layers.Dense(5, use_bias=False)
        layer2.build((None, 7))
        g2 = GPTQ(layer2)
        g2.update_hessian_with_batch(x_flat)

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_raises_on_feature_mismatch(self):
        x = ops.array(np.random.randn(8, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 6))  # wrong in_features
        g = GPTQ(layer)

        with self.assertRaisesRegex(ValueError, "do not match input features"):
            g.update_hessian_with_batch(x)

        with self.assertRaisesRegex(ValueError, "cannot be None"):
            g.update_hessian_with_batch(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            g.update_hessian_with_batch(
                ops.array(np.empty((0, 7), dtype="float32"))
            )

    def test_num_samples_accumulates_correctly(self):
        """Tests that the number of samples is accumulated correctly when
        streaming updates are used."""
        x = ops.array(np.random.randn(64, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = GPTQ(layer)

        g.update_hessian_with_batch(x[:5])
        g.update_hessian_with_batch(x[5:30])
        g.update_hessian_with_batch(x[30:])

        self.assertEqual(g.num_samples, 64)

    def test_numeric_stability_large_values(self):
        """Tests numeric stability of hessian update with large input values."""
        x = ops.multiply(ops.array(np.random.randn(32, 7), "float32"), 1e6)
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))

        g = GPTQ(layer)
        g.update_hessian_with_batch(x)

        # Should be finite and symmetric
        self.assertTrue(ops.all(ops.isfinite(g.hessian)))
        self.assertTrue(ops.all(ops.equal(g.hessian, ops.transpose(g.hessian))))

    def test_einsumdense_2d_kernel_hessian_shape(self):
        x = layers.Input((7,))
        y = layers.EinsumDense("ab,bc->ac", output_shape=(5,))(x)
        model = keras.Model(x, y)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        g = GPTQ(einsum_dense_layer)

        # should infer rows==7
        self.assertEqual(ops.shape(g.hessian), (7, 7))

    def test_einsumdense_3d_kernel_streaming_equals_big_batch(self):
        """Tests that streaming updates to the Hessian are equivalent to a big
        batch update."""
        # Construct a tiny attention-like einsum with 3D kernel
        x = layers.Input((7,))
        qkv = layers.EinsumDense("bf,fhk->bhk", output_shape=(2, 3))(
            x
        )  # heads=2, head_dim=3
        model = keras.Model(x, qkv)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        x = ops.array(np.random.randn(50, 7), "float32")

        g1 = GPTQ(einsum_dense_layer)
        g1.update_hessian_with_batch(x)

        g2 = GPTQ(einsum_dense_layer)
        g2.update_hessian_with_batch(x[:20])
        g2.update_hessian_with_batch(x[20:])

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

def _direct_reference_dequantized(
    weights_T,
    inv_hessian,
    *,
    compute_scale_zero,
    per_channel,
    group_size,
    activation_order,
):
    """Direct, permutation-aware & grouping-aware baseline (no error feedback
    because inv_hessian is identity). Returns dequantized weights with shape
    [out_features, in_features].
    """
    in_features = ops.shape(weights_T)[1]

    # Optional activation-order permutation (mirrors impl)
    if activation_order:
        order_metric = ops.reciprocal(ops.add(ops.diagonal(inv_hessian), 1e-12))
        perm = _stable_permutation(order_metric)  # descending
        inv_perm = ops.argsort(perm)
        work_T = ops.take(weights_T, perm, axis=1)
    else:
        work_T = weights_T
        inv_perm = None

    out = ops.zeros_like(work_T)

    if not per_channel:
        # Per-tensor: one scalar (s,z) per column (broadcast across rows)
        for j in range(in_features):
            col2d = ops.expand_dims(work_T[:, j], 1)  # [out,1]
            s, z, m = compute_scale_zero(col2d)
            q = quantize_with_zero_point(col2d, s, z, m)
            dq = dequantize_with_zero_point(q, s, z)[:, 0]  # [out]
            out = ops.slice_update(out, (0, j), ops.expand_dims(dq, axis=1))
    elif group_size == -1:
        # Per-channel, no grouping: distinct (s,z) per row & column
        for j in range(in_features):
            col2d = ops.expand_dims(work_T[:, j], 1)  # [out,1]
            s, z, m = compute_scale_zero(col2d)       # [out,1]
            q = quantize_with_zero_point(col2d, s, z, m)
            dq = dequantize_with_zero_point(q, s, z)[:, 0]
            out = ops.slice_update(out, (0, j), ops.expand_dims(dq, axis=1))
    else:
        # Grouped per-channel: reuse (s,z) (shape [out,1]) across all cols in the group
        for g in range(0, in_features, group_size):
            g_end = min(g + group_size, in_features)
            group_slice = work_T[:, g:g_end]          # [out, group_len]
            s, z, m = compute_scale_zero(group_slice) # expect [out,1]
            for j in range(g, g_end):
                col2d = ops.expand_dims(work_T[:, j], 1)
                q = quantize_with_zero_point(col2d, s, z, m)
                dq = dequantize_with_zero_point(q, s, z)[:, 0]
                out = ops.slice_update(out, (0, j), ops.expand_dims(dq, axis=1))

    # Undo permutation to match original column order
    if activation_order:
        out = ops.take(out, inv_perm, axis=1)
    return out


class GPTQIdentityTest(testing.TestCase, parameterized.TestCase):
    BITS = 4
    QMIN = 0
    QMAX = (1 << BITS) - 1  # 15 for 4-bit

    @parameterized.named_parameters(
        dict(
            testcase_name="per_tensor_bs4",
            per_channel=False, group_size=-1, blocksize=4, activation_order=False
        ),
        dict(
            testcase_name="per_tensor_bs64",
            per_channel=False, group_size=-1, blocksize=64, activation_order=False
        ),
        dict(
            testcase_name="per_channel_nogroup_bs4",
            per_channel=True, group_size=-1, blocksize=4, activation_order=False
        ),
        dict(
            testcase_name="grouped8_bs4",
            per_channel=True, group_size=8, blocksize=4, activation_order=False
        ),
        dict(
            testcase_name="grouped8_bs64",
            per_channel=True, group_size=8, blocksize=64, activation_order=False
        ),
        dict(
            testcase_name="grouped8_activation_order",
            per_channel=True, group_size=8, blocksize=4, activation_order=True
        ),
    )
    def test_identity_inv_hessian_matches_direct_quantization(
        self, per_channel, group_size, blocksize, activation_order
    ):
        # ---- Test data ----
        in_features, out_features = 32, 128
        weights = ops.reshape(
            ops.linspace(-0.9, 1.1, in_features * out_features, dtype="float32"),
            (in_features, out_features),
        )
        weights_T = ops.transpose(weights)  # [out, in]
        inv_hessian = ops.eye(in_features, dtype="float32")  # identity

        # ---- Parameter function matching the path under test ----
        get_quant_params = partial(
            compute_quantization_parameters,
            bits=4,
            symmetric=False,
            per_channel=per_channel,
            group_size=group_size,
            weight=True,
        )

        # ---- System under test ----
        q_int, scale_map, zero_map = gptq_quantize_matrix(
            weights_T,
            inv_hessian,
            blocksize=blocksize,
            per_channel=per_channel,
            group_size=group_size,
            activation_order=activation_order,
            compute_scale_zero=get_quant_params,
        )
        got = dequantize_with_zero_point(q_int, scale_map, zero_map)

        # ---- Direct, grouping-aware baseline ----
        want = _direct_reference_dequantized(
            weights_T,
            inv_hessian,
            compute_scale_zero=get_quant_params,
            per_channel=per_channel,
            group_size=group_size,
            activation_order=activation_order,
        )

        # ---- Equality ----
        self.assertAllClose(got, want, rtol=1e-6, atol=1e-6)

        # 2) Dequantized matrix is close to original matrix.
        #    For uniform (nearest) quantization, |x - dq(x)| <= 0.5*scale elementwise
        #    (relaxed to 1.0*scale for edge bins where clamping may occur).
        # Build a full scale map for elementwise bounds.
        if not per_channel:
            # per-tensor: scale_map is [1, in]; broadcast to [out, in]
            scale_full = ops.broadcast_to(scale_map, ops.shape(weights_T))
        else:
            # per-channel (incl. grouped): test impl materializes full [out, in]
            scale_full = scale_map

        # Edge elements: q == QMIN or q == QMAX (possible clamp) ⇒ allow up to 1.0*scale.
        is_edge = ops.logical_or(
            ops.equal(q_int, self.QMIN), ops.equal(q_int, self.QMAX)
        )
        half_step = ops.multiply(scale_full, 0.5)
        bound = ops.where(is_edge, scale_full, half_step)  # [out, in]

        abs_err = ops.subtract(got, weights_T)     # [out, in]
        # Verify abs_err <= bound + tiny_numerical_slack
        slack = 1e-6
        violation = ops.subtract(abs_err, ops.add(bound, slack))
        # Expect max(violation) <= 0
        max_violation = ops.max(violation)
        self.assertLessEqual(float(max_violation.numpy().item()), 0.0 + 1e-12)

    def test_activation_order_permutation_is_undone(self):
        in_features, out_features = 8, 6
        layer = layers.Dense(out_features, use_bias=False)
        layer.build((None, in_features))
        weights = ops.array(
            np.random.randn(in_features, out_features), "float32"
        )
        layer.set_weights([weights])

        # generate a non-trivial order metric.
        diag = ops.linspace(10.0, 1.0, in_features, dtype="float32")
        diag = ops.random.shuffle(diag)
        H = ops.diag(diag)

        # Ensure it generates a non-trivial permutation
        perm = _stable_permutation(diag)
        self.assertFalse(ops.all(ops.equal(perm, ops.arange(in_features))))

        # Quantize with activation order
        g1 = GPTQ(
            layer,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                group_size=-1,
                activation_order=True,
            ),
        )
        g1.hessian = H
        g1.quantize_and_correct_layer()

        # Quantize without activation order
        layer2 = layers.Dense(out_features, use_bias=False)
        layer2.build((None, in_features))
        layer2.set_weights([ops.copy(weights)])

        g2 = GPTQ(
            layer2,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                group_size=-1,
                activation_order=False,
            ),
        )
        g2.hessian = H
        g2.quantize_and_correct_layer()

        # The weights should be identical since permutation is undone
        self.assertAllClose(layer.get_weights()[0], layer2.get_weights()[0])


def _get_sequence_classifier():
    """Transformer-based sequence classifier

    tokens -> Embedding -> Transformer -> GAP -> Dense(num_classes).
    """
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
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return models.Model(inputs, outputs)


def _get_simple_model():
    return models.Sequential([layers.Dense(10, input_shape=(5,))])


def _mean_kl(p, q):
    # Add small epsilon for numerical stability
    eps = 1e-8
    p = ops.clip(p, eps, 1.0)
    q = ops.clip(q, eps, 1.0)
    # Compute KL divergence
    # D_KL(P || Q) = sum(P * log(P / Q))
    return ops.mean(
        ops.sum(ops.multiply(p, ops.subtract(ops.log(p), ops.log(q))), axis=-1)
    )


def _top1_match_rate(a_logits, b_logits):
    """Calculates the top-1 match rate between two sets of logits.

    Formula: T = 1/N * sum(1{argmax(a_i) == argmax(b_i)})
    """
    return ops.mean(
        ops.equal(ops.argmax(a_logits, axis=-1), ops.argmax(b_logits, axis=-1))
    )


DATASETS = {
    "string_dataset": lambda: _string_dataset(
        CALIBRATION_TEXT, NUM_SAMPLES, SEQ_LEN
    ),
    "token_dataset": lambda: _token_dataset(NUM_SAMPLES, SEQ_LEN),
}

CONFIGS = {
    "default": {},
    "per_channel": {"group_size": -1, "per_channel": True},
    "act_order": {"activation_order": True},
    "symmetric": {"symmetric": True},
    "group_wise": {"group_size": 2},
}


def _pad_or_trim_1d(ids, length):
    """Pads or trims a 1D array to a specified length."""
    ids = ops.ravel(ops.array(ids, "int64"))
    if len(ids) < length:
        ids = ops.concatenate(
            [ids, ops.zeros(length - len(ids), dtype=ids.dtype)]
        )
    else:
        ids = ids[:length]
    return ids


def _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Tokenizes strings to char-IDs or passes through int arrays;
    outputs shape (1, seq_len)."""

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
    """Yields string slices"""
    rng = np.random.default_rng(seed=0)
    L = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = rng.integers(0, L) if L > 1 else 0
        yield long_text[start : start + sequence_length]


def _token_dataset(
    num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN, vocab_size=VOCAB_SIZE
):
    """Yields tokenized samples."""
    rng = np.random.default_rng(seed=0)
    for _ in range(num_samples):
        yield rng.integers(
            low=0, high=vocab_size, size=(1, sequence_length), dtype=np.int64
        )


@pytest.mark.requires_trainable_backend
@pytest.mark.skipif(
    backend.backend() == "torch",
    reason="torch gives low accuracy on CI, but works well locally",
)
class TestModelQuantization(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": dataset_id, "dataset": dataset}
                for dataset_id, dataset in DATASETS.items()
            ],
            [
                {"testcase_name": config_id, "config": config}
                for config_id, config in CONFIGS.items()
            ],
        )
    )
    def test_quantize_gptq_combinations(self, dataset, config):
        """Tests GPTQ quantization on a tiny transformer classifier.

        Validates classification performance of the quantized model
        with respect to the full-precision baseline.
        """
        rng = np.random.default_rng(seed=0)

        # Build the calibration set.
        calibration_set = list(
            dataset() if isinstance(dataset, Callable) else dataset
        )
        self.assertNotEmpty(calibration_set)

        # Build classifier and tokenizer
        model = _get_sequence_classifier()

        model.compile(run_eagerly=True)

        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build an eval batch drawn from the SAME distribution as calibration
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[rng.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Baseline logits
        y_ref = model.predict(x_eval)

        base_cfg = dict(
            dataset=calibration_set,
            tokenizer=tokenizer,
            weight_bits=W_BITS,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=32,
            symmetric=False,
            activation_order=False,
        )
        gptq_cfg = GPTQConfig(**{**base_cfg, **config})

        # Quantize
        model.quantize("gptq", config=gptq_cfg)

        # Post-quant logits
        y_q = model.predict(x_eval)

        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref, p_q = ops.softmax(y_ref), ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        self.assertGreaterEqual(
            top1_match, 0.6, f"Top-1 agreement too low: {top1_match:.3f}"
        )
        self.assertLessEqual(kl, 0.50, f"KL divergence too high: {kl:.3f}")

    @parameterized.named_parameters(
        {
            "testcase_name": "gptq_with_invalid_config",
            "mode": "gptq",
            "config": {"weight_bits": 4},
            "expected_exception": ValueError,
            "error_msg": "must be of type",
        },
        {
            "testcase_name": "non_gptq_with_unsupported_config",
            "mode": "int8",
            "config": GPTQConfig(dataset=["a"], tokenizer=lambda x: x),
            "expected_exception": ValueError,
            "error_msg": "only supported for 'gptq'",
        },
    )
    def test_quantize_scenarios(
        self, mode, config, expected_exception, error_msg
    ):
        model = _get_simple_model()
        with self.assertRaisesRegex(expected_exception, error_msg):
            model.quantize(mode, config=config)
