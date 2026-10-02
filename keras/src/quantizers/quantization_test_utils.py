"""Fixtures shared by the quantization tests.

`calibrate_layer` is the only place in the tests that constructs a
`Calibrator`, so a change to how a calibration mode quantizes one layer
is one edit here. `tiny_calibration_model` is the model the calibration
tests run `Model.quantize` on. `input_gradient` takes a layer's input
gradient on the active backend.

`LAYERS` is the table of quantizable layers that the conformance test
(`conformance_test.py`) runs, built from the einsum equations in
`EINSUM_EQUATIONS`. It holds the built-in layers and three third-party
layers: `PermutedDense` (a 2-D kernel whose geometry overrides
`contract`), `Pointwise1D` (a 3-D kernel) and `TokenTable` (a lookup).
The third-party layers use only the quantization protocol of
`keras.src.quantizers.geometry`, so they show that a layer outside Keras
is quantizable with the built-in modes.
"""

import numpy as np

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src.layers.layer import Layer
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes
from keras.src.quantizers.geometry import LookupGeometry
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.saving import serialization_lib

# The modes whose values come from a calibration pass.
CALIBRATION_MODES = ("gptq", "awq")

# --- Calibration ----------------------------------------------------------


def calibration_config(mode, **kwargs):
    """The config of calibration `mode`, without a dataset by default.

    Args:
        mode: `"gptq"` or `"awq"`.
        **kwargs: Arguments of the mode's config class.
    """
    kwargs.setdefault("dataset", None)
    kwargs.setdefault("tokenizer", None)
    return strategy_registry.get_strategy(mode).config_cls(**kwargs)


def calibrate_layer(layer, config, *batches, solve=True):
    """Calibrates a float layer in `config.mode` on `batches` of its inputs.

    This is the only place in the tests that constructs a `Calibrator`.
    The calibrator observes the batches. With `solve=True`, it then solves
    for the codes and swaps the float layer for the quantized one in one
    step. With `solve=False`, the layer stays float and the calibrator
    only observes, so a test can read the statistics or solve later.

    Args:
        layer: The float layer to calibrate.
        config: The config of a calibration mode.
        *batches: Inputs of the layer, observed in order.
        solve: Whether to solve and quantize the layer.

    Returns:
        The calibrator.
    """
    strategy = strategy_registry.get_strategy(config.mode)
    calibrator = strategy.calibrator_cls(strategy, layer, config)
    for batch in batches:
        calibrator.observe(batch)
    if solve:
        calibrator.quantize()
    return calibrator


def calibration_statistic(calibrator):
    """The statistic a calibrator accumulates from the layer's inputs.

    GPTQ's Hessian, or AWQ's mean activation magnitudes.
    """
    if calibrator.strategy.name == "gptq":
        return calibrator.hessian
    return calibrator.activation_magnitudes


def token_dataset(num_samples, sequence_length, vocab_size, rng):
    """`num_samples` random token batches of shape `(1, sequence_length)`."""
    return [
        rng.integers(0, vocab_size, (1, sequence_length), dtype=np.int32)
        for _ in range(num_samples)
    ]


def tiny_calibration_model(
    block_layers,
    *,
    vocab_size=48,
    sequence_length=16,
    embed_dim=8,
    head_units=4,
    dtype=None,
):
    """Embedding, one sequential block, mean pooling and a `Dense` head.

    Args:
        block_layers: The layers of the block, in order. The first one
            takes the `(batch, sequence_length, embed_dim)` embeddings.
        vocab_size: The number of tokens.
        sequence_length: The length of an input sequence.
        embed_dim: The embedding width.
        head_units: The number of outputs of the head.
        dtype: The dtype policy of every layer the function creates.

    Returns:
        `(model, structure)`, where `structure` is the model's
        `quantization_layer_structure`.
    """
    inputs = layers.Input((sequence_length,), dtype="int32")
    embedding = layers.Embedding(vocab_size, embed_dim, dtype=dtype)
    block = models.Sequential(block_layers)
    x = block(embedding(inputs))
    x = layers.GlobalAveragePooling1D(dtype=dtype)(x)
    model = models.Model(inputs, layers.Dense(head_units, dtype=dtype)(x))
    structure = {
        "pre_block_layers": [embedding],
        "sequential_blocks": [block],
    }
    return model, structure


def tiny_transformer_classifier(
    vocab_size=1000, sequence_length=128, num_classes=32
):
    """Embedding, one transformer block, mean pooling and a `Dense` head.

    The block has a multi-head attention and a two-layer feed-forward
    network, so its quantizable layers form several calibration stages.
    The model's layers are `[inputs, embedding, block, pooling, head]`.
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

    inputs = layers.Input(shape=(sequence_length,), dtype="int32")
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes)(x)
    return models.Model(inputs, outputs)


# --- Third-party quantizable layers ----------------------------------------

# The variables a projection stores per mode, in the order of the released
# formats. It is the spec of `Dense` and `EinsumDense`.
PROJECTION_SPEC = {
    None: ["kernel", "bias"],
    "ternary": ["kernel", "bias", "kernel_scale"],
    "int8": ["kernel", "bias", "kernel_scale"],
    "int4": ["kernel", "bias", "kernel_scale", "kernel_zero", "g_idx"],
    "float8": [
        "kernel",
        "bias",
        "inputs_scale",
        "inputs_amax_history",
        "kernel_scale",
        "kernel_amax_history",
        "outputs_grad_scale",
        "outputs_grad_amax_history",
    ],
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

# The variables a lookup stores per mode. It is the spec of `Embedding`.
LOOKUP_SPEC = {
    None: ["embeddings"],
    "int8": ["embeddings", "embeddings_scale"],
    "int4": ["embeddings", "embeddings_scale", "embeddings_zero", "g_idx"],
}


class ThirdPartyProjection(Layer):
    """A projection layer written against the quantization protocol only.

    A subclass gives the float kernel's shape and the geometry. This
    class holds the rest of the protocol: the attributes the strategies
    read (`kernel_shape`, `bias`, `activation`), a `build` that
    lets a mode create the weight storage, the `kernel` property, LoRA,
    the variable serialization spec, the two save/load one-liners and the
    `quantization_config` in the layer config.
    """

    def __init__(self, units, use_bias=True, quantization_config=None, **kw):
        super().__init__(**kw)
        self.units = units
        self.use_bias = use_bias
        self.activation = None
        self.lora_rank = None
        self.lora_alpha = None
        self.quantization_config = quantization_config

    def float_kernel_shape(self, input_shape):
        raise NotImplementedError

    def build(self, input_shape):
        self.kernel_shape = self.float_kernel_shape(input_shape)
        # A mode that owns the weight storage creates the kernel. A mode
        # that keeps the float kernel (float8) adds its variables after
        # the float weights, in the order that `quantize` gives.
        owns_weight_storage = self._strategy_owns_weight_storage()
        if owns_weight_storage:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        else:
            self._kernel = self.add_weight(
                name="kernel",
                shape=self.kernel_shape,
                initializer="glorot_uniform",
            )
        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer="random_normal"
            )
        if self.quantization_mode and not owns_weight_storage:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )

    @property
    def kernel(self):
        quantized_weight = self._quantized_weight()
        kernel = (
            self._kernel
            if quantized_weight is None
            else quantized_weight.unpack()
        )
        if self.lora_enabled:
            kernel = ops.add(
                kernel,
                (self.lora_alpha / self.lora_rank)
                * ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
            )
        return kernel

    def call(self, inputs):
        x = self._quantization_geometry().contract(inputs, self.kernel)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def enable_lora(self, rank, lora_alpha=None):
        self._check_lora_supported(self.quantization_mode)
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=self.kernel_shape[:-1] + (rank,),
            initializer="he_uniform",
            dtype="float32",
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel_shape[-1]),
            initializer="zeros",
            dtype="float32",
        )
        if self._quantized_weight() is None:
            self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = rank if lora_alpha is None else lora_alpha

    @property
    def variable_serialization_spec(self):
        return PROJECTION_SPEC

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")

    def get_config(self):
        return {
            **super().get_config(),
            "units": self.units,
            "use_bias": self.use_bias,
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config")
            )
        )
        return super().from_config(config)


class _PermutedInputsView(ContractionView):
    """A contraction view whose inputs go through a fixed permutation."""

    def __init__(self, permutation, kernel_shape, kernel_axes, **axes):
        super().__init__(kernel_shape, kernel_axes, **axes)
        self.permutation = permutation

    def inputs_to_view(self, inputs):
        return ops.take(
            super().inputs_to_view(inputs), self.permutation, axis=-1
        )


class PermutedGeometry(ProjectionGeometry):
    """A 2-D kernel contracted against permuted inputs."""

    def _permuted(self, inputs):
        return ops.take(inputs, self.layer.permutation, axis=-1)

    def contract(self, inputs, kernel):
        return ops.matmul(self._permuted(inputs), kernel)

    def contract_grad(self, upstream, float_kernel):
        gradient = ops.matmul(upstream, ops.transpose(float_kernel))
        return ops.take(gradient, self.layer.inverse_permutation, axis=-1)

    def add_lora_delta(self, inputs, x):
        return super().add_lora_delta(self._permuted(inputs), x)

    def contraction_view(self):
        return _PermutedInputsView(
            self.layer.permutation,
            self.weight_shape,
            self.kernel_axes,
            input_batch_axes=(),
            input_contracted_axes=(-1,),
        )


class PermutedDense(ThirdPartyProjection):
    """A `Dense` behind a fixed permutation of its input features."""

    def float_kernel_shape(self, input_shape):
        features = input_shape[-1]
        self.permutation = np.random.default_rng(7).permutation(features)
        self.inverse_permutation = np.argsort(self.permutation)
        return (features, self.units)

    def _quantization_geometry(self):
        return PermutedGeometry(self)


class PointwiseGeometry(ProjectionGeometry):
    """The `(1, in, out)` kernel of a pointwise convolution.

    A kernel of another layout than `(input_dim, units)` describes its
    axes in `kernel_axes`; the stored scale layout and the calibration
    view derive from it.
    """

    @property
    def kernel_axes(self):
        return KernelAxes(contracted=(0, 1), free=(2,))

    def contract(self, inputs, kernel):
        return ops.einsum("btc,kcd->btd", inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        return ops.einsum("btd,kcd->btc", upstream, float_kernel)

    def add_lora_delta(self, inputs, x):
        layer = self.layer
        if layer.lora_enabled:
            lora_x = ops.einsum("btc,kcr->btr", inputs, layer.lora_kernel_a)
            lora_x = ops.matmul(lora_x, layer.lora_kernel_b)
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, layer.compute_dtype)
        return x


class Pointwise1D(ThirdPartyProjection):
    """A `Conv1D` with `kernel_size=1`, its kernel in the conv layout."""

    def float_kernel_shape(self, input_shape):
        return (1, input_shape[-1], self.units)

    def _quantization_geometry(self):
        return PointwiseGeometry(self)


class TokenTable(Layer):
    """A lookup layer written against the quantization protocol only.

    It holds the attributes the strategies read (`input_dim`,
    `output_dim`), a `build` that lets a mode create the table, the
    `embeddings` property, LoRA, the variable serialization spec, the two
    save/load one-liners and the `quantization_config` in the layer
    config.
    """

    def __init__(self, input_dim, output_dim, quantization_config=None, **kw):
        super().__init__(**kw)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lora_rank = None
        self.lora_alpha = None
        self.quantization_config = quantization_config

    def build(self, input_shape=None):
        table_shape = (self.input_dim, self.output_dim)
        if self._strategy_owns_weight_storage():
            self.quantized_build(
                table_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        else:
            self._embeddings = self.add_weight(
                name="embeddings",
                shape=table_shape,
                initializer=initializers.RandomUniform(-1.0, 1.0),
            )

    @property
    def embeddings(self):
        quantized_weight = self._quantized_weight()
        table = (
            self._embeddings
            if quantized_weight is None
            else quantized_weight.unpack()
        )
        if self.lora_enabled:
            table = ops.add(
                table,
                (self.lora_alpha / self.lora_rank)
                * ops.matmul(self.lora_embeddings_a, self.lora_embeddings_b),
            )
        return table

    def call(self, inputs):
        inputs = ops.cast(inputs, "int32")
        outputs = ops.take(self.embeddings, inputs, axis=0)
        return ops.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.output_dim,)

    def enable_lora(self, rank, lora_alpha=None):
        self._check_lora_supported(self.quantization_mode)
        self._tracker.unlock()
        self.lora_embeddings_a = self.add_weight(
            name="lora_embeddings_a",
            shape=(self.input_dim, rank),
            initializer="he_uniform",
            dtype="float32",
        )
        self.lora_embeddings_b = self.add_weight(
            name="lora_embeddings_b",
            shape=(rank, self.output_dim),
            initializer="zeros",
            dtype="float32",
        )
        self._embeddings.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = rank if lora_alpha is None else lora_alpha

    @property
    def variable_serialization_spec(self):
        return LOOKUP_SPEC

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "embeddings")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "embeddings")

    def get_config(self):
        return {
            **super().get_config(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config")
            )
        )
        return super().from_config(config)

    def _quantization_geometry(self):
        return LookupGeometry(self)


class ReleasedProtocolLayer(Layer):
    """A third-party layer on the per-layer protocol of Keras 3.12-3.15.

    It has no geometry: it overrides `quantize`, `quantized_build` and
    `quantized_call` itself. `quantized_build` sets `_is_quantized`, then
    `quantize` assigns the quantized dtype policy.
    """

    def __init__(self, units=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), name="w")
        if self.quantization_mode:
            self.quantized_build(input_shape, mode=self.quantization_mode)

    def call(self, inputs):
        return ops.matmul(inputs, self.w)

    def quantized_build(self, input_shape, mode, config=None):
        self.scale = self.add_weight(
            shape=(self.units,), initializer="ones", trainable=False
        )
        self._is_quantized = True

    def quantized_call(self, inputs):
        return ops.multiply(ops.matmul(inputs, self.w), self.scale)

    def quantize(self, mode=None, type_check=True, config=None):
        self._check_quantize_args(mode, self.compute_dtype)
        self._tracker.unlock()
        self.quantized_build(None, mode)
        self._tracker.lock()
        if self.dtype_policy.quantization_mode is None:
            self.dtype_policy = f"{mode}_from_{self.dtype_policy.name}"

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# The objects a saved model of a third-party layer needs to load.
CUSTOM_OBJECTS = {
    cls.__name__: cls
    for cls in (PermutedDense, Pointwise1D, TokenTable, ReleasedProtocolLayer)
}

# --- Layer table -----------------------------------------------------------

# name -> (equation, output_shape, bias_axes, input_shape).
EINSUM_EQUATIONS = {
    # A 2-D kernel whose contracted axis leads.
    "einsum": ("abc,cd->abd", (None, 6), "d", (None, 3, 12)),
    # The keras-hub Gemma query projection: the contracted axis does not
    # lead the kernel.
    "einsum_permuted": ("btd,ndh->btnh", (None, 2, 3), "nh", (None, 3, 12)),
    # A mixture-of-experts down projection: the expert axis is shared by
    # the inputs, the kernel and the outputs.
    "einsum_batched": ("bec,ecd->bed", (4, 6), "d", (None, 4, 12)),
}


def _einsum_case(name):
    equation, output_shape, bias_axes, input_shape = EINSUM_EQUATIONS[name]

    def make(**kwargs):
        return layers.EinsumDense(
            equation,
            output_shape=output_shape,
            bias_axes=bias_axes,
            bias_initializer="random_normal",
            **kwargs,
        )

    return make, input_shape


class LayerCase:
    """One row of the layer table.

    Args:
        make: Creates the layer, unbuilt, from layer keyword arguments.
        input_shape: The build shape, or `None` for a lookup, which
            takes `(batch, 3)` token ids.
        weight_name: The spec entry of the quantized weight (`"kernel"`
            or `"embeddings"`), held at `_<weight_name>`.
        third_party: Whether the layer is defined outside Keras.
    """

    def __init__(self, make, input_shape, weight_name, third_party=False):
        self.make = make
        self.input_shape = input_shape
        self.weight_name = weight_name
        self.third_party = third_party

    @property
    def is_lookup(self):
        return self.weight_name == "embeddings"


LAYERS = {
    "dense": LayerCase(
        lambda **kw: layers.Dense(6, bias_initializer="random_normal", **kw),
        (None, 12),
        "kernel",
    ),
    **{
        name: LayerCase(*_einsum_case(name), "kernel")
        for name in EINSUM_EQUATIONS
    },
    "ternary_dense": LayerCase(
        lambda **kw: layers.TernaryDense(
            6, bias_initializer="random_normal", **kw
        ),
        (None, 12),
        "kernel",
    ),
    "embedding": LayerCase(
        lambda **kw: layers.Embedding(10, 8, **kw), None, "embeddings"
    ),
    "reversible_tied": LayerCase(
        lambda **kw: layers.ReversibleEmbedding(10, 8, tie_weights=True, **kw),
        None,
        "embeddings",
    ),
    "reversible_untied": LayerCase(
        lambda **kw: layers.ReversibleEmbedding(10, 8, tie_weights=False, **kw),
        None,
        "embeddings",
    ),
    "permuted_dense": LayerCase(
        lambda **kw: PermutedDense(6, **kw), (None, 12), "kernel", True
    ),
    "pointwise": LayerCase(
        lambda **kw: Pointwise1D(6, **kw), (None, 3, 12), "kernel", True
    ),
    "token_table": LayerCase(
        lambda **kw: TokenTable(10, 8, **kw), None, "embeddings", True
    ),
}


def build_layer(kind, **kwargs):
    """A built layer of the table's row `kind`."""
    case = LAYERS[kind]
    layer = case.make(**kwargs)
    if case.input_shape is None:
        layer.build()
    else:
        layer.build(case.input_shape)
    return layer


def layer_inputs(kind, rng, batch_size=4):
    """A batch of inputs for the table's row `kind`."""
    input_shape = LAYERS[kind].input_shape
    if input_shape is None:
        return rng.integers(0, 10, (batch_size, 3)).astype("int32")
    return rng.standard_normal((batch_size,) + input_shape[1:]).astype(
        "float32"
    )


def input_gradient(layer, x):
    """The gradient of `sum(layer(x))` with respect to `x`."""
    if backend.backend() == "jax":
        import jax  # Only this backend runs the branch.

        return np.asarray(jax.grad(lambda v: ops.sum(layer(v)))(x))
    if backend.backend() == "torch":
        import torch  # Only this backend runs the branch.

        v = torch.tensor(x, requires_grad=True)
        ops.sum(layer(v)).backward()
        return v.grad.detach().cpu().numpy()
    import tensorflow as tf  # Only this backend runs the branch.

    v = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(v)
        y = tf.reduce_sum(layer(v))
    return tape.gradient(y, v).numpy()


def policy_built(layer):
    """A new layer built from `layer`'s config, as a saved model reloads.

    The config carries the quantized dtype policy, so the new layer holds
    the mode's variables and no float weight. LoRA is left out: a merged
    save loads into a layer without it.
    """
    config = layer.get_config()
    config.pop("lora_rank", None)
    config.pop("lora_alpha", None)
    new = type(layer).from_config(config)
    shapes = layer._build_shapes_dict
    if shapes:
        new.build(shapes["input_shape"])
    else:
        new.build()
    return new
