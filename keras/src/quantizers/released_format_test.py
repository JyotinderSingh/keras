"""The released int8 and int4 stores load bit-exactly.

`RELEASED_STORES` holds what `save_own_variables` writes in a Keras 3.15.1
process for small `Dense`, `EinsumDense`, `Embedding` and
`ReversibleEmbedding` layers quantized in place to int8 and to int4 (per
channel, block size 2, and the default block size 128). A layer built from
the saved policy name must read each store by position, list
`layer.weights` in the released order, save the same entries at the same
positions, and give the outputs of a NumPy formula written from the
literals. The formulas fix the int4 packing (two signed codes per byte, the
low nibble first) and the scale forms.

The released range of the int4 projection stores (`Dense`, `EinsumDense`)
is v3.14.0 to v3.15.1: earlier releases pack those codes along the other
axis. Variable names and paths are not part of the format and are not
checked.
"""

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing

# Generated in Keras 3.15.1 on the JAX backend. Each layer gets float
# weights `0.5 * sin(1.7 * i + phase)` over the flat index `i` (phase 0.3
# for the kernel or table, 1.1 for the bias, 0.7 for the untied reverse
# table) and is quantized in place with `layer.quantize`: `"int8"`, or
# `"int4"` with `Int4QuantizationConfig(block_size=None)`
# (`int4_per_channel`), with `block_size=2` (`int4_block_2`), or with no
# config (`int4_default`, block size 128). For each row:
# - `policy`: the saved dtype policy name.
# - `weights`: the store position of each entry of `layer.weights`, for a
#   layer built from `policy`.
# - `values_only`: positions checked by value, not by dtype. Keras 3.15.1
#   stores the untied grouped reverse zero point as `float32`.
# - `store`: `(dtype, value)` per position.
# fmt: off
RELEASED_STORES = {
    ("dense", "int8"): {
        "policy": "int8_from_float32",
        "weights": [1, 0, 2],
        "store": [
            ("int8", [[43, 116, -69], [-112, 93, 76], [-127, -46, 127],
                      [16, -127, 20]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [288.7361, 254.05725, 261.31482]),
        ],
    },
    ("dense", "int4_per_channel"): {
        "policy": "int4/-1_from_float32",
        "weights": [1, 0, 2],
        "store": [
            ("int8", [[98, 12], [90, 4], [-39, 7], [-111, 1]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [15.91459, 14.003156, 14.403178]),
        ],
    },
    ("dense", "int4_block_2"): {
        "policy": "int4/2_from_float32",
        "weights": [1, 0, 2, 3, 4],
        "store": [
            ("int8", [[119, 8], [120, 7], [-8, 7], [-121, 11]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [[0.0356095, 0.006010952, 0.03715845],
                         [0.032914985, 0.021384845, 0.02740435]]),
            ("int8", [[3, -8, -1], [5, 7, -8]]),
            ("float32", [0.0, 0.0, 1.0, 1.0]),
        ],
    },
    ("dense", "int4_default"): {
        "policy": "int4/128_from_float32",
        "weights": [1, 0, 2, 3, 4],
        "store": [
            ("int8", [[119, 8], [105, 3], [-40, 7], [-124, 14]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [[0.039173875, 0.06363574, 0.050061464]]),
            ("int8", [[3, 0, -3]]),
            ("float32", [0.0, 0.0, 0.0, 0.0]),
        ],
    },
    ("dense_no_bias", "int8"): {
        "policy": "int8_from_float32",
        "weights": [0, 1],
        "store": [
            ("int8", [[43, 116, -69], [-112, 93, 76], [-127, -46, 127],
                      [16, -127, 20]]),
            ("float32", [288.7361, 254.05725, 261.31482]),
        ],
    },
    ("dense_no_bias", "int4_per_channel"): {
        "policy": "int4/-1_from_float32",
        "weights": [0, 1],
        "store": [
            ("int8", [[98, 12], [90, 4], [-39, 7], [-111, 1]]),
            ("float32", [15.91459, 14.003156, 14.403178]),
        ],
    },
    ("dense_no_bias", "int4_block_2"): {
        "policy": "int4/2_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[119, 8], [120, 7], [-8, 7], [-121, 11]]),
            ("float32", [[0.0356095, 0.006010952, 0.03715845],
                         [0.032914985, 0.021384845, 0.02740435]]),
            ("int8", [[3, -8, -1], [5, 7, -8]]),
            ("float32", [0.0, 0.0, 1.0, 1.0]),
        ],
    },
    ("dense_no_bias", "int4_default"): {
        "policy": "int4/128_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[119, 8], [105, 3], [-40, 7], [-124, 14]]),
            ("float32", [[0.039173875, 0.06363574, 0.050061464]]),
            ("int8", [[3, 0, -3]]),
            ("float32", [0.0, 0.0, 0.0, 0.0]),
        ],
    },
    ("einsum", "int8"): {
        "policy": "int8_from_float32",
        "weights": [1, 0, 2],
        "store": [
            ("int8", [[[39, 125, -69], [-99, 93, 75]],
                      [[-116, -49, 127], [14, -127, 19]],
                      [[127, -55, -112], [79, 89, -102]],
                      [[-65, 127, 33], [-127, 0, 127]]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [[[264.26596, 274.25357, 261.31482],
                          [256.10544, 254.05725, 256.16455]]]),
        ],
    },
    ("einsum", "int4_per_channel"): {
        "policy": "int4/-1_from_float32",
        "weights": [1, 0, 2],
        "store": [
            ("int8", [[114, -68, 69], [-38, 23, 25], [-41, 74, -91],
                      [124, -110, 112]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [14.565841, 15.116339, 14.403178, 14.116048, 14.003156,
                         14.119306]),
        ],
    },
    ("einsum", "int4_block_2"): {
        "policy": "int4/2_from_float32",
        "weights": [1, 0, 2, 3, 4],
        "store": [
            ("int8", [[119, -120, 119], [-120, 119, -40], [-121, 120, -121],
                      [120, -121, 120]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [[0.039173875, 0.042250894, 0.050061464, 0.02935061,
                          0.057624795, 0.01450134],
                         [0.048483133, 0.04412353, 0.03711299, 0.053687137,
                          0.023278402, 0.059685722]]),
            ("int8", [[3, -4, -3, 5, 1, -8], [-3, -3, 4, 1, -8, -1]]),
            ("float32", [0.0, 0.0, 1.0, 1.0]),
        ],
    },
    ("einsum", "int4_default"): {
        "policy": "int4/128_from_float32",
        "weights": [1, 0, 2, 3, 4],
        "store": [
            ("int8", [[113, -85, 71], [-104, 39, 8], [-121, 120, -121],
                      [123, -127, 113]]),
            ("float32", [0.44560367, 0.16749407, -0.48876506]),
            ("float32", [[0.061361622, 0.04412353, 0.061023824, 0.053687137,
                          0.057624795, 0.059685722]]),
            ("int8", [[-1, -3, -1, 1, 1, -1]]),
            ("float32", [0.0, 0.0, 0.0, 0.0]),
        ],
    },
    ("embedding", "int8"): {
        "policy": "int8_from_float32",
        "weights": [0, 1],
        "store": [
            ("int8", [[41, 127, -74, -108], [105, 84, -127, -52],
                      [123, 14, -127, 19], [127, -53, -113, 82],
                      [96, -110, -68, 127]]),
            ("float32", [279.3365, 288.7361, 254.05725, 264.26596, 274.25357]),
        ],
    },
    ("embedding", "int4_per_channel"): {
        "policy": "int4/-1_from_float32",
        "weights": [0, 1],
        "store": [
            ("int8", [[114, -84], [86, -39], [23, 25], [-41, 90], [-91, 124]]),
            ("float32", [15.3965, 15.91459, 14.003156, 14.565841, 15.116339]),
        ],
    },
    ("embedding", "int4_block_2"): {
        "policy": "int4/2_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[127, -120], [119, -40], [-89, 120], [-121, 120],
                      [-121, 120]]),
            ("float32", [[0.020459244, 0.00809762], [0.004801735, 0.017382223],
                         [0.028808469, 0.03832173], [0.045290288, 0.04925141],
                         [0.049942058, 0.047316376]]),
            ("int8", [[-8, 7], [-8, 7], [-8, 5], [-4, 1], [0, -3]]),
            ("float32", [0.0, 0.0, 1.0, 1.0]),
        ],
    },
    ("embedding", "int4_default"): {
        "policy": "int4/128_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[114, -118], [87, -40], [23, 24], [-57, 72],
                      [-123, 123]]),
            ("float32", [[0.056068737], [0.053622168], [0.06572607],
                         [0.060661998], [0.057505727]]),
            ("int8", [[-1], [0], [0], [-1], [-1]]),
            ("float32", [0.0, 0.0, 0.0, 0.0]),
        ],
    },
    ("reversible_untied", "int8"): {
        "policy": "int8_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[41, 127, -74, -108], [105, 84, -127, -52],
                      [123, 14, -127, 19], [127, -53, -113, 82],
                      [96, -110, -68, 127]]),
            ("float32", [279.3365, 288.7361, 254.05725, 264.26596, 274.25357]),
            ("int8", [[90, 86, -107, -60, 119], [31, -127, 4, 127, -37],
                      [-127, 67, 102, -93, -75], [122, 47, -127, -15, 127]]),
            ("float32", [278.33096, 255.16446, 261.8271, 257.4049, 254.02235]),
        ],
    },
    ("reversible_untied", "int4_per_channel"): {
        "policy": "int4/-1_from_float32",
        "weights": [0, 1, 2, 3],
        "store": [
            ("int8", [[114, -84], [86, -39], [23, 25], [-41, 90], [-91, 124]]),
            ("float32", [15.3965, 15.91459, 14.003156, 14.565841, 15.116339]),
            ("int8", [[37, -107, 10, 125, -25], [121, 52, -106, -5, 124]]),
            ("float32", [15.341077, 14.064183, 14.431415, 14.187673,
                         14.001232]),
        ],
    },
    ("reversible_untied", "int4_block_2"): {
        "policy": "int4/2_from_float32",
        "weights": [0, 1, 2, 3, 4, 5, 6],
        "values_only": [6],
        "store": [
            ("int8", [[127, -120], [119, -40], [-89, 120], [-121, 120],
                      [-121, 120]]),
            ("float32", [[0.020459244, 0.00809762], [0.004801735, 0.017382223],
                         [0.028808469, 0.03832173], [0.045290288, 0.04925141],
                         [0.049942058, 0.047316376]]),
            ("int8", [[-8, 7], [-8, 7], [-8, 5], [-4, 1], [0, -3]]),
            ("float32", [0.0, 0.0, 1.0, 1.0]),
            ("int8", [[7, -121, 120, 120, -121], [120, 119, -121, 72, 120]]),
            ("float32", [[0.014044266, 0.055696655, 0.028396677, 0.04837914,
                          0.04086345],
                         [0.05961546, 0.0052471105, 0.05826334, 0.02026093,
                          0.05304232]]),
            ("float32", [[-8.0, 1.0, 6.0, -3.0, -4.0],
                         [0.0, -8.0, 0.0, 7.0, -2.0]]),
        ],
    },
    ("reversible_untied", "int4_default"): {
        "policy": "int4/128_from_float32",
        "weights": [0, 1, 2, 3, 4, 5, 6],
        "values_only": [6],
        "store": [
            ("int8", [[114, -118], [87, -40], [23, 24], [-57, 72],
                      [-123, 123]]),
            ("float32", [[0.056068737], [0.053622168], [0.06572607],
                         [0.060661998], [0.057505727]]),
            ("int8", [[-1], [0], [0], [-1], [-1]]),
            ("float32", [0.0, 0.0, 0.0, 0.0]),
            ("int8", [[37, -121, 9, 122, -73], [120, 70, -121, -40, 120]]),
            ("float32", [[0.05961546, 0.055696655, 0.05826334, 0.0570089,
                          0.05304232]]),
            ("float32", [[0.0, 1.0, 0.0, -2.0, -2.0]]),
        ],
    },
}
# fmt: on

# A tied `ReversibleEmbedding` writes exactly the `Embedding` store.
RELEASED_STORES.update(
    {
        ("reversible_tied", variant): row
        for (kind, variant), row in list(RELEASED_STORES.items())
        if kind == "embedding"
    }
)

PROJECTIONS = {
    # kind: (equation, kernel shape)
    "dense": ("ab,bc->ac", (4, 3)),
    "dense_no_bias": ("ab,bc->ac", (4, 3)),
    "einsum": ("ab,bcd->acd", (4, 2, 3)),
}
GROUPED = ("int4_block_2", "int4_default")


def _sine(shape, phase):
    size = int(np.prod(shape))
    values = 0.5 * np.sin(1.7 * np.arange(size) + phase)
    return values.reshape(shape).astype("float32")


INPUTS = _sine((2, 4), 0.9)
TOKEN_IDS = np.array([[0, 3, 4], [1, 2, 2]], dtype="int32")
REVERSE_INPUTS = _sine((2, 4), 2.1)


def _build_layer(kind, dtype):
    if kind == "dense":
        layer = layers.Dense(3, dtype=dtype)
    elif kind == "dense_no_bias":
        layer = layers.Dense(3, use_bias=False, dtype=dtype)
    elif kind == "einsum":
        layer = layers.EinsumDense(
            "ab,bcd->acd", output_shape=(2, 3), bias_axes="d", dtype=dtype
        )
    elif kind == "embedding":
        layer = layers.Embedding(5, 4, dtype=dtype)
    else:
        layer = layers.ReversibleEmbedding(
            5, 4, tie_weights=kind == "reversible_tied", dtype=dtype
        )
    layer.build((None, 4) if kind in PROJECTIONS else None)
    return layer


def _released_store(row):
    return {
        str(position): np.array(value, dtype=dtype)
        for position, (dtype, value) in enumerate(row["store"])
    }


def _unpack_int4(packed, length, axis):
    """Unpacks two signed int4 codes per byte, the low nibble first."""
    packed = np.moveaxis(packed, axis, -1)
    low = np.bitwise_and(packed, 0x0F)
    low = np.where(low > 7, low - 16, low)
    high = np.right_shift(packed, 4)
    codes = np.stack([low, high], axis=-1)
    codes = codes.reshape(packed.shape[:-1] + (-1,))[..., :length]
    return np.moveaxis(codes, -1, axis)


def _dequantize(codes, scale, zero=None, g_idx=None, axis=0):
    """The real values of codes quantized along `axis`.

    Without a zero point, `scale` holds one multiplier per slice along
    `axis` and the value is `codes / scale`. With a zero point, `scale` and
    `zero` hold one entry per group along `axis`, `g_idx` names the group
    of each position, and the value is `(codes - zero) * scale`.
    """
    codes = codes.astype("float32")
    if zero is None:
        if scale.ndim < codes.ndim:
            scale = np.expand_dims(scale, axis)
        return codes / scale
    groups = g_idx.astype("int32")
    zero = np.take(zero, groups, axis=axis).astype("float32")
    return (codes - zero) * np.take(scale, groups, axis=axis)


def _int8_activations(x):
    """Rounds `x` to per-row abs-max int8 codes and back (W8A8)."""
    scale = np.float32(127.0) / (
        np.max(np.abs(x), axis=-1, keepdims=True) + np.float32(1e-7)
    )
    return np.clip(np.round(x * scale), -127, 127) / scale


def _expected_outputs(kind, variant, store):
    """The outputs of a layer that holds `store`, from NumPy alone.

    Returns the outputs and, for a `ReversibleEmbedding`, the reverse
    outputs (else `None`). The int4 reverse projection is weight-only.
    """
    entries = [store[str(position)] for position in range(len(store))]
    if kind in PROJECTIONS:
        equation, kernel_shape = PROJECTIONS[kind]
        kernel, *params = entries
        bias = params.pop(0) if kind != "dense_no_bias" else 0.0
        if variant == "int8":
            inputs = _int8_activations(INPUTS)
        else:
            inputs = INPUTS
            kernel = _unpack_int4(kernel, np.prod(kernel_shape[1:]), axis=1)
        kernel = _dequantize(kernel, *params, axis=0).reshape(kernel_shape)
        return np.einsum(equation, inputs, kernel) + bias, None

    num_forward = 4 if variant in GROUPED else 2
    forward, reverse = entries[:num_forward], entries[num_forward:]
    table, *params = forward
    if variant != "int8":
        table = _unpack_int4(table, 4, axis=1)
    table = _dequantize(table, *params, axis=1)
    outputs = table[TOKEN_IDS]
    if kind == "embedding":
        return outputs, None
    if kind == "reversible_tied":
        reverse_table = table.T
    else:
        reverse_table, *params = reverse
        if variant != "int8":
            reverse_table = _unpack_int4(reverse_table, 4, axis=0)
        if variant in GROUPED:
            params.append(forward[3])
        reverse_table = _dequantize(reverse_table, *params, axis=0)
    if variant == "int8":
        reverse_inputs = _int8_activations(REVERSE_INPUTS)
    else:
        reverse_inputs = REVERSE_INPUTS
    return outputs, reverse_inputs @ reverse_table


def _named_rows(variant=None, kinds=None):
    return [
        {"testcase_name": f"{k}_{v}", "kind": k, "variant": v}
        for k, v in RELEASED_STORES
        if (variant is None or v == variant) and (kinds is None or k in kinds)
    ]


class ReleasedFormatTest(testing.TestCase):
    def assertOutputsMatchFormula(self, layer, kind, variant, store):
        outputs, reverse_outputs = _expected_outputs(kind, variant, store)
        inputs = INPUTS if kind in PROJECTIONS else TOKEN_IDS
        # TPU matmuls run at bfloat16 precision by default, which moves the
        # int4 outputs away from the formula by up to about 1.3e-3.
        self.assertAllClose(
            layer(inputs), outputs, tpu_atol=1e-2, tpu_rtol=1e-2
        )
        if reverse_outputs is not None:
            self.assertAllClose(
                layer(REVERSE_INPUTS, reverse=True),
                reverse_outputs,
                tpu_atol=1e-2,
                tpu_rtol=1e-2,
            )

    @parameterized.named_parameters(_named_rows())
    def test_policy_built_layer_reads_released_store(self, kind, variant):
        if variant != "int8" and testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on Tensorflow GPU")
        row = RELEASED_STORES[(kind, variant)]
        store = _released_store(row)
        values_only = row.get("values_only", ())
        layer = _build_layer(kind, row["policy"])
        self.assertEqual(layer.dtype_policy.name, row["policy"])
        layer.load_own_variables(store)

        # `layer.weights` lists the store entries in the released order.
        self.assertLen(layer.weights, len(row["weights"]))
        for variable, position in zip(layer.weights, row["weights"]):
            expected = store[str(position)]
            self.assertAllEqual(variable, expected)
            if position not in values_only:
                self.assertEqual(
                    backend.standardize_dtype(variable.dtype),
                    expected.dtype.name,
                )

        # The layer saves the same entries at the same positions.
        saved = {}
        layer.save_own_variables(saved)
        self.assertEqual(list(saved), list(store))
        for key, expected in store.items():
            self.assertAllEqual(saved[key], expected)
            if int(key) not in values_only:
                self.assertEqual(
                    backend.standardize_dtype(saved[key].dtype),
                    expected.dtype.name,
                )

        self.assertOutputsMatchFormula(layer, kind, variant, store)

    @parameterized.named_parameters(
        _named_rows(
            "int4_per_channel",
            ("embedding", "reversible_tied", "reversible_untied"),
        )
    )
    def test_bare_int4_policy_reads_per_channel_lookup_store(
        self, kind, variant
    ):
        # Keras 3.13 saves per-channel int4 lookups under the bare policy
        # name, with the same stores as Keras 3.15.1.
        if testing.tensorflow_uses_gpu():
            self.skipTest("Segfault on Tensorflow GPU")
        store = _released_store(RELEASED_STORES[(kind, variant)])
        layer = _build_layer(kind, "int4_from_float32")
        layer.load_own_variables(store)
        self.assertOutputsMatchFormula(layer, kind, variant, store)
