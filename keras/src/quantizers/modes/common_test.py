"""The group index of a mode that fixes its groups."""

import os

import h5py
import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.quantization_test_utils import PermutedDense
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.testing import test_utils

K, N = 256, 64


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return np.asarray(ops.convert_to_numpy(ops.cast(x, "float32")))


def _fixed_groups(length, group):
    """`k // group` for each of `length` positions `k`, as the variable
    stores it."""
    return (np.arange(length) // group).astype("float32")


def _keras_file(test, model, layer, g_idx):
    """A `.weights.h5` file of `model` whose `g_idx` holds `g_idx`, as a
    file that another tool wrote."""
    path = os.path.join(test.get_temp_dir(), "weights.weights.h5")
    model.save_weights(path)
    index = test_utils.serialized_variable_names(layer).index("g_idx")
    with h5py.File(path, "r+") as f:
        for group in f["layers"].values():
            if group["vars"].attrs["name"] == layer.name:
                group["vars"][str(index)][...] = g_idx
    return path


class FixedGroupsTest(testing.TestCase):
    def test_fixed_and_free_group_indices(self):
        """The mode, the group size and the geometry decide whether the
        view reads the stored `g_idx`. int4 in groups, AWQ, GPTQ with one
        group per problem and a third-party projection fix their groups:
        the index holds `k // G` and the view has no `g_idx`. GPTQ with
        several groups and stacked problems with a partial group read
        it."""
        dense = (None, K)
        experts = (None, 5, 2, K)
        cases = (
            ("int4/64", "dense", dense, (K, 64)),
            ("awq/4/64", "dense", dense, (K, 64)),
            ("gptq/4/-1", "dense", dense, (K, K)),
            ("gptq/4/512", "dense", dense, (K, 512)),
            ("gptq/4/64", "dense", dense, None),
            ("awq/4/64", "experts", experts, (2 * K, 64)),
            ("gptq/4/-1", "experts", experts, (2 * K, K)),
            ("gptq/4/64", "experts", experts, None),
            ("awq/4/96", "experts", experts, None),
            ("int4/64", "third_party", dense, (K, 64)),
            ("awq/4/64", "third_party", dense, (K, 64)),
        )
        for policy, kind, shape, fixed in cases:
            dtype = f"{policy}_from_float32"
            if kind == "dense":
                layer = layers.Dense(N, dtype=dtype)
            elif kind == "experts":
                layer = layers.EinsumDense(
                    "btei,eid->bted", (None, 2, N), dtype=dtype
                )
            else:
                layer = PermutedDense(N, dtype=dtype)
            layer.build(shape)
            view = layer._quantized_weight()
            if fixed is None:
                self.assertIs(view.g_idx, layer.g_idx, policy)
                continue
            self.assertIsNone(view.g_idx, policy)
            # A layer that nothing has written holds the fixed groups.
            self.assertAllEqual(layer.g_idx, _fixed_groups(*fixed), policy)

    def test_calibration_writes_the_fixed_groups(self):
        """A calibration writes the `g_idx` of a fixed index, and the value
        it writes is the fixed one, `k // G`. Stacked problems with a
        partial group show why their view reads the index: the calibrator
        numbers the groups of each problem after those of the problem
        before."""
        cases = (
            ("awq", 64, "dense", True),
            ("gptq", -1, "dense", True),
            ("gptq", 512, "dense", True),
            ("awq", 64, "third_party", True),
            ("awq", 64, "experts", True),
            ("gptq", -1, "experts", True),
            ("awq", 96, "experts", False),
        )
        for mode, group, kind, fixed in cases:
            name = f"{mode}/{group}/{kind}"
            if kind == "experts":
                layer = layers.EinsumDense("btei,eid->bted", (None, 2, N))
                shape = (5, 2, K)
            elif kind == "third_party":
                layer, shape = PermutedDense(N), (K,)
            else:
                layer, shape = layers.Dense(N), (K,)
            layer.build((None,) + shape)
            layer._kernel.assign(_random(layer._kernel.shape))
            config = calibration_config(mode, group_size=group)
            calibrate_layer(layer, config, _random((16,) + shape, 2))
            view = layer._quantized_weight()
            self.assertEqual(view.g_idx is None, fixed, name)
            stored = _numpy(layer.g_idx)
            groups = _fixed_groups(stored.shape[0], view.scheme.group_size)
            if fixed:
                self.assertAllEqual(stored, groups, name)
            else:
                self.assertFalse(np.array_equal(stored, groups), name)

    @parameterized.named_parameters(
        ("awq", "awq", 64), ("gptq_one_group", "gptq", -1)
    )
    def test_uncalibrated_layer_round_trips(self, mode, group):
        """A layer built from its policy, which no calibration has written,
        holds the fixed groups, so it saves and loads back. A file whose
        `g_idx` holds other values (all zeros, or the groups in another
        order) loads too, and gives the outputs of the model that wrote
        it: no path reads the stored index of a fixed mode."""
        dtype = f"{mode}/4/{group}_from_float32"

        def model():
            inputs = layers.Input((K,))
            layer = layers.Dense(N, name="target", dtype=dtype)
            return models.Model(inputs, layer(inputs)), layer

        x = _random((4, K), 7)
        saved, layer = model()
        fixed = _fixed_groups(K, K if group == -1 else group)
        self.assertAllEqual(layer.g_idx, fixed)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        saved.save(path)
        loaded = saving.load_model(path)
        self.assertAllEqual(loaded.get_layer("target").g_idx, fixed)
        self.assertAllEqual(_numpy(loaded(x)), _numpy(saved(x)))

        float_layer = layers.Dense(N, name="target")
        float_layer.build((None, K))
        float_layer._kernel.assign(_random((K, N), 3))
        config = calibration_config(mode, group_size=group)
        calibrate_layer(float_layer, config, _random((16, K), 5))
        inputs = layers.Input((K,))
        source = models.Model(inputs, float_layer(inputs))
        expected = _numpy(source(x))
        rng = np.random.default_rng(3)
        for other in (np.zeros((K,), "float32"), rng.permutation(fixed)):
            path = _keras_file(self, source, float_layer, other)
            loaded, layer = model()
            loaded.load_weights(path)
            self.assertAllEqual(layer.g_idx, other)
            self.assertAllEqual(_numpy(loaded(x)), expected)
