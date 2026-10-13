"""The group index of a mode that fixes its groups (`ContiguousGroups`)."""

import os
import warnings

import h5py
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import distribution
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.legacy.saving import legacy_h5_format
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.modes.common import ContiguousGroups
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import PermutedDense
from keras.src.quantizers.quantization_test_utils import ThirdPartyProjection
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.testing import test_utils
from keras.src.testing.test_utils import named_product

K, N = 256, 64


class BaseLoadDense(ThirdPartyProjection):
    """A third-party projection with the base `save_own_variables` and
    `load_own_variables`, in place of the two one-liners of the spec. They
    save and load the variables in the order of tracking."""

    save_own_variables = layers.Layer.save_own_variables
    load_own_variables = layers.Layer.load_own_variables

    def float_kernel_shape(self, input_shape):
        return (input_shape[-1], self.units)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)


# name -> (mode, group size, layer class) of a layer whose mode fixes its
# groups.
FIXED = {
    "int4": ("int4", 64, layers.Dense),
    "awq": ("awq", 64, layers.Dense),
    "gptq_one_group": ("gptq", -1, layers.Dense),
    "int4_base_load": ("int4", 64, BaseLoadDense),
}


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return np.asarray(ops.convert_to_numpy(ops.cast(x, "float32")))


def _model(name, seed=0):
    """A functional model of one layer named "target", in `FIXED[name]`."""
    mode, group, layer_class = FIXED[name]
    inputs = layers.Input((K,))
    layer = layer_class(N, name="target")
    model = models.Model(inputs, layer(inputs))
    layer._kernel.assign(_random((K, N), seed))
    if mode == "int4":
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=group))
    else:
        config = calibration_config(mode, group_size=group)
        calibrate_layer(layer, config, _random((64, K), seed + 100))
    return model, layer


def _groups(layer):
    """The groups that the check of `layer.g_idx` accepts."""
    check = layer.g_idx._assign_check
    return (np.arange(check.length) // check.group_size).astype("float32")


def _other_groups(layer):
    """A `g_idx` that is not the fixed one: its groups in another order,
    or with one group, rows in the groups 1 to 3."""
    groups = _groups(layer)
    if not groups.any():
        other = groups.copy()
        other[:3] = [1, 2, 3]
        return other
    return np.random.default_rng(3).permutation(groups)


def _message(layer):
    return f"k // {layer.g_idx._assign_check.group_size}"


def _variables(layer):
    return [_numpy(v) for v in layer.weights]


def _stored_index(layer):
    """The key of `g_idx` in the store of a Keras-format save of `layer`."""
    if type(layer).load_own_variables is layers.Layer.load_own_variables:
        # The base save stores the variables in the order of tracking.
        tracked = layer._trainable_variables + layer._non_trainable_variables
        return [id(v) for v in tracked].index(id(layer.g_idx))
    return test_utils.serialized_variable_names(layer).index("g_idx")


def _keras_file(test, model, layer, g_idx):
    """A `.weights.h5` file of `model` whose `g_idx` holds `g_idx`, as a
    file that another tool wrote."""
    path = os.path.join(test.get_temp_dir(), "weights.weights.h5")
    model.save_weights(path)
    index = _stored_index(layer)
    with h5py.File(path, "r+") as f:
        for group in f["layers"].values():
            if group["vars"].attrs["name"] == layer.name:
                group["vars"][str(index)][...] = g_idx
    return path


def _legacy_file(test, model, layer, g_idx):
    """A legacy h5 weights file of `model` whose `g_idx` holds `g_idx`."""
    path = os.path.join(test.get_temp_dir(), "legacy.h5")
    with h5py.File(path, "w") as f:
        legacy_h5_format.save_weights_to_hdf5_group(f, model)
        f[f"{layer.name}/{layer.name}/g_idx"][...] = g_idx
    return path


def _layer_set_weights(test, model, layer, g_idx):
    weights = layer.get_weights()
    weights[[v.name for v in layer.weights].index("g_idx")] = g_idx
    layer.set_weights(weights)


def _model_set_weights(test, model, layer, g_idx):
    weights = model.get_weights()
    weights[[id(v) for v in model.weights].index(id(layer.g_idx))] = g_idx
    model.set_weights(weights)


def _set_state_tree(test, model, layer, g_idx):
    # A state tree with the one entry: its path is that of the variable.
    tree = g_idx
    for name in reversed(layer.g_idx.path.split("/")):
        tree = {name: tree}
    model.set_state_tree({"non_trainable_variables": tree})


def _load_weights(test, model, layer, g_idx):
    model.load_weights(_keras_file(test, model, layer, g_idx))


def _legacy_h5(test, model, layer, g_idx):
    with h5py.File(_legacy_file(test, model, layer, g_idx), "r") as f:
        legacy_h5_format.load_weights_from_hdf5_group(f, model)


def _keras_load(test, model, path, skip_mismatch=False):
    model.load_weights(path, skip_mismatch=skip_mismatch)


def _legacy_load(test, model, path, skip_mismatch=False):
    with h5py.File(path, "r") as f:
        legacy_h5_format.load_weights_from_hdf5_group(
            f, model, skip_mismatch=skip_mismatch
        )


def _legacy_load_by_name(test, model, path, skip_mismatch=False):
    with h5py.File(path, "r") as f:
        legacy_h5_format.load_weights_from_hdf5_group_by_name(
            f, model, skip_mismatch=skip_mismatch
        )


# loader -> (the function that writes a file of `model` whose `g_idx`
# holds a value, the load of that file).
LOADERS = {
    "keras": (_keras_file, _keras_load),
    "legacy": (_legacy_file, _legacy_load),
    "legacy_by_name": (_legacy_file, _legacy_load_by_name),
}


# The writes of a value into a variable that check it first.
WRITERS = {
    "layer_set_weights": _layer_set_weights,
    "model_set_weights": _model_set_weights,
    "set_state_tree": _set_state_tree,
    "load_weights": _load_weights,
    "legacy_h5": _legacy_h5,
}


class ContiguousGroupsTest(testing.TestCase):
    def assertVariables(self, layer, values):
        """Each variable of `layer` holds its value in `values`."""
        self.assertEqual(len(layer.weights), len(values))
        for variable, value in zip(layer.weights, values):
            self.assertAllEqual(_numpy(variable), value, variable.path)

    def test_check(self):
        """The check refuses other groups of the same shape. It passes a
        value of another shape, which the write refuses with its shape
        error, and it compares by value."""
        check = ContiguousGroups(6, 2)
        check(np.array([0, 0, 1, 1, 2, 2], "float32"))
        check(np.zeros((5,), "float32"))
        with self.assertRaisesRegex(ValueError, r"k // 2.*\[0.0, 1.0, 0.0"):
            check(np.array([0, 1, 0, 1, 2, 2], "float32"))
        self.assertEqual(check, ContiguousGroups(6, 2))
        self.assertEqual(hash(check), hash(ContiguousGroups(6, 2)))

    def test_fixed_and_free_group_indices(self):
        """The mode, the group size and the geometry decide whether the
        index has a check: int4 in groups, AWQ, GPTQ with one group per
        problem and a third-party projection do; GPTQ with several groups
        and stacked problems with a partial group do not."""
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
        for policy, kind, shape, expected in cases:
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
            check = layer.g_idx._assign_check if expected else None
            if expected is None:
                self.assertIsNone(
                    getattr(layer.g_idx, "_assign_check", None), policy
                )
                continue
            self.assertEqual(check, ContiguousGroups(*expected), policy)
            # A layer that nothing has written holds the fixed groups.
            self.assertAllEqual(layer.g_idx, _groups(layer))

    def test_calibration_writes_the_fixed_groups(self):
        """A calibration writes the `g_idx` of a fixed index, and the value
        it writes is the one that the check accepts. Stacked problems with
        a partial group show why they get no check: the calibrator numbers
        the groups of each problem after those of the problem before."""
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
            check = getattr(layer.g_idx, "_assign_check", None)
            self.assertEqual(check is not None, fixed, name)
            stored = _numpy(layer.g_idx)
            if fixed:
                self.assertAllEqual(stored, _groups(layer), name)
            else:
                contiguous = np.arange(stored.shape[0]) // group
                self.assertFalse(np.array_equal(stored, contiguous), name)

    @parameterized.named_parameters(
        named_product(mode=sorted(FIXED), writer=sorted(WRITERS))
    )
    def test_writes_of_other_groups_raise(self, mode, writer):
        """A write of other groups raises, and the layer and its outputs
        stay as they were. A write of the fixed groups passes."""
        model, layer = _model(mode)
        x = _random((4, K), 7)
        before, variables = _numpy(model(x)), _variables(layer)
        with self.assertRaisesRegex(ValueError, _message(layer)):
            WRITERS[writer](self, model, layer, _other_groups(layer))
        self.assertVariables(layer, variables)
        self.assertAllEqual(_numpy(model(x)), before)
        WRITERS[writer](self, model, layer, _groups(layer))
        self.assertAllEqual(_numpy(model(x)), before)

    @parameterized.named_parameters(
        named_product(mode=sorted(FIXED), loader=sorted(LOADERS))
    )
    def test_refused_load_keeps_the_layer(self, mode, loader):
        """Under `skip_mismatch=True`, a file with other groups warns, and
        each variable of the layer keeps its value: every check runs
        before the first write. The loads are the Keras format and the
        legacy h5 format, by topology and by name."""
        write_file, load = LOADERS[loader]
        source, source_layer = _model(mode)
        other = _other_groups(source_layer)
        path = write_file(self, source, source_layer, other)
        model, layer = _model(mode, seed=11)
        variables = _variables(layer)
        self.assertFalse(
            all(
                np.array_equal(a, b)
                for a, b in zip(variables, _variables(source_layer))
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load(self, model, path, skip_mismatch=True)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any(_message(layer) in m for m in messages), messages)
        self.assertVariables(layer, variables)
        with self.assertRaisesRegex(ValueError, _message(layer)):
            load(self, model, path)
        self.assertVariables(layer, variables)

    @parameterized.named_parameters(named_product(loader=sorted(LOADERS)))
    def test_group_size_mismatch_raises_the_shape_error(self, loader):
        """A file of int4 in groups of 64 loaded into int4 in groups of 32:
        the `kernel_scale` shapes differ, and that shape error raises, not
        the group error of `g_idx`. The layer keeps every value."""
        write_file, load = LOADERS[loader]
        source, source_layer = _model("int4")
        path = write_file(self, source, source_layer, _groups(source_layer))
        inputs = layers.Input((K,))
        layer = layers.Dense(N, name="target", dtype="int4/32_from_float32")
        model = models.Model(inputs, layer(inputs))
        variables = _variables(layer)
        with self.assertRaisesRegex(ValueError, "[Ss]hape") as raised:
            load(self, model, path)
        self.assertIn("(8, 64)", str(raised.exception))
        self.assertNotIn("k // ", str(raised.exception))
        self.assertVariables(layer, variables)

    @parameterized.named_parameters(
        ("awq", "awq/4/64"), ("gptq_one_group", "gptq/4/-1")
    )
    def test_uncalibrated_layer_round_trips(self, policy):
        """A layer built from its policy, which no calibration has written,
        holds the fixed groups, so it saves and loads back. A stored
        all-zeros `g_idx` of several groups raises."""
        inputs = layers.Input((K,))
        layer = layers.Dense(N, name="target", dtype=f"{policy}_from_float32")
        model = models.Model(inputs, layer(inputs))
        x = _random((4, K), 7)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded = saving.load_model(path)
        self.assertAllEqual(loaded.get_layer("target").g_idx, _groups(layer))
        self.assertAllEqual(_numpy(loaded(x)), _numpy(model(x)))
        if policy == "awq/4/64":
            path = _keras_file(self, model, layer, np.zeros((K,), "float32"))
            with self.assertRaisesRegex(ValueError, "k // 64"):
                model.load_weights(path)

    @pytest.mark.multi_device
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Uses a JAX device mesh."
    )
    def test_distributed_load_and_refused_write(self):
        """Under `DataParallel` and under `ModelParallel` with a sharded
        `g_idx`, a `.keras` file of int4 and AWQ loads and gives the same
        outputs, and a `set_weights` of other groups raises."""
        devices = distribution.list_devices()
        mesh = distribution.DeviceMesh((len(devices),), ["model"], devices)
        layout_map = distribution.LayoutMap(mesh)
        layout_map["target/g_idx"] = ("model",)
        layout_map["target/kernel"] = ("model", None)
        distributions = (
            distribution.DataParallel(devices=devices),
            distribution.ModelParallel(
                layout_map=layout_map, batch_dim_name=None
            ),
        )
        x = _random((4, K), 7)
        for mode in ("int4", "awq"):
            model, _ = _model(mode)
            expected = _numpy(model(x))
            path = os.path.join(self.get_temp_dir(), f"{mode}.keras")
            model.save(path)
            for strategy in distributions:
                with strategy.scope():
                    loaded = saving.load_model(path)
                    layer = loaded.get_layer("target")
                    # A sharded contraction sums in another order.
                    self.assertAllClose(
                        _numpy(loaded(x)), expected, atol=1e-4, rtol=1e-4
                    )
                    with self.assertRaisesRegex(ValueError, _message(layer)):
                        _layer_set_weights(
                            self, loaded, layer, _other_groups(layer)
                        )
                    self.assertAllEqual(layer.g_idx, _groups(layer))
