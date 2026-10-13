import warnings

import jax
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.backend.jax import trainer as jax_trainer
from keras.src.distribution import distribution_lib
from keras.src.quantizers.modes.common import ContiguousGroups
from keras.src.quantizers.quantization_config import Int4QuantizationConfig


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
class TrainerWriteBackTest(testing.TestCase):
    def test_trainer_write_back_runs_no_check(self):
        """The trainer writes back the state of its compiled step inside
        `skip_assign_checks()`, so no `_assign_check` runs, and no value is
        read on the host. The check below refuses the stored `g_idx`: a
        direct `assign` of it raises, and every trainer call passes."""
        layer = layers.Dense(8)
        model = models.Sequential([layers.Input((64,)), layer])
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=32))
        model.compile(loss="mse", optimizer="sgd")
        layer.g_idx._assign_check = ContiguousGroups(64, 1)
        g_idx = layer.g_idx.numpy()
        with self.assertRaisesRegex(ValueError, "k // 1"):
            layer.g_idx.assign(g_idx)
        x = np.random.default_rng(0).standard_normal((16, 64))
        y = np.zeros((16, 8))
        model.fit(x, y, batch_size=8, epochs=1, verbose=0)
        model.evaluate(x, y, batch_size=8, verbose=0)
        model.predict(x, batch_size=8, verbose=0)
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)
        model.predict_on_batch(x)
        self.assertAllEqual(layer.g_idx, g_idx)


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
@pytest.mark.multi_device
class JAXTrainerTest(testing.TestCase, parameterized.TestCase):
    def _make_distribution(self, dist_type):
        if dist_type == "data_parallel":
            return distribution_lib.DataParallel()
        devices = backend_dlib.list_devices()
        n = len(devices)
        mesh = distribution_lib.DeviceMesh((n,), ["model"], devices)
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])
        return distribution_lib.ModelParallel(layout_map=layout_map)

    # ----------------------------------------------------------------
    # Mixed-sharding warning tests
    # ----------------------------------------------------------------
    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_warns_when_model_built_outside_scope(self, dist_type):
        """Model built outside distribution -> mixed warning on compile."""
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created outside any distribution scope — weights are local.
        model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        for w in model.weights:
            self.assertIsInstance(
                w.value.sharding, jax.sharding.SingleDeviceSharding
            )

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertGreater(
                len(mixed),
                0,
                "Expected a mixed-sharding warning but none was raised",
            )
            msg = str(mixed[0].message)
            self.assertIn("SingleDeviceSharding", msg)
            self.assertIn("set_distribution", msg)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_no_warning_when_model_built_inside_scope(self, dist_type):
        """Model built inside distribution scope -> no warning."""
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created inside scope — weights get proper sharding.
        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertEqual(
                len(mixed),
                0,
                "Unexpected mixed-sharding warning when model is "
                "built inside scope",
            )

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_train_on_batch(self, dist_type):
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])
            model.compile(loss="mse", optimizer="adam")

            inputs = np.random.normal(size=(8, 16)).astype("float32")
            labels = np.random.normal(size=(8, units)).astype("float32")
            sw = np.random.uniform(size=(8,)).astype("float32")

            # With sample weight.
            model.train_on_batch(x=inputs, y=labels, sample_weight=sw)
            model.test_on_batch(x=inputs, y=labels, sample_weight=sw)

            # Without sample weight.
            model.train_on_batch(x=inputs, y=labels)
            model.test_on_batch(x=inputs, y=labels)
            model.predict_on_batch(x=inputs)
            model.fit(x=inputs, y=labels, epochs=1, verbose=0)
            model.evaluate(x=inputs, y=labels, verbose=0)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_jax_epoch_iterator_with_none_elements(self, dist_type):
        def generator():
            yield (np.ones((16, 32)), None)

        with self._make_distribution(dist_type).scope():
            iterator = jax_trainer.JAXEpochIterator(
                x=generator(), steps_per_epoch=1
            )

            epoch_iter = iterator._get_iterator()
            batch = next(epoch_iter)

        self.assertIsNone(batch[1])
        self.assertIsNotNone(batch[0])
