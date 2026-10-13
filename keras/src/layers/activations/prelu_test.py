import numpy as np
import pytest

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src import testing
from keras.src.layers.activations import prelu


class PReLUTest(testing.TestCase):
    def test_prelu(self):
        self.run_layer_test(
            prelu.PReLU,
            init_kwargs={
                "alpha_initializer": "zeros",
                "alpha_regularizer": "L1",
                "alpha_constraint": "MaxNorm",
                "shared_axes": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_prelu_correctness(self):
        def np_prelu(x, alpha):
            return (x > 0) * x + (x <= 0) * alpha * x

        inputs = np.random.randn(2, 10, 5, 3)
        prelu_layer = prelu.PReLU(
            alpha_initializer="glorot_uniform",
            alpha_regularizer="l1",
            alpha_constraint="non_neg",
            shared_axes=(1, 2),
        )
        prelu_layer.build(inputs.shape)

        weights = np.random.random((1, 1, 3))
        prelu_layer.alpha.assign(weights)
        ref_out = np_prelu(inputs, weights)
        self.assertAllClose(prelu_layer(inputs), ref_out)

    def _assert_alpha_follows_the_active_device(self, layer):
        inputs = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], "float32")
        active = ops.zeros(()).device
        outputs = layer(inputs)
        self.assertEqual(outputs.device, active)
        self.assertAllClose(
            outputs, np.where(inputs > 0, inputs, 0.25 * inputs)
        )

        # The gradient reaches `alpha`, on the device it lives on.
        ops.sum(outputs).backward()
        grad = layer.alpha.value.grad
        self.assertEqual(grad.device, layer.alpha.value.device)
        self.assertAllClose(grad, np.sum(np.minimum(inputs, 0), axis=0))

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_call_under_cpu_on_layer_built_on_default_device(self):
        layer = prelu.PReLU(alpha_initializer=initializers.Constant(0.25))
        layer.build((None, 3))
        self.assertNotEqual(layer.alpha.value.device.type, "cpu")
        with backend.device("cpu"):
            self._assert_alpha_follows_the_active_device(layer)

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_call_on_default_device_on_layer_built_on_cpu(self):
        layer = prelu.PReLU(alpha_initializer=initializers.Constant(0.25))
        with backend.device("cpu"):
            layer.build((None, 3))
        self.assertEqual(layer.alpha.value.device.type, "cpu")
        self._assert_alpha_follows_the_active_device(layer)
