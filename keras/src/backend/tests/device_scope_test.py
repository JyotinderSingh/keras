import numpy as np
import pytest

from keras.src import backend
from keras.src import ops
from keras.src import testing


class DeviceTest(testing.TestCase):
    @pytest.mark.skipif(
        not testing.tensorflow_uses_gpu(), reason="tf on GPU only"
    )
    def test_tf_device_scope(self):
        with backend.device("cpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)
        with backend.device("CPU:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)

        # When leaving the scope, the device should be back with gpu:0
        t = backend.ops.numpy.ones((2, 1))
        self.assertIn("GPU:0", t.device)

        # Also verify the explicit gpu device
        with backend.device("gpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertIn("GPU:0", t.device)

    @pytest.mark.skipif(not testing.jax_uses_gpu(), reason="jax on GPU only")
    def test_jax_device_scope(self):
        import jax

        with backend.device("cpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, jax.devices("cpu")[0])
        with backend.device("CPU:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, jax.devices("cpu")[0])

        # When leaving the scope, the device should be back with gpu:0
        t = backend.ops.numpy.ones((2, 1))
        self.assertEqual(t.device, jax.devices("gpu")[0])

        # Also verify the explicit gpu device
        with backend.device("gpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, jax.devices("gpu")[0])

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_invalid_jax_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device(123).__enter__()

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_torch_device_scope(self):
        import torch

        with backend.device("cpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))
        with backend.device("CPU:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))

        # When leaving the scope, the device should be back with gpu:0
        t = backend.ops.numpy.ones((2, 1))
        self.assertEqual(t.device, torch.device("cuda", 0))

        # Also verify the explicit gpu -> cuda conversion
        with backend.device("gpu:0"):
            t = backend.ops.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cuda", 0))

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_invalid_torch_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device(123).__enter__()

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_meta_device(self):
        import torch

        with torch.device("meta"):
            x = torch.ones(5)

        t = backend.ops.convert_to_tensor(x)

        if not torch.cuda.is_available():
            self.assertEqual(t.device, torch.device("cpu"))
        else:
            self.assertEqual(t.device, torch.device("cuda", 0))

    def _torch_variables(self):
        # A float variable and an integer one, on the active device.
        return (
            backend.Variable(np.array([[1, -2], [3, -4]], "float32")),
            backend.Variable(
                np.array([[1, 2], [3, 4]], "int32"),
                trainable=False,
                dtype="int32",
            ),
        )

    def _assert_torch_variable_follows_the_active_device(
        self, variable, int_variable
    ):
        # Torch operators that read a variable return tensors on the active
        # device, the one `keras.ops` creates tensors on, whichever device the
        # variable was created on.
        import torch
        import torch.nn.functional as F

        value = ops.convert_to_numpy(variable)
        int_value = ops.convert_to_numpy(int_variable)
        active = ops.zeros(()).device
        ones = ops.ones(value.shape)
        int_ones = ops.ones(value.shape, "int32")
        cases = {
            "t - v": (ones - variable, 1.0 - value),
            "t * v": (ones * variable, value),
            "F.linear(t, v)": (
                F.linear(ones, variable),
                np.ones(value.shape) @ value.T,
            ),
            "F.linear(t, weight=v)": (
                F.linear(ones, weight=variable),
                np.ones(value.shape) @ value.T,
            ),
            "F.relu(v)": (F.relu(variable), np.maximum(value, 0)),
            "F.log_softmax(v)": (
                F.log_softmax(variable, dim=-1),
                value - np.log(np.sum(np.exp(value), axis=-1, keepdims=True)),
            ),
            "-v": (-variable, -value),
            "+v": (+variable, value),
            "abs(v)": (abs(variable), np.abs(value)),
            "~v * t": (~int_variable * int_ones, ~int_value),
            "v[0]": (variable[0], value[0]),
            "torch.abs(v)": (torch.abs(variable), np.abs(value)),
            "torch.abs(input=v)": (torch.abs(input=variable), np.abs(value)),
            # A function name with an inner `_` is not an in-place function.
            "torch.index_select(v)": (
                torch.index_select(
                    variable, 0, torch.tensor([1, 0], device=active)
                ),
                value[[1, 0]],
            ),
            "-v * t": (-variable * ones, -value),
            # A function whose name ends in `__` is an operator, not one that
            # writes into its first argument.
            "Tensor.__rsub__(v, t)": (
                torch.Tensor.__rsub__(variable, ones),
                1.0 - value,
            ),
            # A variable that is the source of an in-place update of a tensor
            # is read on the active device.
            "t.add_(v)": (ones.clone().add_(variable), 1.0 + value),
            "t.add_(other=v)": (
                ones.clone().add_(other=variable),
                1.0 + value,
            ),
            "t.copy_(v)": (torch.zeros_like(ones).copy_(variable), value),
        }
        for name, (result, expected) in cases.items():
            self.assertEqual(result.device, active, msg=name)
            self.assertAllClose(result, expected, msg=name)

        # The gradient reaches the variable, on the device it lives on.
        variable.value.grad = None
        ops.sum(ones - variable).backward()
        grad = variable.value.grad
        self.assertEqual(grad.device, variable.value.device)
        self.assertAllClose(grad, -np.ones_like(value))

    def _assert_torch_in_place_calls_update_the_variable(self, variable):
        # A torch function that writes into a variable updates the variable on
        # the device it lives on, whichever device is active. A copy on the
        # active device would take the update and leave the variable unchanged.
        import torch
        import torch.nn.functional as F

        initial = ops.convert_to_numpy(variable)
        device = variable.value.device
        twos = torch.full(initial.shape, 2.0, device=device)
        cases = {
            "F.relu(v, inplace=True)": (
                lambda: F.relu(variable, inplace=True),
                np.maximum(initial, 0),
            ),
            "torch.relu_(v)": (
                lambda: torch.relu_(variable),
                np.maximum(initial, 0),
            ),
            "torch.relu_(input=v)": (
                lambda: torch.relu_(input=variable),
                np.maximum(initial, 0),
            ),
            "torch.nn.init.constant_(v, 3)": (
                lambda: torch.nn.init.constant_(variable, 3.0),
                np.full_like(initial, 3.0),
            ),
            "torch.add(t, t, out=v)": (
                lambda: torch.add(twos, twos, out=variable),
                np.full_like(initial, 4.0),
            ),
        }
        with torch.no_grad():
            for name, (call, expected) in cases.items():
                variable.assign(initial)
                call()
                self.assertAllClose(variable, expected, msg=name)
                self.assertEqual(variable.value.device, device, msg=name)

            if device != ops.zeros(()).device:
                # The variable is not moved, so operands on the active device
                # raise, as they do for any `out` tensor on another device.
                with self.assertRaises(RuntimeError):
                    torch.add(
                        ops.ones(initial.shape),
                        ops.ones(initial.shape),
                        out=variable,
                    )

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_variable_on_a_single_device(self):
        variable, int_variable = self._torch_variables()
        self._assert_torch_variable_follows_the_active_device(
            variable, int_variable
        )
        self._assert_torch_in_place_calls_update_the_variable(variable)

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_torch_variable_built_on_default_device_used_under_cpu(self):
        variable, int_variable = self._torch_variables()
        self.assertNotEqual(variable.value.device.type, "cpu")
        with backend.device("cpu"):
            self._assert_torch_variable_follows_the_active_device(
                variable, int_variable
            )
            self._assert_torch_in_place_calls_update_the_variable(variable)

    @pytest.mark.skipif(
        not testing.torch_uses_gpu(), reason="torch on GPU only"
    )
    def test_torch_variable_built_on_cpu_used_on_default_device(self):
        with backend.device("cpu"):
            variable, int_variable = self._torch_variables()
        self.assertEqual(variable.value.device.type, "cpu")
        self._assert_torch_variable_follows_the_active_device(
            variable, int_variable
        )
        self._assert_torch_in_place_calls_update_the_variable(variable)

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_variable_on_meta_device_is_not_materialized(self):
        import torch

        # A variable created under the meta device holds no data. Reading it
        # on another device must not replace it with uninitialized values.
        with backend.device("meta"):
            variable = backend.Variable(np.ones((2, 2), "float32"))
        self.assertEqual((-variable).device.type, "meta")
        self.assertEqual(abs(variable).device.type, "meta")
        self.assertEqual(torch.abs(variable).device.type, "meta")

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_variable_operators_do_not_break_the_graph(self):
        import torch
        import torch._dynamo as dynamo
        import torch.nn.functional as F

        # Dynamo traces the in-place check of `Variable.__torch_function__`
        # through `dynamo.assume_constant_result`. Without it, torch 2.14
        # breaks the graph at each operation that reads a variable.
        variable = backend.Variable(np.array([[1, -2], [3, -4]], "float32"))
        x = ops.ones((2, 2))
        explanation = dynamo.explain(
            lambda x: torch.where(x > 0, x, variable) + F.relu(variable)
        )(x)
        self.assertEqual(explanation.graph_break_count, 0)
