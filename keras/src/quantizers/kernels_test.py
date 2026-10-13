"""Route selection for a 4-bit weight-only contraction (`kernels.py`).

The library has no kernel route yet. The tests run the reference path as
a route (`force_route("reference_wrapped")`), and `matrix`, a route of
this file that contracts the `(M, K)` inputs with the stored matrix.
"""

import contextlib
import math
import os
import platform
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import distribution
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import AutocastScope
from keras.src.layers.core.einsum_dense import EinsumAxes
from keras.src.quantizers import kernels
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import PermutedDense
from keras.src.quantizers.quantization_test_utils import Pointwise1D
from keras.src.quantizers.quantization_test_utils import ThirdPartyProjection
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing.test_utils import named_product

if backend.backend() == "jax":
    import jax
elif backend.backend() == "torch":
    import torch

# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
# Each equation is reshape-compatible.
ROUTABLE = {
    "dense": (None, 96, (3, 256)),
    "dense_ragged": (None, 48, (2, 3, 200)),
    "qkv_dnh": ("btd,dnh->btnh", (None, 4, 16), (2, 5, 128)),
    "query_ndh": ("btd,ndh->btnh", (None, 4, 16), (2, 5, 128)),
    "output_nhd": ("btnh,nhd->btd", (None, 96), (2, 5, 4, 32)),
    "expert_gate": ("btd,edi->btei", (None, 3, 16), (2, 5, 128)),
    "contracted_first": ("abc,cde->abde", (5, 4, 16), (2, 5, 128)),
    "two_contracted_apart": ("abcd,ced->abe", (None, 32), (2, 4, 8, 16)),
    "outer_product": ("bt,nh->btnh", (None, 4, 6), (2, 5)),
}
# Equations that gate 3 refuses.
REFUSED = {
    "batch_axis": ("ebd,edf->ebf", (None, 16), (3, 2, 128)),
    "summed_input_axis": ("abc,cd->ad", (16,), (2, 3, 128)),
    "contracted_out_of_order": ("abcd,edc->abe", (None, 32), (2, 4, 8, 16)),
}


class StandardProjection(ThirdPartyProjection):
    """A third-party layer with the standard contraction."""

    def float_kernel_shape(self, input_shape):
        return (input_shape[-1], self.units)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)


def _matrix(layer, geometry, quantized_weight, inputs, config):
    """The `matrix` route: `(M, K)` inputs times the stored matrix."""
    kernel = quantized_weight.dequantize(layer.compute_dtype)
    if quantized_weight.permutation is not None:
        kernel = ops.transpose(kernel, quantized_weight.permutation)
    return ops.matmul(inputs, ops.reshape(kernel, (inputs.shape[-1], -1)))


def _untested():
    """A route whose library version is outside its tested range."""
    return kernels.RouteSource(
        _matrix, version=lambda: "9.0.0", tested_versions=("1.2",)
    )


@contextlib.contextmanager
def _routes(rows=None, required=None, **sources):
    """Adds `matrix` and `sources` to the route table, `rows` to the
    allow-list, and `required` as this backend's `"require"` kernels."""
    sources = {"matrix": kernels.RouteSource(_matrix), **sources}
    rows = rows or {}
    kernels._ROUTES.update(sources)
    kernels._ALLOW_LIST.update(rows)
    if required:
        kernels._REQUIRED[backend.backend()] = required
    try:
        # Only `"auto"` reads the allow-list, and a test runner can set
        # `"require"` for the process. So the block runs under `"auto"`,
        # and the policy is as before when the block ends. A block inside
        # this block can set a different policy.
        with kernels.policy("auto"):
            yield
    finally:
        for name in sources:
            del kernels._ROUTES[name]
        for key in rows:
            del kernels._ALLOW_LIST[key]
        kernels._REQUIRED.pop(backend.backend(), None)


def _device_type():
    """The device type of `list_devices`: `"cpu"`, `"gpu"`, `"tpu"`..."""
    if backend.backend() == "numpy":
        return "cpu"
    return distribution.list_devices()[0].split(":")[0]


def _row_key(dtype, bucket):
    """The allow-list key of this backend and device."""
    device = None
    if backend.backend() == "torch":
        device = str(ops.convert_to_tensor(np.ones(1)).device)
    return (backend.backend(), kernels._device_class(device), dtype, bucket)


def _layer(case, block_size=32, dtype="float32", bias=False):
    """A layer of `case`, quantized to int4 from a random float kernel."""
    equation, output, input_shape = case
    if equation is None:
        layer = layers.Dense(output, use_bias=bias, dtype=dtype)
    else:
        bias_axes = equation[-1] if bias else None
        layer = layers.EinsumDense(
            equation, output, bias_axes=bias_axes, dtype=dtype
        )
    layer.build(input_shape)
    rng = np.random.default_rng(0)
    layer._kernel.assign(rng.standard_normal(layer._kernel.shape))
    config = Int4QuantizationConfig(block_size=block_size)
    layer.quantize("int4", config=config)
    return layer


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _oracle(layer, x):
    view = layer._quantized_weight()
    scale = ops.convert_to_numpy(layer.kernel_scale)
    if layer.compute_dtype == "bfloat16":
        # The layer reads its scale in the compute dtype.
        scale = scale.astype(ml_dtypes.bfloat16)
    grouped = layer.g_idx is not None
    return oracle_contract(
        x,
        ops.convert_to_numpy(layer._kernel),
        scale,
        columns=view.layout.orig_len,
        kernel_shape=view.shape,
        permutation=view.permutation,
        zero_point=ops.convert_to_numpy(layer.kernel_zero) if grouped else None,
        g_idx=ops.convert_to_numpy(layer.g_idx) if grouped else None,
        equation=getattr(layer, "equation", None),
        dtype=layer.compute_dtype,
    )


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _direct(layer, x):
    """A direct call of the selected route, on the tensors the layer reads."""
    geometry = layer._quantization_geometry()
    with AutocastScope(layer.compute_dtype):
        view = layer._quantized_weight()
        view = view.with_tensors(view.read_tensors())
        inputs = ops.cast(x, layer.compute_dtype)
        route = kernels.select_route(layer, geometry, view, inputs)
        return kernels.run_route(route, layer, geometry, view, inputs)


def _counts(layer, x):
    """The routes that one call of `layer` selects."""
    kernels.route_counts.clear()
    layer(x)
    return dict(kernels.route_counts)


class ReshapeCompatibleTest(testing.TestCase):
    @parameterized.named_parameters(
        ("matmul", "ab,bc->ac", 2, 1),
        ("qkv", "btd,dnh->btnh", 3, 1),
        ("query", "btd,ndh->btnh", 3, 1),
        ("output", "btnh,nhd->btd", 4, 2),
        ("expert_gate", "btd,edi->btei", 3, 1),
        ("contracted_first", "abc,cde->abde", 3, 1),
        ("two_contracted_apart", "abcd,ced->abe", 4, 2),
        ("outer_product", "bt,nh->btnh", 2, 0),
        ("left_ellipsis", "...b,bc->...c", 3, 1),
        ("batch_axis", "ebd,edf->ebf", 3, None),
        ("summed_input_axis", "abc,cd->ad", 3, None),
        ("contracted_out_of_order", "abcd,edc->abe", 4, None),
        ("outputs_permuted", "btd,dnh->bnth", 3, None),
        ("kernel_free_permuted", "btd,dnh->bthn", 3, None),
    )
    def test_verdict(self, equation, rank, contracted):
        axes = EinsumAxes.from_equation(equation, rank)
        self.assertEqual(kernels.reshape_compatible(axes), contracted)


class RouteSuiteTest(testing.TestCase):
    """Each route against the float64 oracle and the reference path."""

    @parameterized.named_parameters(
        named_product(
            case=sorted(ROUTABLE),
            block_size=[32, 128, -1],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    def test_routes(self, case, block_size, dtype):
        layer = _layer(ROUTABLE[case], block_size, dtype)
        x = _random(ROUTABLE[case][2])
        oracle = _oracle(layer, x)
        with kernels.policy("off"):
            reference = _numpy(layer(x))
        floor, bound = (1e-3, 2**-7) if dtype != "float32" else (1e-6, 1e-5)
        self.assertLessEqual(_err(reference, oracle), bound)
        with _routes():
            for name in ("reference_wrapped", "matrix"):
                kernels.route_counts.clear()
                with kernels.force_route(name):
                    y, again, direct = layer(x), layer(x), _direct(layer, x)
                self.assertEqual(dict(kernels.route_counts), {name: 3})
                y = _numpy(y)
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertAllEqual(_numpy(again), y)
                self.assertAllEqual(_numpy(direct), y)
                err = _err(y, oracle)
                self.assertLessEqual(
                    err, 1.25 * _err(reference, oracle) + floor
                )
                self.assertLessEqual(err, bound)
                if name == "reference_wrapped":
                    self.assertAllEqual(y, reference)

    @parameterized.named_parameters(
        named_product(
            case=["dense", "query_ndh", "output_nhd", "two_contracted_apart"],
            block_size=[32, -1],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    def test_exact_data(self, case, block_size, dtype):
        """Small integers and power-of-two scales give the oracle exactly."""
        layer = _layer(ROUTABLE[case], block_size, dtype)
        view = layer._quantized_weight()
        rng = np.random.default_rng(2)
        rows, columns = view.codes.shape[0], view.layout.orig_len
        codes = rng.integers(-1, 2, (rows, columns)).astype("int8")
        layer._kernel.assign(pack_int4(codes, axis=-1)[0])
        powers = rng.integers(1, 4, layer.kernel_scale.shape)
        # A grouped scale multiplies and a per-channel scale divides.
        sign = -1 if layer.g_idx is not None else 1
        layer.kernel_scale.assign(2.0 ** (sign * powers))
        if layer.kernel_zero is not None:
            zero = rng.integers(-1, 2, layer.kernel_zero.shape)
            layer.kernel_zero.assign(zero.astype("int8"))
        # Eight nonzero inputs per row keep every sum exact in bfloat16.
        input_shape = ROUTABLE[case][2]
        x = np.zeros((math.prod(input_shape) // rows, rows), "float32")
        for row in x:
            row[rng.choice(rows, 8, replace=False)] = rng.choice([-1, 1], 8)
        x = x.reshape(input_shape)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        with _routes():
            for name in ("reference_wrapped", "matrix"):
                with kernels.force_route(name):
                    self.assertAllEqual(_numpy(layer(x)), oracle)

    @parameterized.named_parameters(
        named_product(
            case=["dense", "query_ndh", "two_contracted_apart"],
            block_size=[32, -1],
        )
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy has no gradients."
    )
    def test_input_gradient(self, case, block_size):
        """A route replaces the forward only, so the gradient is the same.

        The bias and LoRA stay outside the route.
        """
        layer = _layer(ROUTABLE[case], block_size, bias=True)
        layer.enable_lora(4)
        rng = np.random.default_rng(3)
        layer.lora_kernel_b.assign(rng.random(layer.lora_kernel_b.shape))
        x = _random(ROUTABLE[case][2])
        with kernels.policy("off"):
            expected = input_gradient(layer, x)
        with _routes():
            for name in ("reference_wrapped", "matrix"):
                with kernels.force_route(name):
                    self.assertAllEqual(input_gradient(layer, x), expected)


class FallbackTest(testing.TestCase):
    """Each gate refusal runs the reference path."""

    def assertReference(self, layer, x):
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters([(name, name) for name in REFUSED])
    def test_refused_equation(self, name):
        self.assertReference(_layer(REFUSED[name]), _random(REFUSED[name][2]))

    @parameterized.named_parameters(
        ("custom_contract", PermutedDense), ("custom_kernel_axes", Pointwise1D)
    )
    def test_custom_geometry(self, cls):
        layer = cls(8, use_bias=False)
        layer.build((None, 3, 64))
        layer.quantize("int4")
        self.assertReference(layer, _random((2, 3, 64)))

    def test_empty_inputs(self):
        self.assertReference(_layer(ROUTABLE["dense"]), np.ones((0, 256)))

    def test_standard_third_party_layer_gets_the_route(self):
        layer = StandardProjection(8, use_bias=False)
        layer.build((None, 64))
        layer.quantize("int4")
        with kernels.force_route("reference_wrapped"):
            counts = _counts(layer, _random((2, 64)))
        self.assertEqual(counts, {"reference_wrapped": 1})

    def test_activation_quantizer_keeps_its_path(self):
        layer = layers.Dense(8)
        layer.build((None, 64))
        config = Int4QuantizationConfig(
            block_size=-1, activation_quantizer=AbsMaxQuantizer()
        )
        layer.quantize("int4", config=config)
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(_counts(layer, _random((2, 64))), {})

    def test_auto_without_rows_is_the_reference_path(self):
        layer = _layer(ROUTABLE["query_ndh"])
        x = _random(ROUTABLE["query_ndh"][2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        self.assertEqual(_counts(layer, x), {"reference": 1})
        self.assertAllEqual(_numpy(layer(x)), expected)

    def test_calibration_capture_runs_once_per_call(self):
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        seen = []
        with _routes(), calibration_scope({layer: seen.append}):
            for scope in (
                kernels.policy("off"),
                kernels.policy("auto"),
                kernels.force_route("reference_wrapped"),
                kernels.force_route("matrix"),
            ):
                with scope:
                    layer(x)
        self.assertLen(seen, 4)
        for inputs in seen:
            self.assertAllEqual(_numpy(inputs), x)

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Counts the selections of a trace."
    )
    def test_jax_selects_at_trace_time(self):
        layer = _layer(ROUTABLE["dense"])
        call = jax.jit(lambda v: layer(v))
        x = _random(ROUTABLE["dense"][2])
        kernels.route_counts.clear()
        with kernels.force_route("reference_wrapped"):
            call(x)
            call(x + 1)
        self.assertEqual(dict(kernels.route_counts), {"reference_wrapped": 1})

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Compiles with `torch.compile`."
    )
    def test_torch_compile_does_not_recompile(self):
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        torch._dynamo.reset()
        call = torch.compile(lambda v: layer(v), backend="eager")
        kernels.route_counts.clear()
        # The first call compiles. The calls with new values reuse its code.
        outputs = [_numpy(call(ops.convert_to_tensor(x)))]
        with torch._dynamo.config.patch(error_on_recompile=True):
            for i in (1, 2):
                outputs.append(_numpy(call(ops.convert_to_tensor(x + i))))
        self.assertEqual(dict(kernels.route_counts), {})
        for i, y in enumerate(outputs):
            self.assertAllEqual(y, _numpy(layer(x + i)))

    @pytest.mark.skipif(
        backend.backend() not in ("tensorflow", "jax"),
        reason="Exports a TensorFlow SavedModel.",
    )
    def test_export_traces_the_reference_path(self):
        inputs = layers.Input((256,))
        model = models.Model(inputs, layers.Dense(96)(inputs))
        model.quantize("int4")
        path = os.path.join(self.get_temp_dir(), "model")
        with kernels.force_route("reference_wrapped"):
            kernels.route_counts.clear()
            model.export(path, format="tf_saved_model", verbose=False)
            counts = dict(kernels.route_counts)
            self.assertEqual(
                _counts(model, np.ones((2, 256))), {"reference_wrapped": 1}
            )
        self.assertNotIn("reference_wrapped", counts)
        self.assertGreater(counts["reference"], 0)


class AllowListTest(testing.TestCase):
    def test_row_selects_its_route_in_its_bucket(self):
        layer = _layer(ROUTABLE["dense"])
        with _routes({_row_key("float32", "1-16"): (("matrix", None),)}):
            for m, name in ((1, "matrix"), (16, "matrix"), (17, "reference")):
                counts = _counts(layer, np.ones((m, 256), "float32"))
                self.assertEqual(counts, {name: 1})

    def test_first_route_whose_gates_pass(self):
        layer = _layer(ROUTABLE["dense"])
        row = (
            ("bfloat16_only", None),
            ("unavailable", None),
            ("untested", None),
            ("matrix", None),
        )
        with _routes(
            {_row_key("float32", "1-16"): row},
            bfloat16_only=kernels.RouteSource(_matrix, dtypes=("bfloat16",)),
            unavailable=kernels.RouteSource(
                _matrix, available=lambda m, k, n, view: False
            ),
            untested=_untested(),
        ):
            self.assertEqual(_counts(layer, np.ones((2, 256))), {"matrix": 1})

    def test_set_policy_off(self):
        layer = _layer(ROUTABLE["dense"])
        with _routes({_row_key("float32", "1-16"): (("matrix", None),)}):
            previous = kernels._state.policy
            kernels.set_policy("off")
            try:
                counts = _counts(layer, np.ones((2, 256)))
            finally:
                kernels.set_policy(previous)
            self.assertEqual(counts, {"reference": 1})
            self.assertEqual(_counts(layer, np.ones((2, 256))), {"matrix": 1})

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="A one-device `DataParallel`."
    )
    def test_distribution_runs_the_reference_path(self):
        layer = _layer(ROUTABLE["dense"])
        devices = distribution.list_devices("cpu")[:1]
        data_parallel = distribution.DataParallel(devices=devices)
        rows = {_row_key("float32", "1-16"): (("matrix", None),)}
        with _routes(rows), data_parallel.scope():
            counts = _counts(layer, np.ones((2, 256)))
        self.assertEqual(counts, {"reference": 1})

    def test_device_class(self):
        if backend.backend() != "torch" and _device_type() != "cpu":
            self.skipTest("The default device is not a CPU.")
        device = "cpu" if backend.backend() == "torch" else None
        cpu = f"cpu-{platform.machine()}"
        self.assertEqual(kernels._device_class(device), cpu)


class PolicyTest(testing.TestCase):
    """The tests compare the state with the state at the start of the test.
    A test runner can set a policy other than `"auto"` for the process."""

    def setUp(self):
        super().setUp()
        self.initial = self.state()

    def state(self):
        return kernels._state.policy, kernels._state.route

    def test_the_default_policy_is_auto(self):
        """A new process starts with the policy `"auto"` and no forced
        route. A test runner can set a different policy for this process,
        so a subprocess reads the state."""
        root = os.path.dirname(os.path.dirname(keras.__file__))
        env = dict(os.environ, KERAS_BACKEND=backend.backend(), PYTHONPATH=root)
        code = (
            "from keras.src.quantizers import kernels\n"
            "print(kernels._state.policy, kernels._state.route)\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stderr[-4000:])
        self.assertEqual(process.stdout.splitlines()[-1], "auto None")

    def test_policy_restores_the_state(self):
        with kernels.policy("off"):
            with kernels.force_route("reference_wrapped"):
                self.assertEqual(self.state(), ("auto", "reference_wrapped"))
            self.assertEqual(self.state(), ("off", None))
        with self.assertRaises(RuntimeError):
            with kernels.policy("off"):
                raise RuntimeError("inside")
        self.assertEqual(self.state(), self.initial)

    def test_policy_decorates_a_function(self):
        @kernels.policy("off")
        def traced():
            return self.state()

        with kernels.force_route("reference_wrapped"):
            self.assertEqual(traced(), ("off", None))
            self.assertEqual(self.state(), ("auto", "reference_wrapped"))

    def test_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            kernels.set_policy("fast")
        with self.assertRaisesRegex(ValueError, "Unknown route"):
            with kernels.force_route("triton"):
                pass
        self.assertEqual(self.state(), self.initial)

    def test_require_raises_without_a_kernel(self):
        with self.assertRaisesRegex(ValueError, "needs a platform with a"):
            kernels.set_policy("require")
        with self.assertRaisesRegex(ValueError, "needs a platform with a"):
            with kernels.policy("require"):
                pass
        self.assertEqual(self.state(), ("auto", None))

    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy lists no devices."
    )
    def test_require_runs_the_platform_kernel(self):
        """`"require"` ignores the allow-list and the version range."""
        layer = _layer(ROUTABLE["dense"])
        refused = _layer(REFUSED["batch_axis"])
        x = _random(REFUSED["batch_axis"][2])
        required = {_device_type(): "untested"}
        with _routes(required=required, untested=_untested()):
            with kernels.policy("require"):
                counts = [
                    _counts(layer, np.ones((2, 256))),
                    _counts(layer, np.ones((0, 256))),
                    _counts(refused, x),
                ]
        self.assertEqual(
            counts, [{"untested": 1}, {"reference": 1}, {"reference": 1}]
        )


class VersionTest(testing.TestCase):
    @parameterized.named_parameters(
        ("release", "3.8.0", ("3.8",), True),
        ("patch", "3.8.2", ("3.8",), True),
        ("local", "3.8.0+cu130", ("3.8",), True),
        ("post", "0.11.2.post1", ("0.11.2",), True),
        ("minor_outside", "3.9.0", ("3.8",), False),
        ("longer_patch", "0.11.20", ("0.11.2",), False),
        ("release_candidate", "3.8.0rc1", ("3.8",), False),
        ("development", "0.12.0.dev20261019", ("0.12.0",), False),
        ("equal_pair", ("0.12.0", "0.12.0"), ("0.11.2", "0.12.0"), True),
        ("unequal_pair", ("0.12.0", "0.11.2"), ("0.11.2", "0.12.0"), False),
    )
    def test_tested(self, installed, prefixes, expected):
        self.assertEqual(kernels.tested(installed, prefixes), expected)
