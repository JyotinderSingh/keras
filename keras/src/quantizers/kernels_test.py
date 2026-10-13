"""Route selection for a 4-bit weight-only contraction (`kernels.py`).

The tests run the reference path as a route
(`force_route("reference_wrapped")`) and the `"matrix"` route, which
contracts the `(M, K)` inputs with the stored matrix, both from
`kernels_test_utils`. The route suites run each kernel route of the
process through real layers. The labels G1 to G8 are those of
`kernels_test_utils`.
"""

import dataclasses
import math
import os
import platform
import subprocess
import sys
import threading

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
from keras.src.backend import config
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.quantized_kernels import Kernel
from keras.src.export.saved_model import ExportArchive
from keras.src.quantizers import kernels
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.kernels_test_utils import admitted
from keras.src.quantizers.kernels_test_utils import counts
from keras.src.quantizers.kernels_test_utils import device_type
from keras.src.quantizers.kernels_test_utils import direct
from keras.src.quantizers.kernels_test_utils import floor_and_bound
from keras.src.quantizers.kernels_test_utils import matrix_route
from keras.src.quantizers.kernels_test_utils import route_tables
from keras.src.quantizers.kernels_test_utils import row_key
from keras.src.quantizers.packing import pack_int4
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_test_utils import PermutedDense
from keras.src.quantizers.quantization_test_utils import Pointwise1D
from keras.src.quantizers.quantization_test_utils import StandardProjection
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.quantizers.quantization_test_utils import reordered_sum_bound
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing.test_utils import named_product
from keras.src.utils.module_utils import torch_xla

if backend.backend() == "jax":
    import jax

# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
# Each equation is reshape-compatible. `dense_aligned` and `qkv_aligned`
# have `K = 256` and `N = 128`, which the shape check of each kernel
# admits in groups of 128 and per channel.
ROUTABLE = {
    "dense": (None, 96, (3, 256)),
    "dense_aligned": (None, 128, (3, 256)),
    "dense_ragged": (None, 48, (2, 3, 200)),
    "qkv_dnh": ("btd,dnh->btnh", (None, 4, 16), (2, 5, 128)),
    "qkv_aligned": ("btd,dnh->btnh", (None, 4, 32), (2, 5, 256)),
    "query_ndh": ("btd,ndh->btnh", (None, 4, 16), (2, 5, 128)),
    "output_nhd": ("btnh,nhd->btd", (None, 96), (2, 5, 4, 32)),
    "expert_gate": ("btd,edi->btei", (None, 3, 16), (2, 5, 128)),
    "contracted_first": ("abc,cde->abde", (5, 4, 16), (2, 5, 128)),
    "two_contracted_apart": ("abcd,ced->abe", (None, 32), (2, 4, 8, 16)),
    "outer_product": ("bt,nh->btnh", (None, 4, 6), (2, 5)),
}
# Equations that `matrix_contraction` refuses.
REFUSED = {
    "batch_axis": ("ebd,edf->ebf", (None, 16), (3, 2, 128)),
    "summed_input_axis": ("abc,cd->ad", (16,), (2, 3, 128)),
    "contracted_out_of_order": ("abcd,edc->abe", (None, 32), (2, 4, 8, 16)),
}
# name -> (mode, group size, activation order).
CALIBRATED = {
    "gptq": ("gptq", 128, False),
    "gptq_per_channel": ("gptq", -1, False),
    "gptq_act_order": ("gptq", 128, True),
    "awq": ("awq", 128, False),
}


def _untested(kernel=False):
    """A route outside its tested version range. With `kernel`, the route
    is a kernel for every device, so `"require"` runs it."""
    any_device = Kernel(
        name="untested",
        matmul=None,
        dtypes=(),
        supports_device=lambda device_class: True,
        requirements="any device",
        available=None,
    )
    return matrix_route(
        tested=lambda: False, kernel=any_device if kernel else None
    )


def _float_layer(case, dtype="float32", bias=False):
    """A built float layer of `case` with a random kernel."""
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
    return layer


def _layer(case, block_size=32, dtype="float32", bias=False):
    """A layer of `case`, quantized to int4 from a random float kernel."""
    layer = _float_layer(case, dtype, bias)
    config = Int4QuantizationConfig(block_size=block_size)
    layer.quantize("int4", config=config)
    return layer


def _calibrated(case, mode, dtype="float32"):
    """A layer of `case`, calibrated in the `CALIBRATED` mode `mode`."""
    layer = _float_layer(case, dtype)
    mode, group_size, activation_order = CALIBRATED[mode]
    kwargs = {"activation_order": True} if activation_order else {}
    config = calibration_config(mode, group_size=group_size, **kwargs)
    calibrate_layer(layer, config, _random((16,) + case[2][1:], seed=4))
    return layer


def _random(shape, seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype("f4")


def _numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def _oracle(layer, x):
    view = layer._quantized_weight()

    def read(tensor, scale=False):
        if tensor is None:
            return None
        value = ops.convert_to_numpy(tensor)
        if scale and layer.compute_dtype != "float32":
            # The layer reads its scales in the compute dtype.
            value = value.astype(ml_dtypes.finfo(layer.compute_dtype).dtype)
        return value

    return oracle_contract(
        x,
        read(view.codes),
        read(view.scale, scale=True),
        columns=view.layout.orig_len,
        kernel_shape=view.shape,
        permutation=view.permutation,
        zero_point=read(view.zero_point),
        g_idx=read(view.g_idx),
        group_size=view.scheme.group_size,
        input_scales=read(view.input_scales, scale=True),
        equation=getattr(layer, "equation", None),
        dtype=layer.compute_dtype,
    )


def _err(y, oracle):
    return np.max(np.abs(y - oracle)) / np.max(np.abs(oracle))


def _check_route(test, layer, x, name):
    """Runs the forced route `name` against the oracle and the reference.

    The error is at most 1.25 times the reference's plus a floor, and at
    most an absolute bound (G1, G2). Where the reference path's own error
    is above that bound (bfloat16 GPTQ, on some data), G2 takes the
    reference path's error. The outputs are finite (G4) and deterministic
    (G5), and equal a direct call of the route (G8). Returns the outputs
    of the route and of the reference path.
    """
    oracle = _oracle(layer, x)
    with kernels.policy("off"):
        reference = _numpy(layer(x))
    floor, bound = floor_and_bound(layer.compute_dtype)
    kernels.route_counts.clear()
    with kernels.force_route(name):
        y, again, called = layer(x), layer(x), direct(layer, x)
    test.assertEqual(dict(kernels.route_counts), {name: 3})
    y = _numpy(y)
    test.assertTrue(np.all(np.isfinite(y)))
    test.assertAllEqual(_numpy(again), y)
    test.assertAllEqual(_numpy(called), y)
    err, reference_err = _err(y, oracle), _err(reference, oracle)
    test.assertLessEqual(err, 1.25 * reference_err + floor)
    test.assertLessEqual(err, max(bound, reference_err))
    return y, reference


def _exact_inputs(input_shape, rows, rng):
    """Inputs with eight nonzero entries of 1 or -1 per contracted row."""
    x = np.zeros((math.prod(input_shape) // rows, rows), "float32")
    for row in x:
        row[rng.choice(rows, 8, replace=False)] = rng.choice([-1, 1], 8)
    return x.reshape(input_shape)


class RouteSuiteTest(testing.TestCase):
    """Each route that a call admits, against the float64 oracle and the
    reference path: `"reference_wrapped"`, `"matrix"` and the kernel
    routes of this process (`kernels_test_utils.admitted`).
    `dense_aligned` and `qkv_aligned` pass the shape checks of each
    kernel."""

    def routes(self, layer, x):
        """The routes that a call of `layer` on `x` admits. The wrapped
        reference path and the `"matrix"` route admit each call."""
        names = admitted(layer, x)
        self.assertIn("reference_wrapped", names)
        self.assertIn("matrix", names)
        return names

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
        with kernels.policy("off"):
            err = _err(_numpy(layer(x)), _oracle(layer, x))
        self.assertLessEqual(err, floor_and_bound(layer.compute_dtype)[1])
        with route_tables():
            for name in self.routes(layer, x):
                y, reference = _check_route(self, layer, x, name)
                if name == "reference_wrapped":
                    self.assertAllEqual(y, reference)

    @parameterized.named_parameters(
        named_product(
            case=[
                "dense",
                "dense_aligned",
                "query_ndh",
                "output_nhd",
                "two_contracted_apart",
            ],
            block_size=[32, 128, -1],
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
        x = _exact_inputs(ROUTABLE[case][2], rows, rng)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        with route_tables():
            for name in self.routes(layer, x):
                with kernels.force_route(name):
                    self.assertEqual(counts(layer, x), {name: 1})
                    self.assertAllEqual(_numpy(layer(x)), oracle)

    @parameterized.named_parameters(
        named_product(
            case=[
                "dense",
                "dense_aligned",
                "query_ndh",
                "two_contracted_apart",
            ],
            block_size=[32, 128, -1],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy has no gradients."
    )
    def test_input_gradient(self, case, block_size, dtype):
        """A route replaces the forward only, so the gradient is the same.

        The bias and LoRA stay outside the route.
        """
        layer = _layer(ROUTABLE[case], block_size, dtype, bias=True)
        layer.enable_lora(4)
        rng = np.random.default_rng(3)
        layer.lora_kernel_b.assign(rng.random(layer.lora_kernel_b.shape))
        x = _random(ROUTABLE[case][2])
        with kernels.policy("off"):
            expected = input_gradient(layer, x)
        with route_tables():
            for name in self.routes(layer, x):
                with kernels.force_route(name):
                    self.assertAllEqual(input_gradient(layer, x), expected)

    @parameterized.named_parameters(
        named_product(
            case=["dense_aligned", "dense_ragged", "query_ndh", "output_nhd"],
            mode=sorted(CALIBRATED),
            dtype=["float32", "mixed_bfloat16", "mixed_float16"],
        )
    )
    def test_calibrated(self, case, mode, dtype):
        """GPTQ and AWQ codes (uint8 storage)."""
        layer = _calibrated(ROUTABLE[case], mode, dtype)
        x = _random(ROUTABLE[case][2])
        with route_tables():
            for name in self.routes(layer, x):
                _check_route(self, layer, x, name)

    @parameterized.named_parameters(
        named_product(
            case=["dense_aligned", "query_ndh"],
            mode=["gptq", "gptq_act_order", "awq"],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy has no gradients."
    )
    def test_calibrated_input_gradient(self, case, mode, dtype):
        """A route runs inside the custom gradient of a routed GPTQ or AWQ
        call, so its input gradient is the wrapped reference's, bit for
        bit (G7). For `Dense` in float32, it is also the reference path's.
        Autodiff takes that gradient with another matmul, which can sum
        the `N` products of an entry in another order:
        `reordered_sum_bound`."""
        layer = _calibrated(ROUTABLE[case], mode, dtype)
        x = _random(ROUTABLE[case][2])
        with kernels.force_route("reference_wrapped"):
            expected = input_gradient(layer, x)
        with route_tables():
            for name in self.routes(layer, x):
                with kernels.force_route(name):
                    self.assertAllEqual(input_gradient(layer, x), expected)
        if case == "dense_aligned" and dtype == "float32":
            # The output gradient is ones, so entry `k` sums row `k` of
            # the `(K, N)` kernel.
            kernel = _numpy(layer._quantized_weight().dequantize("float32"))
            bound = reordered_sum_bound(
                np.max(np.sum(np.abs(kernel), axis=1)), kernel.shape[1]
            )
            with kernels.policy("off"):
                self.assertAllClose(
                    input_gradient(layer, x), expected, atol=bound, rtol=0
                )


class FallbackTest(testing.TestCase):
    """Each gate refusal runs the reference path."""

    def assertReference(self, layer, x):
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters([(name, name) for name in REFUSED])
    def test_refused_equation(self, name):
        self.assertReference(_layer(REFUSED[name]), _random(REFUSED[name][2]))

    def test_custom_geometry_without_the_hook(self):
        # `PermutedGeometry` overrides `contract` and does not define
        # `matrix_contraction`, so no route runs it.
        layer = PermutedDense(8, use_bias=False)
        layer.build((None, 3, 64))
        layer.quantize("int4")
        self.assertReference(layer, _random((2, 3, 64)))

    def test_custom_geometry_with_the_hook(self):
        # `PointwiseGeometry` describes its `(1, in, out)` kernel as the
        # `(in, out)` matrix of the last input axis, so the routes run it.
        layer = Pointwise1D(8, use_bias=False)
        layer.build((None, 3, 64))
        layer.quantize("int4")
        x = _random((2, 3, 64))
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(counts(layer, x), {"reference_wrapped": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    def test_empty_inputs(self):
        self.assertReference(_layer(ROUTABLE["dense"]), np.ones((0, 256)))

    def test_standard_third_party_layer_gets_the_route(self):
        layer = StandardProjection(8, use_bias=False)
        layer.build((None, 64))
        layer.quantize("int4")
        with kernels.force_route("reference_wrapped"):
            selected = counts(layer, _random((2, 64)))
        self.assertEqual(selected, {"reference_wrapped": 1})

    def test_activation_quantizer_keeps_its_path(self):
        layer = layers.Dense(8)
        layer.build((None, 64))
        config = Int4QuantizationConfig(
            block_size=-1, activation_quantizer=AbsMaxQuantizer()
        )
        layer.quantize("int4", config=config)
        with kernels.force_route("reference_wrapped"):
            self.assertEqual(counts(layer, _random((2, 64))), {})

    def test_auto_without_rows_is_the_reference_path(self):
        layer = _layer(ROUTABLE["query_ndh"])
        x = _random(ROUTABLE["query_ndh"][2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with route_tables():
            self.assertEqual(counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    def test_calibration_capture_runs_once_per_call(self):
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        seen = []
        with route_tables(), calibration_scope({layer: seen.append}):
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
            selected = dict(kernels.route_counts)
            self.assertEqual(
                counts(model, np.ones((2, 256))), {"reference_wrapped": 1}
            )
        self.assertNotIn("reference_wrapped", selected)
        self.assertGreater(selected["reference"], 0)

    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="No `ExportArchive` on NumPy."
    )
    def test_export_archive_traces_the_reference_path(self):
        """A hand-built `ExportArchive` traces outside `Model.export`: in
        `write_out` on JAX and TensorFlow, and in `track_and_add_endpoint`
        on torch. Each traces under the policy `"off"` itself."""
        if backend.backend() == "torch" and not torch_xla.available:
            self.skipTest("The torch archive needs `torch_xla`.")
        inputs = layers.Input((256,))
        model = models.Model(inputs, layers.Dense(96)(inputs))
        model.quantize("int4")
        path = os.path.join(self.get_temp_dir(), "archive")
        signature = [layers.InputSpec(shape=(None, 256), dtype="float32")]
        with kernels.force_route("reference_wrapped"):
            kernels.route_counts.clear()
            archive = ExportArchive()
            if backend.backend() == "torch":
                archive.track_and_add_endpoint("call", model, signature)
            else:
                archive.track(model)
                archive.add_endpoint("call", model.call, signature)
            archive.write_out(path)
            selected = dict(kernels.route_counts)
        self.assertNotIn("reference_wrapped", selected)
        self.assertGreater(selected["reference"], 0)


class MeasuredRowsTest(testing.TestCase):
    def test_row_selects_its_route_in_its_bucket(self):
        layer = _layer(ROUTABLE["dense"])
        with route_tables({row_key("float32", "1-16"): ("matrix",)}):
            for m, name in ((1, "matrix"), (16, "matrix"), (17, "reference")):
                selected = counts(layer, np.ones((m, 256), "float32"))
                self.assertEqual(selected, {name: 1})

    def test_first_route_whose_gates_pass(self):
        layer = _layer(ROUTABLE["dense"])
        row = ("bfloat16_only", "unavailable", "untested", "matrix")
        with route_tables(
            {row_key("float32", "1-16"): row},
            routes={
                "bfloat16_only": matrix_route(dtypes=("bfloat16",)),
                "unavailable": matrix_route(
                    available=lambda device_class, m, k, n, view, dtype: False
                ),
                "untested": _untested(),
            },
        ):
            self.assertEqual(counts(layer, np.ones((2, 256))), {"matrix": 1})

    def test_disable_quantized_kernels(self):
        layer = _layer(ROUTABLE["dense"])
        with route_tables({row_key("float32", "1-16"): ("matrix",)}):
            # `policy("auto")` puts the policy back at the end of the block.
            with kernels.policy("auto"):
                keras.config.disable_quantized_kernels()
                self.assertFalse(keras.config.is_quantized_kernels_enabled())
                selected = counts(layer, np.ones((2, 256)))
            self.assertEqual(selected, {"reference": 1})
            self.assertEqual(counts(layer, np.ones((2, 256))), {"matrix": 1})

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="A one-device `DataParallel`."
    )
    def test_distribution_runs_the_reference_path(self):
        layer = _layer(ROUTABLE["dense"])
        devices = distribution.list_devices("cpu")[:1]
        data_parallel = distribution.DataParallel(devices=devices)
        rows = {row_key("float32", "1-16"): ("matrix",)}
        with route_tables(rows), data_parallel.scope():
            selected = counts(layer, np.ones((2, 256)))
        self.assertEqual(selected, {"reference": 1})

    def test_device_class(self):
        device = device_type()
        if backend.backend() == "torch":
            device = str(ops.convert_to_tensor(np.ones(1, "float32")).device)
        if not device.startswith("cpu"):
            self.skipTest("The default device is not a CPU.")
        cpu = f"cpu-{platform.machine()}"
        self.assertEqual(kernels._device_class(), cpu)


class PolicyTest(testing.TestCase):
    """The tests compare the state with the state at the start of the test.
    `pytest --kernel-policy` can set a policy other than `"auto"` for the
    process."""

    def setUp(self):
        super().setUp()
        self.initial = self.state()

    def state(self):
        return config._quantized_kernels_policy(), kernels._state.route

    def new_process(self, **variables):
        """Runs a new process with `variables` in its environment, which
        prints the state at import. `pytest --kernel-policy` can set a
        different policy for this process, so the tests of the initial
        state read a subprocess."""
        root = os.path.dirname(os.path.dirname(keras.__file__))
        env = dict(os.environ)
        env.pop("KERAS_QUANTIZED_KERNELS", None)
        env.update(KERAS_BACKEND=backend.backend(), PYTHONPATH=root)
        env.update(variables)
        code = (
            "import keras\n"
            "from keras.src.backend import config\n"
            "from keras.src.quantizers import kernels\n"
            "print(config._quantized_kernels_policy(), kernels._state.route, "
            "keras.config.is_quantized_kernels_enabled())\n"
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_environment_variable(self):
        """`KERAS_QUANTIZED_KERNELS` takes `auto` or `off`, in any case."""
        process = self.new_process(KERAS_QUANTIZED_KERNELS="OFF")
        self.assertEqual(process.returncode, 0, process.stderr[-4000:])
        self.assertEqual(process.stdout.splitlines()[-1], "off None False")
        process = self.new_process(KERAS_QUANTIZED_KERNELS="require")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("KERAS_QUANTIZED_KERNELS", process.stderr)
        self.assertIn("'require'", process.stderr)

    def test_public_switch(self):
        """`disable_quantized_kernels()` sets `"off"` for the process: a
        `clear_session()` keeps it, and a worker thread reads it."""
        seen = []
        # `policy` puts the initial policy back at the end of the block.
        with kernels.policy(self.initial[0]):
            keras.config.disable_quantized_kernels()
            self.assertEqual(self.state(), ("off", self.initial[1]))
            self.assertFalse(keras.config.is_quantized_kernels_enabled())
            clear_session()
            self.assertEqual(self.state(), ("off", self.initial[1]))
            worker = threading.Thread(
                target=lambda: seen.append(self.state()[0])
            )
            worker.start()
            worker.join()
            keras.config.enable_quantized_kernels()
            self.assertEqual(self.state(), ("auto", self.initial[1]))
            self.assertIs(keras.config.is_quantized_kernels_enabled(), True)
        self.assertEqual(seen, ["off"])
        self.assertEqual(self.state(), self.initial)

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
        """`policy` leaves the forced route alone: under `"off"` no route
        runs, forced or not."""

        @kernels.policy("off")
        def traced():
            return self.state()

        with kernels.force_route("reference_wrapped"):
            self.assertEqual(traced(), ("off", "reference_wrapped"))
            self.assertEqual(self.state(), ("auto", "reference_wrapped"))

    def test_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            with kernels.policy("fast"):
                pass
        with self.assertRaisesRegex(ValueError, "Unknown route"):
            with kernels.force_route("unknown"):
                pass
        self.assertEqual(self.state(), self.initial)

    def test_require_raises_without_a_kernel(self):
        """The test takes this backend's kernels out of the route table."""
        with route_tables(required=()):
            with self.assertRaisesRegex(ValueError, "needs a platform with"):
                with kernels.policy("require"):
                    pass
        self.assertEqual(self.state(), self.initial)

    def test_require_runs_the_platform_kernel(self):
        """`"require"` runs the kernel routes whose device check passes,
        and ignores the measured rows and the version range. A call that the
        gates refuse runs the reference path."""
        layer = _layer(ROUTABLE["dense"])
        refused = _layer(REFUSED["batch_axis"])
        x = _random(REFUSED["batch_axis"][2])
        routes = dict(kernels._ROUTES)
        with route_tables(
            required=("untested",), routes={"untested": _untested(True)}
        ):
            with kernels.policy("require"):
                selected = [
                    counts(layer, np.ones((2, 256))),
                    counts(layer, np.ones((0, 256))),
                    counts(refused, x),
                ]
        self.assertEqual(
            selected, [{"untested": 1}, {"reference": 1}, {"reference": 1}]
        )
        # The routes of the platform's kernels stay.
        self.assertEqual(kernels._ROUTES, routes)

    def test_require_names_the_requirements(self):
        """On a device that no kernel supports, `"require"` raises with
        the requirements of each kernel."""
        never = dataclasses.replace(
            _untested(True).kernel,
            name="never",
            supports_device=lambda device_class: False,
            requirements="a device that does not exist",
        )
        with route_tables(
            required=("never",), routes={"never": matrix_route(kernel=never)}
        ):
            with self.assertRaisesRegex(
                ValueError, "never needs a device that does not exist"
            ):
                with kernels.policy("require"):
                    pass
        self.assertEqual(self.state(), self.initial)
