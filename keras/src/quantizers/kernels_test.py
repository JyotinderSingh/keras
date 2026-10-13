"""Route selection for a 4-bit weight-only contraction (`kernels.py`).

The library has one route, N5 (`"n5"`, sum-then-scale in `keras.ops`).
The tests also run the reference path as a route
(`force_route("reference_wrapped")`), and `matrix`, a route of this file
that contracts the `(M, K)` inputs with the stored matrix.
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
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import input_gradient
from keras.src.quantizers.quantization_test_utils import oracle_contract
from keras.src.quantizers.quantization_test_utils import reordered_sum_bound
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing.test_utils import named_product

if backend.backend() == "jax":
    import jax
elif backend.backend() == "tensorflow":
    import tensorflow as tf
elif backend.backend() == "torch":
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

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
# name -> (mode, group size, activation order).
CALIBRATED = {
    "gptq": ("gptq", 128, False),
    "gptq_per_channel": ("gptq", -1, False),
    "gptq_act_order": ("gptq", 128, True),
    "awq": ("awq", 128, False),
}
# An `M` in each bucket.
BUCKET_ROWS = {"1-16": 4, "17-64": 33, "65-256": 130, "257+": 300}


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
    """Adds `matrix` and `sources` to the route table, sets the allow-list
    to `rows`, and sets `required` as this backend's `"require"` kernels
    (`{}` for none). The tables are as before when the block ends."""
    sources = {"matrix": kernels.RouteSource(_matrix), **sources}
    allow_list = dict(kernels._ALLOW_LIST)
    required_kernels = dict(kernels._REQUIRED)
    kernels._ROUTES.update(sources)
    kernels._ALLOW_LIST.clear()
    kernels._ALLOW_LIST.update(rows or {})
    if required is not None:
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
        kernels._ALLOW_LIST.clear()
        kernels._ALLOW_LIST.update(allow_list)
        kernels._REQUIRED.clear()
        kernels._REQUIRED.update(required_kernels)


def _device_type():
    """The device type of `list_devices`: `"cpu"`, `"gpu"`, `"tpu"`..."""
    if backend.backend() == "numpy":
        return "cpu"
    return distribution.list_devices()[0].split(":")[0]


def _row_key(dtype, bucket):
    """The allow-list key of this backend and device."""
    device = None
    if backend.backend() == "torch":
        device = str(ops.convert_to_tensor(np.ones(1, "float32")).device)
    return (backend.backend(), kernels._device_class(device), dtype, bucket)


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
        input_scales=read(view.input_scales, scale=True),
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


def _floor_and_bound(layer):
    """The G1 floor and the G2 bound of the layer's compute dtype."""
    if layer.compute_dtype == "float32":
        return 1e-6, 1e-5
    if layer.compute_dtype == "float16":
        return 2e-4, 2**-9
    return 1e-3, 2**-7


def _check_route(test, layer, x, name):
    """Runs the forced route `name` against the oracle and the reference.

    The error is at most 1.25 times the reference's plus a floor, and at
    most an absolute bound (G1, G2). The outputs are finite (G4) and
    deterministic (G5), and equal a direct call of the route (G8).
    Returns the outputs of the route and of the reference path.
    """
    oracle = _oracle(layer, x)
    with kernels.policy("off"):
        reference = _numpy(layer(x))
    floor, bound = _floor_and_bound(layer)
    kernels.route_counts.clear()
    with kernels.force_route(name):
        y, again, direct = layer(x), layer(x), _direct(layer, x)
    test.assertEqual(dict(kernels.route_counts), {name: 3})
    y = _numpy(y)
    test.assertTrue(np.all(np.isfinite(y)))
    test.assertAllEqual(_numpy(again), y)
    test.assertAllEqual(_numpy(direct), y)
    err = _err(y, oracle)
    test.assertLessEqual(err, 1.25 * _err(reference, oracle) + floor)
    test.assertLessEqual(err, bound)
    return y, reference


def _contiguous(layer):
    """Whether `g_idx[k] == k // G` for every stored row `k`."""
    view = layer._quantized_weight()
    rows = view.codes.shape[0]
    groups = np.arange(rows) // min(view.scheme.group_size, rows)
    return np.array_equal(_numpy(view.g_idx), groups)


def _host_tensors(fn):
    """Runs `fn` and returns how many tensors torch makes from host data
    (`aten.lift_fresh`), for example from a Python number operand.

    Torch copies such a tensor to the device on each call, and a CUDA
    graph capture refuses the copy. The op runs on the CPU device too.
    """
    count = 0

    class HostTensors(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal count
            if func is torch.ops.aten.lift_fresh.default:
                count += 1
            return func(*args, **(kwargs or {}))

    with HostTensors():
        fn()
    return count


def _exact_inputs(input_shape, rows, rng):
    """Inputs with eight nonzero entries of 1 or -1 per contracted row."""
    x = np.zeros((math.prod(input_shape) // rows, rows), "float32")
    for row in x:
        row[rng.choice(rows, 8, replace=False)] = rng.choice([-1, 1], 8)
    return x.reshape(input_shape)


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
        with kernels.policy("off"):
            err = _err(_numpy(layer(x)), _oracle(layer, x))
        self.assertLessEqual(err, _floor_and_bound(layer)[1])
        names = ["reference_wrapped", "matrix", "n5"]
        if case == "dense_ragged" and block_size != -1:
            names.remove("n5")  # A partial group; `N5Test` runs the refusal.
        with _routes():
            for name in names:
                y, reference = _check_route(self, layer, x, name)
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
        x = _exact_inputs(ROUTABLE[case][2], rows, rng)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        with _routes():
            for name in ("reference_wrapped", "matrix", "n5"):
                with kernels.force_route(name):
                    self.assertEqual(_counts(layer, x), {name: 1})
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
            for name in ("reference_wrapped", "matrix", "n5"):
                with kernels.force_route(name):
                    self.assertAllEqual(input_gradient(layer, x), expected)


class N5Test(testing.TestCase):
    """N5 through `force_route("n5")`, and under `"auto"` with a row."""

    @parameterized.named_parameters(
        named_product(
            block_size=[32, 64, 128, 256],
            groups=[1, 2, 8],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    def test_grouped_shapes(self, block_size, groups, dtype):
        """int4 codes (int8 storage) with `K` of 1, 2 and 8 groups."""
        rows = groups * block_size
        layer = _layer((None, 10, (1, rows)), block_size, dtype)
        for m in (1, 3, 16, 17, 33, 130):
            _check_route(self, layer, _random((m, rows), seed=m), "n5")

    @parameterized.named_parameters(
        named_product(
            case=[("per_channel", -1, 200), ("short_group", 128, 48)],
            dtype=["float32", "mixed_bfloat16", "mixed_float16"],
        )
    )
    def test_rows_without_whole_groups(self, case, dtype):
        """Per-channel int4 has no groups, and a block longer than `K` is
        one group."""
        _, block_size, rows = case
        layer = _layer((None, 10, (1, rows)), block_size, dtype)
        _check_route(self, layer, _random((5, rows)), "n5")

    @parameterized.named_parameters(
        named_product(
            case=["dense", "query_ndh", "output_nhd"],
            mode=sorted(CALIBRATED),
            dtype=["float32", "mixed_bfloat16", "mixed_float16"],
        )
    )
    def test_calibrated(self, case, mode, dtype):
        """GPTQ and AWQ codes (uint8 storage). With groups that are not
        contiguous, the guard gives the reference path's output."""
        layer = _calibrated(ROUTABLE[case], mode, dtype)
        x = _random(ROUTABLE[case][2])
        y, reference = _check_route(self, layer, x, "n5")
        if not _contiguous(layer):
            self.assertAllEqual(y, reference)

    @parameterized.named_parameters(
        named_product(dtype=["float32", "mixed_bfloat16"])
    )
    def test_guard_runs_the_reference_path(self, dtype):
        """GPTQ with activation order scatters the groups."""
        layer = _calibrated(ROUTABLE["dense"], "gptq_act_order", dtype)
        x = _random(ROUTABLE["dense"][2])
        self.assertFalse(_contiguous(layer))
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("n5"):
            self.assertEqual(_counts(layer, x), {"n5": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)
            if backend.backend() == "jax":
                # `ops.cond` traces both branches.
                y = jax.jit(lambda v: layer(v))(x)
                self.assertAllEqual(_numpy(y), expected)

    @parameterized.named_parameters(
        named_product(
            mode=["gptq", "gptq_per_channel", "awq"],
            dtype=["float32", "mixed_bfloat16"],
        )
    )
    def test_exact_data(self, mode, dtype):
        """uint8 storage: small integers and power-of-two scales give the
        oracle exactly."""
        layer = _calibrated(ROUTABLE["dense"], mode, dtype)
        view = layer._quantized_weight()
        rng = np.random.default_rng(2)
        rows, columns = view.codes.shape[0], view.layout.orig_len
        codes = rng.integers(7, 10, (rows, columns)).astype("uint8")
        layer.quantized_kernel.assign(pack_int4(codes, -1, "uint8")[0])
        zero = rng.integers(7, 10, layer.kernel_zero.shape)
        layer.kernel_zero.assign(zero.astype("uint8"))
        powers = rng.integers(1, 4, layer.kernel_scale.shape)
        layer.kernel_scale.assign(2.0**-powers)
        if mode == "awq":
            layer.awq_scales.assign(2.0 ** rng.integers(0, 2, rows))
        x = _exact_inputs(ROUTABLE["dense"][2], rows, rng)
        oracle = _oracle(layer, x).astype("float32")
        with kernels.policy("off"):
            self.assertAllEqual(_numpy(layer(x)), oracle)
        with kernels.force_route("n5"):
            self.assertEqual(_counts(layer, x), {"n5": 1})
            self.assertAllEqual(_numpy(layer(x)), oracle)

    @parameterized.named_parameters(
        named_product(case=["dense", "query_ndh"], mode=["gptq", "awq"])
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy has no gradients."
    )
    def test_calibrated_input_gradient(self, case, mode):
        """N5 runs inside the custom gradient of a routed GPTQ or AWQ call,
        so its input gradient is the wrapped reference's, bit for bit. For
        `Dense`, it is also the reference path's. Autodiff takes that
        gradient with another matmul, which can sum the `N` products of an
        entry in another order: `reordered_sum_bound`."""
        layer = _calibrated(ROUTABLE[case], mode)
        x = _random(ROUTABLE[case][2])
        with kernels.force_route("reference_wrapped"):
            expected = input_gradient(layer, x)
        with kernels.force_route("n5"):
            self.assertAllEqual(input_gradient(layer, x), expected)
        if case == "dense":
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

    @parameterized.named_parameters(
        ("odd_columns", (None, 7, (2, 64)), 32),
        ("partial_group", (None, 10, (2, 200)), 32),
        ("no_rows", (None, 10, (0, 64)), 32),
    )
    def test_refusals(self, case, block_size):
        """Odd `N` and a partial last group fail gate 5, and `M = 0` gate
        2, so the reference path runs."""
        layer = _layer(case, block_size)
        x = _random(case[2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("n5"):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters(
        ("int4", None), ("gptq", "gptq"), ("awq", "awq")
    )
    def test_auto_row(self, mode):
        """Under `"auto"`, an N5 row runs N5 in its M bucket, and the
        layer output equals a direct call of the route (G8)."""
        case = ROUTABLE["dense"]
        layer = _layer(case) if mode is None else _calibrated(case, mode)
        with _routes({_row_key("float32", "1-16"): (("n5", None),)}):
            x = _random((4, 256))
            self.assertEqual(_counts(layer, x), {"n5": 1})
            self.assertAllEqual(_numpy(layer(x)), _numpy(_direct(layer, x)))
            self.assertEqual(
                _counts(layer, _random((17, 256))), {"reference": 1}
            )

    def test_rows_of_this_platform(self):
        """Each allow-list row of this backend and device runs N5 (G8)."""
        platform_key = _row_key("float32", "1-16")[:2]
        rows = [key for key in kernels._ALLOW_LIST if key[:2] == platform_key]
        if not rows:
            self.skipTest("The allow-list has no row for this platform.")
        for _, _, dtype, bucket in rows:
            self.assertEqual(
                kernels._ALLOW_LIST[(*platform_key, dtype, bucket)],
                (("n5", None),),
            )
            policy = {"bfloat16": "mixed_bfloat16", "float16": "mixed_float16"}
            layer = _layer(ROUTABLE["dense"], 128, policy.get(dtype, dtype))
            x = _random((BUCKET_ROWS[bucket], 256))
            # Only `"auto"` reads the allow-list.
            with kernels.policy("auto"):
                self.assertEqual(_counts(layer, x), {"n5": 1})
                direct = _numpy(_direct(layer, x))
                self.assertAllEqual(_numpy(layer(x)), direct)

    @parameterized.named_parameters(("grouped", 32), ("per_channel", -1))
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Compiles with `jax.jit`."
    )
    def test_jax_jit_runs_n5(self, block_size):
        layer = _layer(ROUTABLE["dense"], block_size)
        x = _random(ROUTABLE["dense"][2])
        oracle = _oracle(layer, x)
        with kernels.policy("off"):
            reference = _numpy(layer(x))
        kernels.route_counts.clear()
        with kernels.force_route("n5"):
            y = _numpy(jax.jit(lambda v: layer(v))(x))
        self.assertEqual(dict(kernels.route_counts), {"n5": 1})
        err = _err(y, oracle)
        self.assertLessEqual(err, 1.25 * _err(reference, oracle) + 1e-6)
        self.assertLessEqual(err, 1e-5)

    @parameterized.named_parameters(
        ("contiguous_groups", None), ("activation_order", "gptq_act_order")
    )
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Counts JAX compilations."
    )
    def test_jax_eager_call_compiles_once(self, mode):
        """An eager call branches on the guard in Python, so a second call
        with the same shapes compiles nothing. An eager `lax.cond` would
        compile both branches again on each call."""
        case = ROUTABLE["dense"]
        layer = _layer(case) if mode is None else _calibrated(case, mode)
        x = _random(case[2])
        compiles = []

        def count(event, duration_secs, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles.append(duration_secs)

        kernels.route_counts.clear()
        with kernels.force_route("n5"):
            first = _numpy(layer(x))
            jax.monitoring.register_event_duration_secs_listener(count)
            try:
                second = _numpy(layer(x))
            finally:
                jax.monitoring.unregister_event_duration_listener(count)
        self.assertEqual(dict(kernels.route_counts), {"n5": 2})
        self.assertEqual(compiles, [])
        self.assertAllEqual(second, first)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Traces a `tf.function`."
    )
    def test_tensorflow_graphs_run_the_reference_path(self):
        """N5 runs eagerly only on TensorFlow."""
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        for jit_compile in (False, True):
            call = tf.function(lambda v: layer(v), jit_compile=jit_compile)
            kernels.route_counts.clear()
            with kernels.force_route("n5"):
                call(x)
            self.assertEqual(dict(kernels.route_counts), {"reference": 1})

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Builds a DTensor variable."
    )
    def test_dtensor_codes_run_the_reference_path(self):
        """Gate 6 reads the value of a variable that the view of a GPTQ
        call holds."""
        layer = _calibrated(ROUTABLE["dense"], "gptq")
        geometry = layer._quantization_geometry()
        view = layer._quantized_weight()
        x = ops.convert_to_tensor(_random((4, 256)))
        if torch.distributed.is_initialized():
            self.skipTest("Another test holds the default process group.")
        store = os.path.join(self.get_temp_dir(), "store")
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{store}", rank=0, world_size=1
        )
        try:
            mesh = distribution.DeviceMesh((1,), ["x"], ["cpu:0"])
            layout = distribution.TensorLayout(("x", None), mesh)
            with distribution.ModelParallel(
                layout_map=distribution.LayoutMap(mesh)
            ).scope():
                codes = backend.Variable(
                    ops.convert_to_numpy(view.codes),
                    dtype="uint8",
                    trainable=False,
                    layout=layout,
                )
            tensors = (codes,) + view.read_tensors()[1:]
            with _routes({_row_key("float32", "1-16"): (("n5", None),)}):
                route = kernels.select_route(layer, geometry, view, x)
                self.assertEqual(route.name, "n5")
                sharded = view.with_tensors(tensors)
                self.assertIsNone(
                    kernels.select_route(layer, geometry, sharded, x)
                )
        finally:
            torch.distributed.destroy_process_group()

    @parameterized.named_parameters(
        ("int4_grouped", None, 32),
        ("int4_per_channel", None, -1),
        ("gptq", "gptq", None),
        ("awq", "awq", None),
    )
    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Records the torch dispatch."
    )
    def test_torch_call_makes_no_host_tensor(self, mode, block_size):
        """A call of N5 and of its group guard makes no tensor from host
        data, which a CUDA graph capture refuses."""
        case = ROUTABLE["dense"]
        if mode is None:
            layer = _layer(case, block_size)
        else:
            layer = _calibrated(case, mode)
        x = ops.convert_to_tensor(_random(case[2]))
        with kernels.force_route("n5"):
            self.assertEqual(_counts(layer, x), {"n5": 1})
            self.assertEqual(_host_tensors(lambda: layer(x)), 0)


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

    def test_inputs_in_another_dtype(self):
        """A layer with `autocast=False` leaves its inputs uncast, and a
        `matrix` route serves only inputs in the compute dtype."""
        layer = _float_layer(ROUTABLE["dense"], "mixed_bfloat16")
        layer.autocast = False
        layer.quantize("int4", config=Int4QuantizationConfig(block_size=32))
        x = _random((3, 256))
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("n5"):
            self.assertEqual(_counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

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
        with _routes():
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
        """N5 runs eagerly only, so a compiled call with an N5 row runs
        the reference path."""
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        torch._dynamo.reset()
        call = torch.compile(lambda v: layer(v), backend="eager")
        kernels.route_counts.clear()
        with _routes({_row_key("float32", "1-16"): (("n5", None),)}):
            # The first call compiles. The calls with new values reuse its
            # code.
            outputs = [_numpy(call(ops.convert_to_tensor(x)))]
            with torch._dynamo.config.patch(error_on_recompile=True):
                for i in (1, 2):
                    outputs.append(_numpy(call(ops.convert_to_tensor(x + i))))
            self.assertEqual(dict(kernels.route_counts), {})
            self.assertEqual(_counts(layer, x), {"n5": 1})
        for i, y in enumerate(outputs):
            with kernels.policy("off"):
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
            with kernels.force_route("unknown"):
                pass
        self.assertEqual(self.state(), self.initial)

    def test_require_raises_without_a_kernel(self):
        """The test clears this backend's `"require"` kernels, because on
        some hosts the static entries of `_REQUIRED` give one: Triton K1
        on a CUDA host with Triton (torch), and on JAX, Mosaic GPU K1 on
        a GPU and Pallas TPU K1 on a TPU."""
        with _routes(required={}):
            with self.assertRaisesRegex(ValueError, "needs a platform with"):
                kernels.set_policy("require")
            with self.assertRaisesRegex(ValueError, "needs a platform with"):
                with kernels.policy("require"):
                    pass
        self.assertEqual(self.state(), self.initial)

    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy lists no devices."
    )
    def test_require_runs_the_platform_kernel(self):
        """`"require"` ignores the allow-list and the version range."""
        layer = _layer(ROUTABLE["dense"])
        refused = _layer(REFUSED["batch_axis"])
        x = _random(REFUSED["batch_axis"][2])
        required = {_device_type(): "untested"}
        static = {name: dict(row) for name, row in kernels._REQUIRED.items()}
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
        # The static entries of the platforms with a kernel stay.
        self.assertEqual(kernels._REQUIRED, static)


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
