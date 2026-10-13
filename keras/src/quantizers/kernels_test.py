"""Route selection for a 4-bit weight-only contraction (`kernels.py`).

The library has one route in `keras.ops`, `"sum_then_scale"`, and the
kernel routes of the backend. The tests also run the reference path as a
route (`force_route("reference_wrapped")`), the `"matrix"` route, which
contracts the `(M, K)` inputs with the stored matrix, and the interpret
routes of the kernels on a CPU host, all from `kernels_test_utils`. The
route suites run each kernel route of the process through real layers.
The backend test files test the kernels as ops. The labels G1 to G8 are
those of `kernels_test_utils`.
"""

import dataclasses
import math
import os
import platform
import re
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
from keras.src import saving
from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common.backend_utils import M_BUCKET_NAMES
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.quantized_kernels import Kernel
from keras.src.backend.config import is_nnx_enabled
from keras.src.export.saved_model import ExportArchive
from keras.src.quantizers import kernels
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.kernels_test_utils import BUCKET_ROWS
from keras.src.quantizers.kernels_test_utils import admitted
from keras.src.quantizers.kernels_test_utils import counts
from keras.src.quantizers.kernels_test_utils import device_type
from keras.src.quantizers.kernels_test_utils import direct
from keras.src.quantizers.kernels_test_utils import first_tested_route
from keras.src.quantizers.kernels_test_utils import floor_and_bound
from keras.src.quantizers.kernels_test_utils import kernel_routes
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

    from keras.src.backend.jax.ops import quantized as jax_quantized

    if is_nnx_enabled():
        from flax import nnx
elif backend.backend() == "tensorflow":
    import tensorflow as tf
elif backend.backend() == "torch":
    import torch
    from torch._dynamo.testing import CompileCounter
    from torch.utils._python_dispatch import TorchDispatchMode

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


def _contiguous(layer):
    """Whether `g_idx[k] == k // G` for every stored row `k`. A view with
    no `g_idx` has these groups."""
    view = layer._quantized_weight()
    if view.g_idx is None:
        return True
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


class RouteSuiteTest(testing.TestCase):
    """Each route that a call admits, against the float64 oracle and the
    reference path: `"reference_wrapped"`, `"matrix"`, `"sum_then_scale"`
    and the kernel routes of this process (`kernels_test_utils.admitted`).
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
        """GPTQ and AWQ codes (uint8 storage). With groups that are not
        contiguous, the guard of a route that reads the group of row `k`
        as `k // G` gives the reference path's output."""
        layer = _calibrated(ROUTABLE[case], mode, dtype)
        x = _random(ROUTABLE[case][2])
        with route_tables():
            for name in self.routes(layer, x):
                y, reference = _check_route(self, layer, x, name)
                route = kernels._ROUTES[name]
                if route.contiguous_groups and not _contiguous(layer):
                    self.assertAllEqual(y, reference)

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

    @parameterized.named_parameters(
        named_product(dtype=["float32", "mixed_bfloat16"])
    )
    def test_guard_runs_the_reference_path(self, dtype):
        """GPTQ with activation order scatters the groups, so each route
        that reads the group of row `k` as `k // G` runs the reference
        path, eagerly and under `jax.jit`."""
        layer = _calibrated(ROUTABLE["dense_aligned"], "gptq_act_order", dtype)
        x = _random(ROUTABLE["dense_aligned"][2])
        self.assertFalse(_contiguous(layer))
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        names = [
            name
            for name in admitted(layer, x)
            if kernels._ROUTES[name].contiguous_groups
        ]
        self.assertIn("sum_then_scale", names)
        for name in names:
            with kernels.force_route(name):
                self.assertEqual(counts(layer, x), {name: 1})
                self.assertAllEqual(_numpy(layer(x)), expected)
                if backend.backend() == "jax":
                    # `ops.cond` traces both branches.
                    y = jax.jit(lambda v: layer(v))(x)
                    self.assertAllEqual(_numpy(y), expected)

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Compiles with `torch.compile`."
    )
    def test_torch_compiled_kernel_is_one_graph(self):
        """Under `torch.compile`, a float32 layer and a tied reverse lookup
        through each kernel route are one graph, as the reverse lookup of
        the wrapped reference path: the frontends read the float32
        precision as a constant and add no graph break."""
        layer = _layer(ROUTABLE["dense_ragged"], 32, bias=True)
        x = ops.convert_to_tensor(_random(ROUTABLE["dense_ragged"][2]))
        embedding = layers.ReversibleEmbedding(384, 256, tie_weights=True)
        embedding.build()
        embedding.quantize(
            "int4", config=Int4QuantizationConfig(block_size=128)
        )
        hidden = ops.convert_to_tensor(_random((2, 3, 256)))
        calls = [
            (name, lambda v: layer(v), x)
            for name in kernel_routes(kernels._ROUTES)
            if name in admitted(layer, x)
        ]
        calls += [
            (name, lambda v: embedding(v, reverse=True), hidden)
            for name in kernel_routes(kernels._LOGITS_ROUTES)
            if name in admitted(embedding, hidden, reverse=True)
        ]
        if not calls:
            self.skipTest("No kernel route of this process runs here.")
        calls.append(
            ("reference_wrapped", lambda v: embedding(v, reverse=True), hidden)
        )
        for name, call, inputs in calls:
            counter = CompileCounter()
            torch._dynamo.reset()
            with kernels.force_route(name):
                y = torch.compile(call, backend=counter)(inputs)
                expected = call(inputs)
            self.assertEqual(counter.frame_count, 1, name)
            self.assertAllEqual(_numpy(y), _numpy(expected))


class SumThenScaleTest(testing.TestCase):
    """The sum-then-scale route through `force_route`, and under `"auto"`
    with a row."""

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
            _check_route(
                self, layer, _random((m, rows), seed=m), "sum_then_scale"
            )

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
        _check_route(self, layer, _random((5, rows)), "sum_then_scale")

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
        with kernels.force_route("sum_then_scale"):
            self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
            self.assertAllEqual(_numpy(layer(x)), oracle)

    @parameterized.named_parameters(
        ("odd_columns", (None, 7, (2, 64)), 32),
        ("partial_group", (None, 10, (2, 200)), 32),
        ("no_rows", (None, 10, (0, 64)), 32),
    )
    def test_refusals(self, case, block_size):
        """Odd `N` and a partial last group fail the shape check, and
        `M = 0` the row check, so the reference path runs."""
        layer = _layer(case, block_size)
        x = _random(case[2])
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        with kernels.force_route("sum_then_scale"):
            self.assertEqual(counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

    @parameterized.named_parameters(
        ("int4", None), ("gptq", "gptq"), ("awq", "awq")
    )
    def test_auto_row(self, mode):
        """Under `"auto"`, a row runs the route in its M bucket, and the
        layer output equals a direct call of the route (G8)."""
        case = ROUTABLE["dense"]
        layer = _layer(case) if mode is None else _calibrated(case, mode)
        with route_tables({row_key("float32", "1-16"): ("sum_then_scale",)}):
            x = _random((4, 256))
            self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
            self.assertAllEqual(_numpy(layer(x)), _numpy(direct(layer, x)))
            self.assertEqual(
                counts(layer, _random((17, 256))), {"reference": 1}
            )

    @parameterized.named_parameters(("grouped", 32), ("per_channel", -1))
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Compiles with `jax.jit`."
    )
    def test_jax_jit_runs_the_route(self, block_size):
        layer = _layer(ROUTABLE["dense"], block_size)
        x = _random(ROUTABLE["dense"][2])
        oracle = _oracle(layer, x)
        with kernels.policy("off"):
            reference = _numpy(layer(x))
        kernels.route_counts.clear()
        with kernels.force_route("sum_then_scale"):
            y = _numpy(jax.jit(lambda v: layer(v))(x))
        self.assertEqual(dict(kernels.route_counts), {"sum_then_scale": 1})
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
        with kernels.force_route("sum_then_scale"):
            first = _numpy(layer(x))
            jax.monitoring.register_event_duration_secs_listener(count)
            try:
                second = _numpy(layer(x))
            finally:
                jax.monitoring.unregister_event_duration_listener(count)
        self.assertEqual(dict(kernels.route_counts), {"sum_then_scale": 2})
        self.assertEqual(compiles, [])
        self.assertAllEqual(second, first)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Traces a `tf.function`."
    )
    def test_tensorflow_graphs_run_the_reference_path(self):
        """The route runs eagerly only on TensorFlow."""
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        for jit_compile in (False, True):
            call = tf.function(lambda v: layer(v), jit_compile=jit_compile)
            kernels.route_counts.clear()
            with kernels.force_route("sum_then_scale"):
                call(x)
            self.assertEqual(dict(kernels.route_counts), {"reference": 1})

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Builds a DTensor variable."
    )
    def test_dtensor_codes_run_the_reference_path(self):
        """The distribution check reads the value of a variable that the
        view of a GPTQ call holds."""
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
            with route_tables(
                {row_key("float32", "1-16"): ("sum_then_scale",)}
            ):
                route = kernels.select_route(layer, geometry, view, x)
                self.assertEqual(route.name, "sum_then_scale")
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
        """A call of the route and of its group guard makes no tensor from
        host data, which a CUDA graph capture refuses."""
        case = ROUTABLE["dense"]
        if mode is None:
            layer = _layer(case, block_size)
        else:
            layer = _calibrated(case, mode)
        x = ops.convert_to_tensor(_random(case[2]))
        with kernels.force_route("sum_then_scale"):
            self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
            self.assertEqual(_host_tensors(lambda: layer(x)), 0)


# Modes whose mode fixes the groups, so that their view has no `g_idx`:
# `None` is int4.
FIXED = (None, "awq", "gptq_per_channel")


def _quantized(case, mode):
    return _layer(case) if mode is None else _calibrated(case, mode)


def _other_groups(layer):
    """The groups of the layer's `g_idx` in another order."""
    groups = np.asarray(_numpy(layer.g_idx))
    if not groups.any():
        groups[:3] = [1.0, 2.0, 3.0]
        return groups
    return np.random.default_rng(3).permutation(groups)


def _set_g_idx(layer, g_idx):
    weights = layer.get_weights()
    weights[[v.name for v in layer.weights].index("g_idx")] = g_idx
    layer.set_weights(weights)


def _model(layer, case):
    return models.Sequential([layers.Input(case[2][1:]), layer])


class GroupIndexTest(testing.TestCase):
    """int4, AWQ and GPTQ with one group fix their groups, so their view
    has no `g_idx`, no path reads the stored one, and no route needs the
    guard. GPTQ with several groups reads `g_idx` and keeps the guard on
    the value of each call."""

    @parameterized.named_parameters(
        ("int4", "dense", None),
        ("awq", "dense", "awq"),
        ("gptq_one_group", "dense", "gptq_per_channel"),
        ("gptq_one_group_einsum", "qkv_dnh", "gptq_per_channel"),
        ("int4_reversible_tied", "reversible_tied", None),
        ("int4_reversible_untied", "reversible_untied", None),
    )
    def test_no_path_reads_the_stored_g_idx(self, case, mode):
        """The load-bearing invariant of a mode that fixes its groups: no
        path reads the stored `g_idx`. Nothing checks a write of it, so a
        write of other groups through any path (`assign`, `set_weights`,
        a load, a torch `load_state_dict`, a TF checkpoint restore, an
        NNX update) must change no output. Here the write skips `assign`,
        and it changes neither the reference path nor the sum-then-scale
        route, bit for bit. A
        grouped int4 `ReversibleEmbedding` runs its forward and its
        reverse lookup, which have no route in `keras.ops`."""
        if case.startswith("reversible"):
            layer = layers.ReversibleEmbedding(
                50, 64, tie_weights=case == "reversible_tied"
            )
            layer.build()
            config = Int4QuantizationConfig(block_size=16)
            layer.quantize("int4", config=config)
            strategy = strategy_registry.get_strategy(layer.quantization_mode)
            views = strategy.quantized_weights(layer)
            self.assertLen(views, 1 if layer.tie_weights else 2)
            x = np.random.default_rng(2).integers(0, 50, (4, 7))

            def call():
                hidden = layer(x)
                return [hidden, layer(hidden, reverse=True)]

        else:
            case = ROUTABLE[case]
            layer = _quantized(case, mode)
            views = (layer._quantized_weight(),)
            x = _random(case[2])

            def call():
                return [layer(x)]

        for view in views:
            self.assertIsNone(view.g_idx)

        def outputs():
            with kernels.policy("off"):
                reference = [_numpy(y) for y in call()]
            if len(reference) > 1:
                return reference
            with kernels.force_route("sum_then_scale"):
                self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
                return reference + [_numpy(layer(x))]

        before = outputs()
        other = _other_groups(layer)
        layer.g_idx._direct_assign(ops.convert_to_tensor(other, "float32"))
        self.assertAllEqual(_numpy(layer.g_idx), other)
        for y, expected in zip(outputs(), before):
            self.assertAllEqual(y, expected)

    @pytest.mark.skipif(backend.backend() != "jax", reason="Reads a jaxpr.")
    def test_jit_traces_the_guard_only_where_g_idx_is_read(self):
        case = ROUTABLE["dense"]
        x = _random(case[2])
        for mode in FIXED + ("gptq", "gptq_act_order"):
            layer = _quantized(case, mode)
            with kernels.force_route("sum_then_scale"):
                jaxpr = str(jax.make_jaxpr(lambda v: layer(v))(x))
            guarded = mode in ("gptq", "gptq_act_order")
            self.assertEqual(
                layer._quantized_weight().g_idx is not None, guarded
            )
            self.assertEqual("cond[" in jaxpr, guarded, msg=mode)

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Runs `stateless_call` in `jit`."
    )
    def test_stateless_call_reads_the_g_idx_it_is_given(self):
        """GPTQ: the guard reads the `g_idx` of the call on the device.
        The state of an activation-order model, given to a layer whose own
        groups are contiguous, gives the output of the reference path, in
        an eager call and under `jax.jit`."""
        case = ROUTABLE["dense"]
        source = _calibrated(case, "gptq_act_order")
        layer = _calibrated(case, "gptq")
        self.assertTrue(_contiguous(layer))
        self.assertFalse(_contiguous(source))
        x = _random(case[2])
        with kernels.policy("off"):
            expected = _numpy(source(x))
        state = [v.value for v in source.non_trainable_variables]

        def call(state, v):
            return layer.stateless_call([], state, v)[0]

        with kernels.force_route("sum_then_scale"):
            eager = _numpy(call(state, x))
            jitted = _numpy(jax.jit(call)(state, x))
        for y in (eager, jitted):
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertAllEqual(y, expected)

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Compiles a JAX executable."
    )
    def test_compiled_executable_after_a_write(self):
        """An executable compiled before a write. For GPTQ, after a
        `set_weights` from an activation-order model, it gives the output
        of the reference path. For int4, AWQ and GPTQ with one group, a
        `set_weights` of other groups passes and the output does not
        change."""
        case = ROUTABLE["dense"]
        x = _random(case[2])
        source = _calibrated(case, "gptq_act_order")
        for mode in FIXED + ("gptq",):
            layer = _quantized(case, mode)

            def call(state, v):
                return layer.stateless_call([], state, v)[0]

            def state():
                return [v.value for v in layer.non_trainable_variables]

            with kernels.force_route("sum_then_scale"):
                compiled = jax.jit(call).lower(state(), x).compile()
            with kernels.policy("off"):
                reference = jax.jit(call).lower(state(), x).compile()
            before = _numpy(compiled(state(), x))
            if mode == "gptq":
                layer.set_weights(source.get_weights())
                y = _numpy(compiled(state(), x))
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertAllClose(y, _numpy(reference(state(), x)))
                self.assertNotAllClose(y, before)
                continue
            _set_g_idx(layer, _other_groups(layer))
            self.assertAllEqual(_numpy(compiled(state(), x)), before)

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Exports with `jax.export`."
    )
    def test_exported_artifact_with_other_state(self):
        """A `jax.export` artifact of a GPTQ layer whose groups are
        contiguous holds the guard, so the state of an activation-order
        model gives the output of the reference path."""
        case = ROUTABLE["dense"]
        source = _calibrated(case, "gptq_act_order")
        layer = _calibrated(case, "gptq")
        x = _random(case[2])
        with kernels.policy("off"):
            expected = _numpy(source(x))

        def call(state, v):
            return layer.stateless_call([], state, v)[0]

        state = [v.value for v in layer.non_trainable_variables]
        specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in state]
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        with kernels.force_route("sum_then_scale"):
            exported = jax.export.export(jax.jit(call))(specs, x_spec)
        source_state = [v.value for v in source.non_trainable_variables]
        y = _numpy(exported.call(source_state, x))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAllClose(y, expected)

    @pytest.mark.skipif(
        backend.backend() != "jax" or not is_nnx_enabled(),
        reason="Runs NNX updates.",
    )
    def test_nnx_update_with_other_state(self):
        """`nnx.update` writes the state of another model without
        `assign`. GPTQ with the state of an activation-order model gives
        the output of the reference path, eager, under `nnx.jit` and in
        `predict`. For int4 and AWQ, other groups in the stored `g_idx`
        change no output."""
        case = ROUTABLE["dense"]
        x = _random(case[2])
        source = _model(_calibrated(case, "gptq_act_order"), case)
        target = _model(_calibrated(case, "gptq"), case)
        with kernels.policy("off"):
            expected = _numpy(source(x))
        nnx.update(target, nnx.state(source))
        call = nnx.jit(lambda model, v: model(v))
        with kernels.force_route("sum_then_scale"):
            outputs = (
                _numpy(target(x)),
                _numpy(call(target, x)),
                target.predict(x, verbose=0),
            )
        for y in outputs:
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertAllClose(y, expected)
        for mode in (None, "awq"):
            model = _model(_quantized(case, mode), case)
            with kernels.force_route("sum_then_scale"):
                before = (_numpy(model(x)), _numpy(call(model, x)))
            other = _model(_quantized(case, mode), case)
            other_layer = other.layers[0]
            other_layer.g_idx._direct_assign(
                ops.convert_to_tensor(_other_groups(other_layer), "float32")
            )
            nnx.update(model, nnx.state(other))
            self.assertAllEqual(
                _numpy(model.layers[0].g_idx), _numpy(other_layer.g_idx)
            )
            with kernels.force_route("sum_then_scale"):
                self.assertAllEqual(_numpy(model(x)), before[0])
                self.assertAllEqual(_numpy(call(model, x)), before[1])

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Counts JAX compilations."
    )
    def test_load_does_not_trace_other_models_again(self):
        """A GPTQ load into one model compiles nothing again for another
        GPTQ model, whose sum-then-scale route holds the guard: no state
        outside the variables describes the groups."""
        case = ROUTABLE["dense"]
        x = _random(case[2])
        model = _model(_calibrated(case, "gptq"), case)
        with kernels.force_route("sum_then_scale"):
            model.predict(x, verbose=0)
        saved = _model(_calibrated(case, "gptq_act_order"), case)
        path = os.path.join(self.get_temp_dir(), "gptq.keras")
        saved.save(path)
        compiles = []

        def count(event, duration_secs, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles.append(duration_secs)

        saving.load_model(path)
        jax.monitoring.register_event_duration_secs_listener(count)
        try:
            with kernels.force_route("sum_then_scale"):
                model.predict(x, verbose=0)
        finally:
            jax.monitoring.unregister_event_duration_listener(count)
        self.assertEqual(compiles, [])

    def test_activation_order_round_trip(self):
        """A saved GPTQ model in activation order loads its `g_idx`, and
        the sum-then-scale route under the guard gives the output of the
        reference path."""
        case = ROUTABLE["dense"]
        layer = _calibrated(case, "gptq_act_order")
        model = _model(layer, case)
        path = os.path.join(self.get_temp_dir(), "act_order.keras")
        model.save(path)
        loaded = saving.load_model(path)
        x = _random(case[2])
        self.assertAllEqual(_numpy(loaded.layers[0].g_idx), _numpy(layer.g_idx))
        with kernels.policy("off"):
            expected = _numpy(model(x))
        with kernels.force_route("sum_then_scale"):
            y = _numpy(loaded(x))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAllEqual(y, expected)


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
        with kernels.force_route("sum_then_scale"):
            self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
            self.assertAllClose(_numpy(layer(x)), expected, atol=1e-5)

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
        with kernels.force_route("sum_then_scale"):
            self.assertEqual(counts(layer, x), {"reference": 1})
            self.assertAllEqual(_numpy(layer(x)), expected)

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
        backend.backend() != "torch", reason="Compiles with `torch.compile`."
    )
    def test_torch_compile_does_not_recompile(self):
        """The sum-then-scale route runs eagerly only, so a compiled call
        with its row runs the reference path."""
        layer = _layer(ROUTABLE["dense"])
        x = _random(ROUTABLE["dense"][2])
        torch._dynamo.reset()
        call = torch.compile(lambda v: layer(v), backend="eager")
        kernels.route_counts.clear()
        with route_tables({row_key("float32", "1-16"): ("sum_then_scale",)}):
            # The first call compiles. The calls with new values reuse its
            # code.
            outputs = [_numpy(call(ops.convert_to_tensor(x)))]
            with torch._dynamo.config.patch(error_on_recompile=True):
                for i in (1, 2):
                    outputs.append(_numpy(call(ops.convert_to_tensor(x + i))))
            self.assertEqual(dict(kernels.route_counts), {})
            self.assertEqual(counts(layer, x), {"sum_then_scale": 1})
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

    @parameterized.named_parameters(("matmul", False), ("logits", True))
    def test_device_check(self, reverse):
        """On a device that a kernel does not support, a measured row of
        its route runs the reference path. On a device that no kernel
        supports, `"require"` raises and names the requirements of each
        kernel."""
        routes = kernels._LOGITS_ROUTES if reverse else kernels._ROUTES
        device_class = kernels._device_class()
        refused = [
            name
            for name, route in routes.items()
            if route.kernel is not None
            and not route.kernel.supports_device(device_class)
        ]
        if not refused:
            self.skipTest("No kernel of this process refuses the device.")
        if reverse:
            layer = layers.ReversibleEmbedding(
                256, 256, tie_weights=True, dtype="mixed_bfloat16"
            )
            layer.build()
            layer.quantize(
                "int4", config=Int4QuantizationConfig(block_size=128)
            )
        else:
            layer = _layer(ROUTABLE["dense_aligned"], 128, "mixed_bfloat16")
        x = _random((3, 256))
        call_kwargs = {"reverse": True} if reverse else {}
        for name in refused:
            key = "logits_rows" if reverse else "rows"
            rows = {key: {row_key("bfloat16", "1-16"): (name,)}}
            with route_tables(**rows):
                selected = counts(layer, x, **call_kwargs)
            self.assertEqual(selected, {"reference": 1})
        if kernels._required(kernels._ROUTES, device_class):
            return
        for name in refused:
            requirements = re.escape(routes[name].kernel.requirements)
            with self.assertRaisesRegex(ValueError, requirements):
                with kernels.policy("require"):
                    pass

    def test_rows_can_pass_their_gates(self):
        """Each row of both measured tables has a backend, a device class of
        `_device_class`, a compute dtype and an M bucket. Its routes serve
        that backend, device class and compute dtype, so
        a row cannot name a cell that its routes refuse. The device classes
        of each kernel are written out here: a backend registers only its
        own kernels, and a kernel's device check also reads its library."""
        buckets = M_BUCKET_NAMES
        mosaic_gpu = {"cuda-sm80", "cuda-sm86", "cuda-sm89", "cuda-sm90"}
        # Kernel -> (its backend, whether it serves a device class).
        kernel_classes = {
            "triton": (
                "torch",
                lambda c: c.startswith("cuda-sm") and int(c[7:]) >= 80,
            ),
            "mosaic_gpu": ("jax", lambda c: c in mosaic_gpu),
        }
        backends = ("jax", "numpy", "tensorflow", "torch")
        device_classes = r"^(cuda-sm\d+|tpu-.+|mps|cpu-.+|gpu)$"
        # The reverse lookup has kernel routes only.
        tables = (
            (
                kernels._MEASURED_ROWS,
                kernels._ROUTES,
                ("sum_then_scale", *kernel_classes),
            ),
            (
                kernels._MEASURED_LOGITS_ROWS,
                kernels._LOGITS_ROUTES,
                kernel_classes,
            ),
        )
        for measured_rows, sources, names in tables:
            for key, routes in measured_rows.items():
                backend_name, device_class, dtype, bucket = key
                self.assertIn(backend_name, backends)
                self.assertRegex(device_class, device_classes)
                self.assertIn(dtype, ("bfloat16", "float16", "float32"))
                self.assertIn(bucket, buckets)
                self.assertNotEqual(len(routes), 0, key)
                for name in routes:
                    self.assertIn(name, names, key)
                    if name in kernel_classes:
                        owner, serves = kernel_classes[name]
                        self.assertEqual(backend_name, owner, key)
                        self.assertTrue(serves(device_class), key)
                    # A source that this process does not register (Triton
                    # off torch) has no dtypes to read.
                    source = sources.get(name)
                    if source is not None and source.dtypes:
                        self.assertIn(dtype, source.dtypes, key)

    def test_rows_of_this_platform(self):
        """Under `"auto"`, each row of this backend and device runs its
        first route that this process has in its tested version range,
        and the layer output equals a direct call of the route (G8). With
        no such route, the reference path runs. The layer passes the
        shape gates of every route: `K = 256`, `N = 128` and groups of
        128."""
        platform_key = row_key("float32", "1-16")[:2]
        rows = {
            key: routes
            for key, routes in kernels._MEASURED_ROWS.items()
            if key[:2] == platform_key
        }
        if not rows:
            self.skipTest("No measured row has this platform.")
        policy = {"bfloat16": "mixed_bfloat16", "float16": "mixed_float16"}
        for (_, _, dtype, bucket), routes in rows.items():
            with self.subTest(dtype=dtype, bucket=bucket):
                case = (None, 128, (BUCKET_ROWS[bucket], 256))
                layer = _layer(case, 128, policy.get(dtype, dtype))
                x = _random(case[2])
                expected = first_tested_route(routes, kernels._ROUTES)
                # Only `"auto"` reads the measured rows.
                with kernels.policy("auto"):
                    self.assertEqual(counts(layer, x), {expected: 1})
                    y = _numpy(layer(x))
                    if expected != "reference":
                        self.assertAllEqual(y, _numpy(direct(layer, x)))
                if expected == "reference":
                    with kernels.policy("off"):
                        self.assertAllEqual(y, _numpy(layer(x)))

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="The Mosaic GPU kernels need JAX."
    )
    def test_kernel_tests_run_where_rows_are_measured(self):
        """On a JAX host with measured rows, the Mosaic GPU kernel tests run.
        `quantized_test.py` and `quantized_logits_test.py` skip without
        `plgpu.mma`, which needs JAX 0.11.2 or newer, so Python 3.12 or
        newer. On an older Python, this test skips and names it."""
        platform_key = row_key("float32", "1-16")[:2]
        tables = (kernels._MEASURED_ROWS, kernels._MEASURED_LOGITS_ROWS)
        if not any(
            key[:2] == platform_key for table in tables for key in table
        ):
            self.skipTest("No measured row has this platform.")
        if jax_quantized.plgpu.available:
            return
        python = platform.python_version()
        if sys.version_info < (3, 12):
            self.skipTest(
                f"JAX 0.11.2 and newer, which have `plgpu.mma`, need Python "
                f"3.12 or newer. Installed: Python {python}."
            )
        self.fail(
            f"The measured rows name {platform_key[1]!r}, but JAX "
            f"{jax.__version__} has no `plgpu.mma`, so the Mosaic GPU kernel "
            "tests skip. Install JAX 0.11.2 or newer."
        )


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
        """The test takes this backend's kernels out of the route table,
        because a GPU host can have one: the Triton kernels on a CUDA host
        with Triton (torch), and the Mosaic GPU kernels on JAX."""
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
