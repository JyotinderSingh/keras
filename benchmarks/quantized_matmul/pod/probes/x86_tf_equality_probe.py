"""TensorFlow input gradients: autodiff against the custom gradient.

The TF CPU run of S6 (RunPod cpu3c, x86, TF 2.21) failed
`kernels_test.py::N5Test::test_calibrated_input_gradient_dense_{gptq,awq}`
in an `assertAllEqual`: the input gradient of a routed GPTQ or AWQ call
(custom gradient, `matmul(upstream, transpose(W))`) against the reference
path's (autodiff of `matmul(x, W)`, a `MatMul` with `transpose_b=True`).
With oneDNN on, the two run different gemm calls (weights `ab` and `ba`,
`ONEDNN_VERBOSE=1`); under Rosetta (AVX2) and with Eigen they give the
same bits, and on the cpu3c CPU they did not. The header prints the ISA
flags and whether oneDNN is on, so a run shows which kernels it used.

The probe prints, per case, whether the two sides are bit-identical, the
largest absolute difference, and the tolerance of the fix
(`reordered_sum_bound` in `quantization_test_utils.py`). It computes the
bound itself, so it runs on the trees before and after the fix.

    PYTHONPATH=<keras tree> KERAS_BACKEND=tensorflow \
        python x86_tf_equality_probe.py [--all]

The default runs the two failing cases (dense GPTQ and AWQ) and needs
only the Keras runtime dependencies. `--all` also runs the other tests
that the fix changes (`calibration_test.py::test_wrapped_reference`,
`lookup_test.py::test_reference_wrapped_is_the_reference` and
`::test_route_replaces_the_forward_only`); it imports those test modules,
so it needs `pytest` and `absl-py` installed (it does not run pytest).
Exit status: 0 when every check of the fix passes, 1 otherwise.
"""

import functools
import os
import platform
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import keras  # noqa: E402
from keras.src import layers  # noqa: E402
from keras.src import ops  # noqa: E402
from keras.src.quantizers import kernels  # noqa: E402
from keras.src.quantizers.quantization_test_utils import (  # noqa: E402
    calibrate_layer,
)
from keras.src.quantizers.quantization_test_utils import (  # noqa: E402
    calibration_config,
)
from keras.src.quantizers.quantization_test_utils import (  # noqa: E402
    input_gradient,
)

FAILURES = []


def reordered_sum_bound(magnitude, terms):
    """2 * gamma_n * magnitude, gamma_n = n u / (1 - n u), u = 2**-24."""
    unit = 2.0**-24
    gamma = terms * unit / (1 - terms * unit)
    return 2 * gamma * magnitude


def ulp(peak, dtype):
    nmant = ml_dtypes.finfo(dtype).nmant
    return 2.0 ** (np.floor(np.log2(peak)) - nmant)


def to_numpy(x):
    return ops.convert_to_numpy(ops.cast(x, "float32"))


def report(name, actual, expected, bound=None, old_exact=True):
    """One line: bit identity, max |diff|, mismatches, the fix's check."""
    actual, expected = to_numpy(actual), to_numpy(expected)
    same = np.array_equal(actual.view("int32"), expected.view("int32"))
    diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    mismatched = int(np.sum(actual != expected))
    line = (
        f"{name:58s} bit_identical={str(same):5s} max_abs={diff:.3e} "
        f"mismatched={mismatched}/{actual.size}"
    )
    if bound is None:
        if not same:
            FAILURES.append(name)
        print(line + "  [exact check] " + ("PASS" if same else "FAIL"))
        return
    ok = diff <= bound
    if not ok:
        FAILURES.append(name)
    old = "-" if not old_exact else ("PASS" if same else "FAIL")
    print(
        line + f" bound={bound:.3e} ratio={diff / bound:.4f} "
        f"old_exact={old} new={'PASS' if ok else 'FAIL'}"
    )


def environment():
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for row in f:
                if row.startswith("model name"):
                    print("cpu:", row.split(":", 1)[1].strip())
                if row.startswith("flags"):
                    flags = row
                    break
    except OSError:
        print("cpu:", platform.processor() or platform.machine())
    wanted = ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16")
    print("isa:", {name: f" {name}" in flags for name in wanted})
    try:
        from tensorflow.python.util import _pywrap_util_port

        onednn = _pywrap_util_port.IsMklEnabled()
    except Exception as error:  # Only a report line.
        onednn = f"unknown ({error})"
    print("keras:", keras.__file__)
    print("tensorflow:", tf.__version__, "| oneDNN enabled:", onednn)
    for key in ("TF_ENABLE_ONEDNN_OPTS", "ONEDNN_MAX_CPU_ISA"):
        print(f"{key}={os.environ.get(key, '<unset>')}")


def kernels_test_dense():
    """`N5Test.test_calibrated_input_gradient`, dense GPTQ and AWQ."""
    print("\n== kernels_test.py N5Test.test_calibrated_input_gradient_dense")
    for mode in ("gptq", "awq"):
        # `_float_layer` and `_calibrated` of `kernels_test.py`.
        layer = layers.Dense(96, use_bias=False, dtype="float32")
        layer.build((3, 256))
        rng = np.random.default_rng(0)
        layer._kernel.assign(rng.standard_normal(layer._kernel.shape))
        config = calibration_config(mode, group_size=128)
        rows = np.random.default_rng(4).standard_normal((16, 256))
        calibrate_layer(layer, config, rows.astype("f4"))
        x = np.random.default_rng(1).standard_normal((3, 256)).astype("f4")

        with kernels.force_route("reference_wrapped"):
            wrapped = input_gradient(layer, x)
        with kernels.force_route("n5"):
            n5 = input_gradient(layer, x)
        with kernels.policy("off"):
            off = input_gradient(layer, x)
        kernel = to_numpy(layer._quantized_weight().dequantize("float32"))
        # The output gradient is ones: entry `k` sums row `k` of the kernel.
        bound = reordered_sum_bound(
            np.max(np.sum(np.abs(kernel), axis=1)), kernel.shape[1]
        )
        report(f"{mode}: n5 vs reference_wrapped (exact)", n5, wrapped)
        report(f"{mode}: off vs reference_wrapped", off, wrapped, bound)
        # The two gemm calls alone, outside Keras.
        ones = tf.ones((3, kernel.shape[1]))
        transpose_b = tf.matmul(ones, kernel, transpose_b=True)
        transposed = tf.matmul(ones, tf.transpose(kernel))
        print(
            f"{'':4s}{mode}: MatMul(transpose_b) == MatMul(Transpose): "
            f"{np.array_equal(transpose_b.numpy(), transposed.numpy())}, "
            f"off == MatMul(transpose_b): "
            f"{np.array_equal(off, transpose_b.numpy())}, "
            f"wrapped == MatMul(Transpose): "
            f"{np.array_equal(wrapped, transposed.numpy())}"
        )


def calibration_wrapped_reference():
    """`CalibrationRouteTest.test_wrapped_reference`, every case."""
    from keras.src.quantizers.modes import calibration_test as ct

    print("\n== calibration_test.py test_wrapped_reference")
    cases = [(c, m, "float32") for c in ct.ROUTED for m in ct.ROUTED_MODES]
    cases += [(c, m, "mixed_bfloat16") for c in ct.ROUTED for m in ct.MODES]
    for case, mode, dtype in sorted(cases):
        layer, x = ct._routed_layer(case, mode, dtype)
        upstream = np.random.default_rng(1).standard_normal(
            to_numpy(layer(x)).shape
        )
        upstream = upstream.astype("float32")
        for compiled in (False, True):
            with kernels.policy("off"):
                y, expected = ct._vjp(layer, x, upstream, compiled)
            with kernels.force_route("reference_wrapped"):
                outputs, gradient = ct._vjp(layer, x, upstream, compiled)
            name = f"{case}/{mode}/{dtype}/{'jit' if compiled else 'eager'}"
            report(name + " outputs (exact)", outputs, y)
            peak = np.max(np.abs(expected))
            if dtype == "float32":
                view = layer._quantized_weight()
                kernel = view.dequantize("float32")
                if view.permutation is not None:
                    kernel = ops.transpose(kernel, view.permutation)
                rows = np.abs(to_numpy(kernel)).reshape(view.codes.shape[0], -1)
                magnitude = np.max(np.abs(upstream)) * np.max(rows.sum(axis=1))
                bound = reordered_sum_bound(magnitude, rows.shape[1])
                bound += 2 * ulp(peak, "float32")
            else:
                bound = 8 * ulp(peak, "bfloat16")
            # Before the fix only an eager `Dense` was compared exactly.
            old_exact = ct.ROUTED[case][0] is None and not compiled
            report(name + " gradient", gradient, expected, bound, old_exact)


def lookup_gradient_bound(lt, layer, upstream, gradient):
    """`_gradient_bound` of `lookup_test.py`."""
    dtype = layer.reverse_dtype or layer.compute_dtype
    stored, orientation = lt._stored(layer)
    table = kernels.reverse_lookup_view(stored, orientation)
    rows = np.abs(to_numpy(table.dequantize("float32")))
    upstream = np.abs(to_numpy(ops.cast(upstream, dtype)))
    magnitude = np.max(upstream) * np.max(np.sum(rows, axis=1))
    peak = np.max(np.abs(to_numpy(gradient)))
    return reordered_sum_bound(magnitude, rows.shape[1]) + 2 * ulp(peak, dtype)


def lookup_wrapped_reference():
    """`ReverseLookupRouteTest.test_reference_wrapped_is_the_reference`."""
    from keras.src.quantizers.modes import lookup_test as lt

    print("\n== lookup_test.py test_reference_wrapped_is_the_reference")
    for case in lt.ROUTED_CASES:
        layer = lt._reversible(
            case["tie_weights"],
            case["block_size"],
            case["policy"],
            case["reverse_dtype"],
            case["extra"],
        )
        x = lt._random((2, 3, lt.DIM), 1)
        upstream = lt._random((2, 3, lt.VOCAB), 2)
        call = functools.partial(layer, reverse=True)
        if case["in_model"]:
            call = lt.ReverseLookupModel(layer, dtype=case["policy"])
        for compiled in (False, True):
            with kernels.policy("off"):
                expected = lt._logits_and_gradient(call, x, upstream, compiled)
            with kernels.force_route("reference_wrapped"):
                actual = lt._logits_and_gradient(call, x, upstream, compiled)
            name = f"{case['testcase_name']}/{'fn' if compiled else 'eager'}"
            report(name + " logits (exact)", actual[0], expected[0])
            bound = lookup_gradient_bound(lt, layer, upstream, expected[1])
            report(name + " gradient", actual[1], expected[1], bound)


def lookup_route_replaces_the_forward_only():
    """`ReverseLookupRouteTest.test_route_replaces_the_forward_only`."""
    from keras.src.quantizers.modes import lookup_test as lt

    print("\n== lookup_test.py test_route_replaces_the_forward_only")
    for tie_weights in (True, False):
        for block_size in (64, -1):
            for policy in ("float32", "mixed_bfloat16"):
                layer = lt._reversible(tie_weights, block_size, policy)
                call = functools.partial(layer, reverse=True)
                x = lt._random((2, 3, lt.DIM), 1)
                upstream = lt._random((2, 3, lt.VOCAB), 2)
                with kernels.policy("off"):
                    _, expected = lt._logits_and_gradient(call, x, upstream)
                key = lt._row_key(layer.compute_dtype, "1-16")
                rows = {key: (("matrix", None),)}
                matrix = kernels.RouteSource(lt._matrix_logits)
                with (
                    kernels.policy("auto"),
                    lt._lookup_routes(rows, matrix=matrix),
                ):
                    _, gradient = lt._logits_and_gradient(call, x, upstream)
                bound = lookup_gradient_bound(lt, layer, upstream, expected)
                name = f"tied={tie_weights}/{block_size}/{policy}"
                report(name + " gradient", gradient, expected, bound)


if __name__ == "__main__":
    environment()
    kernels_test_dense()
    if "--all" in sys.argv[1:]:
        try:
            import absl.testing  # noqa: F401
            import pytest  # noqa: F401
        except ImportError as error:
            print(f"\n--all skipped: {error} (needs pytest and absl-py)")
        else:
            calibration_wrapped_reference()
            lookup_wrapped_reference()
            lookup_route_replaces_the_forward_only()
    print("\nfailed checks:", len(FAILURES))
    for name in FAILURES:
        print("  ", name)
    sys.exit(1 if FAILURES else 0)
