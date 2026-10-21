"""The calibration modes' shared chassis, tested once per mode."""

import math
import os
import warnings

import ml_dtypes
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers import kernels
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.kernels_test import _numpy
from keras.src.quantizers.kernels_test import _routes
from keras.src.quantizers.quantization_test_utils import assert_calibrated
from keras.src.quantizers.quantization_test_utils import assert_float
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import calibration_statistic
from keras.src.testing.test_utils import named_product

if backend.backend() == "jax":
    import jax
elif backend.backend() == "tensorflow":
    import tensorflow as tf
elif backend.backend() == "torch":
    import torch

MODES = ["gptq", "awq"]


def _config(mode, **kwargs):
    if mode == "awq":
        kwargs.setdefault("num_grid_points", 5)
        kwargs.setdefault("apply_clip", False)
    return calibration_config(mode, **kwargs)


def _calibrate(layer, mode, x, **kwargs):
    """Calibrates the float `layer` with `mode` on `x`.

    The calibrator's solve swaps the layer for the quantized one. Returns
    the calibrator.
    """
    return calibrate_layer(layer, _config(mode, **kwargs), x)


def _float_dense(rng):
    """A built float `Dense(16)` on 12 features, and 32 rows of inputs."""
    layer = layers.Dense(16)
    layer.build((None, 12))
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    return layer, rng.standard_normal((32, 12)).astype("float32")


def _einsum(equation, output_shape, input_shape, kernel=None):
    layer = layers.EinsumDense(equation, output_shape=output_shape)
    layer.build(input_shape)
    if kernel is not None:
        layer.kernel.assign(kernel)
    return layer


def _dense(kernel):
    layer = layers.Dense(kernel.shape[1])
    layer.build((None, kernel.shape[0]))
    layer.kernel.assign(kernel)
    return layer


def _dequantized(layer):
    return ops.convert_to_numpy(layer._quantized_weight().dequantize("float32"))


def _assert_relatively_close(test, actual, desired, tolerance=0.1):
    """`actual` is within relative L2 distance `tolerance` of `desired`."""
    actual = ops.convert_to_numpy(actual)
    desired = ops.convert_to_numpy(desired)
    error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)
    test.assertLess(error, tolerance)


class AxesGeometry(ProjectionGeometry):
    """`AxesProjection`'s geometry: declared kernel axes, its own einsum."""

    @property
    def kernel_axes(self):
        return self.layer.declared_axes

    def contraction_view(self):
        if self.layer.input_batch_axes is None:
            return super().contraction_view()
        return ContractionView(
            self.weight_shape,
            self.kernel_axes,
            input_batch_axes=self.layer.input_batch_axes,
            input_contracted_axes=(-1,),
        )

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)


class AxesProjection(layers.Layer):
    """A third-party N-D projection written to the quantization protocol.

    It is not an `EinsumDense`: its geometry declares the roles of the
    kernel's axes by hand and contracts with `equation`, and the layer
    lists the calibration modes in its `variable_serialization_spec`.
    With `input_batch_axes`, the geometry also overrides
    `contraction_view` to name the input axes of the kernel's batch axes.
    """

    def __init__(
        self,
        equation,
        kernel_shape,
        kernel_axes,
        input_batch_axes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        self.declared_shape = tuple(kernel_shape)
        self.declared_axes = kernel_axes
        self.input_batch_axes = input_batch_axes
        self.bias = None
        self.activation = None

    def build(self, input_shape):
        self.kernel_shape = self.declared_shape
        if self.quantization_mode:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if not self._strategy_owns_weight_storage():
            self._kernel = self.add_weight(
                name="kernel", shape=self.kernel_shape
            )

    @property
    def kernel(self):
        quantized_weight = self._quantized_weight()
        if quantized_weight is None:
            return self._kernel
        return quantized_weight.unpack()

    def call(self, inputs):
        return ops.einsum(self.equation, inputs, self._kernel)

    def _quantization_geometry(self):
        return AxesGeometry(self)

    @property
    def variable_serialization_spec(self):
        stored = ["quantized_kernel", "kernel_scale", "kernel_zero"]
        return {
            None: ["kernel"],
            "gptq": stored + ["g_idx"],
            "awq": stored + ["awq_scales", "g_idx"],
        }

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")


class CalibrationSwapTest(testing.TestCase):
    """A calibration mode quantizes a float layer in one step."""

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_calibrated_layer_holds_no_config(self, mode):
        # The policy names the bit width and the group size; the run's
        # config, with its dataset, stays with the caller.
        rng = np.random.default_rng(0)
        layer, x = _float_dense(rng)
        weights = [w.path for w in layer.weights]
        _calibrate(layer, mode, x, weight_bits=4, group_size=4)
        self.assertEqual(layer.dtype_policy.name, f"{mode}/4/4_from_float32")
        assert_calibrated(self, layer, mode)
        self.assertIsNone(layer.get_config()["quantization_config"])
        # The bias keeps its place; the codes and their parameters follow.
        self.assertEqual(layer.weights[0].path, weights[1])
        self.assertLen(layer.weights, 5 if mode == "gptq" else 6)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_config_from_a_file_is_dropped(self, mode):
        # A config deserialized with a calibrated layer cannot change what
        # the policy name says: the variables follow the policy, and the
        # layer serializes no config again.
        rng = np.random.default_rng(0)
        layer, x = _float_dense(rng)
        _calibrate(layer, mode, x, weight_bits=4, group_size=4)
        layer_config = layer.get_config()
        layer_config["quantization_config"] = saving.serialize_keras_object(
            _config(mode, weight_bits=8 if mode == "gptq" else 4, group_size=2)
        )
        loaded = layers.Dense.from_config(layer_config)
        loaded.build((None, 12))
        self.assertIsNone(loaded.quantization_config)
        self.assertIsNone(loaded.get_config()["quantization_config"])
        self.assertEqual(loaded.kernel_scale.shape, layer.kernel_scale.shape)
        loaded.set_weights(layer.get_weights())
        self.assertAllEqual(loaded(x), layer(x))

    @parameterized.named_parameters(
        named_product(mode=MODES, kind=["dense", "einsum"])
    )
    def test_config_is_dropped_at_build(self, mode, kind):
        # Under `<mode>/4/8`, a layer lays out 4-bit codes in groups of 8
        # rows, whatever the config it was constructed with says.
        config = _config(
            mode, weight_bits=8 if mode == "gptq" else 4, group_size=4
        )
        dtype = f"{mode}/4/8_from_float32"
        if kind == "dense":
            layer = layers.Dense(8, dtype=dtype, quantization_config=config)
        else:
            layer = layers.EinsumDense(
                "ab,bc->ac",
                output_shape=(8,),
                dtype=dtype,
                quantization_config=config,
            )
        layer.build((None, 16))
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(tuple(layer.quantized_kernel.shape), (16, 4))
        self.assertEqual(tuple(layer.kernel_scale.shape), (2, 8))
        self.assertEqual(layer._quantized_weight().scheme.code_range, (0, 15))


class CalibrationEinsumLayoutTest(testing.TestCase):
    """GPTQ and AWQ split an einsum kernel by its equation."""

    @parameterized.named_parameters(
        named_product(
            [
                # Gemma's query projection `[heads, d_model, head_dim]`
                # against the same weights laid out `[d_model, heads,
                # head_dim]`.
                dict(
                    testcase_name="gemma_q",
                    equation="btd,ndh->btnh",
                    reference_equation="btd,dnh->btnh",
                    output_shape=(None, 2, 4),
                ),
                # A mixture-of-experts gate `[experts, d_model, inner]`
                # against `[d_model, experts, inner]`.
                dict(
                    testcase_name="expert_gate",
                    equation="btd,edi->btei",
                    reference_equation="btd,dei->btei",
                    output_shape=(None, 3, 6),
                ),
            ],
            mode=MODES,
        )
    )
    def test_kernel_only_axes_share_one_statistic(
        self, mode, equation, reference_equation, output_shape
    ):
        input_shape = (None, 5, 8)
        rng = np.random.default_rng(seed=3)
        layer = _einsum(equation, output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel)
        reference = _einsum(
            reference_equation,
            output_shape,
            input_shape,
            kernel=np.transpose(kernel, (1, 0, 2)),
        )
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        reference_calibrator = _calibrate(reference, mode, x, group_size=4)

        # One statistic over the model width, shared by every head or
        # expert, and the same weights up to the transpose.
        statistic = calibration_statistic(calibrator)
        self.assertEqual(ops.shape(statistic)[0], 8)
        self.assertAllClose(
            statistic, calibration_statistic(reference_calibrator)
        )
        self.assertAllClose(
            _dequantized(layer),
            np.transpose(_dequantized(reference), (1, 0, 2)),
        )
        self.assertAllClose(layer(x), reference(x))
        # Stored as `(d_model, packed columns)`, like the reference.
        columns = int(np.prod(output_shape[1:]))
        self.assertEqual(tuple(layer.quantized_kernel.shape), (8, columns // 2))
        self.assertEqual(tuple(layer.g_idx.shape), (8,))

    @parameterized.named_parameters(
        named_product(
            [
                # A QKV projection `[d_model, heads, head_dim]` is the
                # `Dense` kernel `(d_model, heads * head_dim)`.
                dict(
                    testcase_name="qkv",
                    equation="btd,dnh->btnh",
                    output_shape=(None, 2, 4),
                    input_shape=(None, 5, 8),
                ),
                # An attention output projection `[heads, head_dim,
                # d_model]` is the `Dense` kernel `(heads * head_dim,
                # d_model)`.
                dict(
                    testcase_name="attention_output",
                    equation="btnh,nhd->btd",
                    output_shape=(None, 8),
                    input_shape=(None, 5, 2, 4),
                ),
            ],
            mode=MODES,
        )
    )
    def test_contracted_axes_match_dense(
        self, mode, equation, output_shape, input_shape
    ):
        rng = np.random.default_rng(seed=5)
        layer = _einsum(equation, output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel)
        dense = _dense(kernel.reshape(8, 8))
        x = rng.standard_normal((2,) + input_shape[1:]).astype("float32")
        x_dense = x.reshape(2, 5, 8)
        calibrator = _calibrate(layer, mode, x, group_size=4)
        dense_calibrator = _calibrate(dense, mode, x_dense, group_size=4)

        self.assertAllClose(
            calibration_statistic(calibrator),
            calibration_statistic(dense_calibrator),
        )
        self.assertAllClose(
            _dequantized(layer).reshape(8, 8), _dequantized(dense)
        )
        self.assertAllClose(ops.reshape(layer(x), (2, 5, 8)), dense(x_dense))

    @parameterized.named_parameters(
        named_product(
            [
                dict(testcase_name="one_group", inner=6, group_size=-1),
                dict(testcase_name="grouped", inner=6, group_size=3),
                # The last group of every expert is shorter.
                dict(testcase_name="ragged", inner=7, group_size=3),
            ],
            mode=MODES,
        )
    )
    def test_batch_axis_calibrates_each_expert_alone(
        self, mode, inner, group_size
    ):
        # A mixture-of-experts down projection `[experts, inner, d_model]`:
        # the expert axis is shared by the inputs, so every expert is its
        # own problem, calibrated from its own slice of the inputs.
        experts, width = 3, 8
        kwargs = dict(group_size=group_size)
        if group_size != -1:
            # GPTQ's permuted groups and AWQ's clipping sample, per expert.
            if mode == "gptq":
                kwargs["activation_order"] = True
            else:
                kwargs["apply_clip"] = True
        rng = np.random.default_rng(seed=11)
        kernel = rng.standard_normal((experts, inner, width)).astype("float32")
        layer = _einsum(
            "btei,eid->bted",
            (None, experts, width),
            (None, 5, experts, inner),
            kernel=kernel,
        )
        x = rng.standard_normal((2, 5, experts, inner)).astype("float32")
        calibrator = _calibrate(layer, mode, x, **kwargs)
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        self.assertEqual(statistic.shape[:2], (experts, inner))

        dequantized = _dequantized(layer)
        outputs = ops.convert_to_numpy(layer(x))
        n_groups = 1 if group_size == -1 else -(-inner // group_size)
        g_idx = ops.convert_to_numpy(layer.g_idx).reshape(experts, inner)
        for expert in range(experts):
            dense = _dense(kernel[expert])
            x_expert = x[:, :, expert, :]
            dense_calibrator = _calibrate(
                dense, mode, x_expert.reshape(-1, inner), **kwargs
            )
            self.assertAllClose(
                statistic[expert], calibration_statistic(dense_calibrator)
            )
            if mode == "gptq":
                self.assertAllClose(dequantized[expert], _dequantized(dense))
                self.assertAllClose(outputs[:, :, expert, :], dense(x_expert))
            else:
                # The statistics above match exactly. AWQ's grid searches
                # then pick discrete candidates, and the batched and the
                # per-expert contractions sum in different orders, so a near
                # tie can pick a neighbouring scale for an expert.
                _assert_relatively_close(
                    self, dequantized[expert], _dequantized(dense)
                )
                _assert_relatively_close(
                    self, outputs[:, :, expert, :], dense(x_expert)
                )
            # The experts stack along the rows, each with its own groups.
            self.assertAllClose(
                g_idx[expert],
                ops.convert_to_numpy(dense.g_idx) + expert * n_groups,
            )
        self.assertEqual(
            tuple(layer.quantized_kernel.shape),
            (experts * inner, width // 2),
        )
        self.assertEqual(
            tuple(layer.kernel_scale.shape), (experts * n_groups, width)
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_two_batch_axes_follow_the_kernel_order(self, mode):
        # `bghc,hgcd->bghd`: the kernel orders its batch axes `[h, g]` and
        # the inputs `[g, h]`. Problem `(h, g)` sees `x[:, g, h, :]`.
        heads, groups, inner, width = 3, 2, 8, 4
        rng = np.random.default_rng(seed=29)
        layer = _einsum(
            "bghc,hgcd->bghd",
            (groups, heads, width),
            (None, groups, heads, inner),
        )
        kernel = ops.convert_to_numpy(layer.kernel)
        x = rng.standard_normal((16, groups, heads, inner)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        self.assertEqual(statistic.shape[:2], (heads * groups, inner))

        dequantized = _dequantized(layer)
        outputs = ops.convert_to_numpy(layer(x))
        for h in range(heads):
            for g in range(groups):
                dense = _dense(kernel[h, g])
                x_problem = x[:, g, h, :]
                dense_calibrator = _calibrate(
                    dense, mode, x_problem, group_size=4
                )
                self.assertAllClose(
                    statistic[h * groups + g],
                    calibration_statistic(dense_calibrator),
                )
                self.assertAllClose(dequantized[h, g], _dequantized(dense))
                self.assertAllClose(outputs[:, g, h, :], dense(x_problem))

    @parameterized.named_parameters(
        named_product(
            [
                # A value head: one problem of one column.
                dict(
                    testcase_name="value_head",
                    equation="btd,d->bt",
                    output_shape=(None,),
                    input_shape=(None, 5, 8),
                ),
                # A per-head gate: every head is a problem of one column.
                dict(
                    testcase_name="head_gate",
                    equation="btnh,nh->btn",
                    output_shape=(None, 3),
                    input_shape=(None, 5, 3, 8),
                ),
            ],
            mode=MODES,
        )
    )
    def test_kernel_without_a_free_axis_matches_dense(
        self, mode, equation, output_shape, input_shape
    ):
        rng = np.random.default_rng(seed=23)
        layer = _einsum(equation, output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel).reshape(-1, 8)
        heads = len(kernel)
        x = rng.standard_normal((2,) + input_shape[1:]).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        statistic = statistic.reshape(heads, 8, -1)
        self.assertEqual(tuple(layer.quantized_kernel.shape), (heads * 8, 1))

        dequantized = _dequantized(layer).reshape(heads, 8)
        outputs = ops.convert_to_numpy(layer(x)).reshape(2, 5, heads)
        x_heads = x.reshape(2, 5, heads, 8)
        for head in range(heads):
            dense = _dense(kernel[head].reshape(8, 1))
            x_head = x_heads[:, :, head, :]
            dense_calibrator = _calibrate(dense, mode, x_head, group_size=4)
            self.assertAllClose(
                statistic[head],
                ops.reshape(calibration_statistic(dense_calibrator), (8, -1)),
            )
            self.assertAllClose(dequantized[head], _dequantized(dense)[:, 0])
            self.assertAllClose(outputs[:, :, head], dense(x_head)[..., 0])

    def test_awq_clipping_search_is_per_expert(self):
        # Over two batches of 300 tokens, each expert's clipping search
        # scores its bounds with that expert's own Hessian, as a `Dense` of
        # that expert does.
        experts, inner, width = 2, 8, 4
        config = _config("awq", group_size=4, apply_clip=True)
        rng = np.random.default_rng(seed=31)
        batches = [
            rng.standard_normal((1, 300, experts, inner)).astype("float32")
            for _ in range(2)
        ]
        # Gaussian weights have tails for the clipping search to cut.
        kernel = rng.standard_normal((experts, inner, width)).astype("float32")

        def calibrated(config):
            layer = _einsum(
                "btei,eid->bted",
                (None, experts, width),
                (None, 300, experts, inner),
                kernel,
            )
            calibrate_layer(layer, config, *batches)
            return _dequantized(layer)

        dequantized = calibrated(config)
        self.assertNotAllClose(
            dequantized, calibrated(_config("awq", group_size=4))
        )
        for expert in range(experts):
            dense = _dense(kernel[expert])
            calibrate_layer(
                dense,
                config,
                *[b[:, :, expert, :].reshape(-1, inner) for b in batches],
            )
            self.assertAllClose(dequantized[expert], _dequantized(dense))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_batch_axis_groups_past_256_under_mixed_bfloat16(self, mode):
        # 4 experts of 80 groups each, numbered across the experts up to
        # 319. bfloat16 holds integers exactly only up to 256, so an
        # autocast `g_idx` would send later rows to another group.
        equation, output_shape = "btei,eid->bted", (None, 4, 4)
        input_shape = (None, 3, 4, 160)
        rng = np.random.default_rng(seed=17)
        layer = _einsum(equation, output_shape, input_shape)
        x = rng.standard_normal((2, 3, 4, 160)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=2)
        self.assertEqual(int(ops.max(layer.g_idx)), 319)
        expected = ops.convert_to_numpy(layer(x))

        store = {}
        layer.save_own_variables(store)
        policy = calibrator.config.dtype_policy_string()
        restored = layers.EinsumDense(
            equation,
            output_shape=output_shape,
            dtype=f"{policy}_from_mixed_bfloat16",
        )
        restored.build(input_shape)
        restored.load_own_variables(store)
        y = ops.convert_to_numpy(ops.cast(restored(x), "float32"))
        atol = 0.02 * np.abs(expected).max()
        self.assertAllClose(y, expected, rtol=0.02, atol=atol)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_permuted_layout_round_trips_through_variables(self, mode):
        rng = np.random.default_rng(seed=13)
        layer = _einsum("btd,ndh->btnh", (None, 2, 4), (None, 5, 8))
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=-1)
        y = ops.convert_to_numpy(layer(x))

        store = {}
        layer.save_own_variables(store)
        restored = layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, 2, 4),
            dtype=f"{calibrator.config.dtype_policy_string()}_from_float32",
        )
        restored.build((None, 5, 8))
        restored.load_own_variables(store)
        self.assertEqual(tuple(restored.quantized_kernel.shape), (8, 4))
        self.assertAllClose(restored(x), y)


class ThirdPartyGeometryTest(testing.TestCase):
    """A third-party N-D projection calibrates through its geometry."""

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_declared_axes_calibrate_like_einsum_dense(self, mode):
        # Gemma's query projection `[heads, d_model, head_dim]`, its axes
        # declared by hand, calibrates like the `EinsumDense` that derives
        # them from the equation, and round-trips through its variables.
        equation = "btd,ndh->btnh"
        kernel_axes = KernelAxes(contracted=(1,), free=(0, 2))
        rng = np.random.default_rng(seed=19)
        layer = AxesProjection(equation, (2, 8, 4), kernel_axes)
        layer.build((None, 5, 8))
        reference = _einsum(
            equation,
            (None, 2, 4),
            (None, 5, 8),
            kernel=ops.convert_to_numpy(layer.kernel),
        )
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        reference_calibrator = _calibrate(reference, mode, x, group_size=4)
        assert_calibrated(self, layer, mode)
        self.assertAllClose(
            calibration_statistic(calibrator),
            calibration_statistic(reference_calibrator),
        )
        self.assertAllClose(_dequantized(layer), _dequantized(reference))
        y = ops.convert_to_numpy(layer(x))
        self.assertAllClose(y, reference(x))
        self.assertEqual(tuple(layer.quantized_kernel.shape), (8, 4))

        store = {}
        layer.save_own_variables(store)
        restored = AxesProjection(
            equation, (2, 8, 4), kernel_axes, dtype=layer.dtype_policy.name
        )
        restored.build((None, 5, 8))
        restored.load_own_variables(store)
        self.assertAllClose(restored(x), y)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_batch_axis_needs_a_view_override(self, mode):
        # A mixture-of-experts down projection `[experts, inner, d_model]`.
        # The default view names no input batch axis, so the layer is
        # refused until its geometry overrides `contraction_view`; with the
        # inputs' expert axis named, it calibrates like `EinsumDense`.
        equation = "btei,eid->bted"
        kernel_axes = KernelAxes(contracted=(1,), free=(2,), batch=(0,))
        refused = AxesProjection(equation, (3, 6, 4), kernel_axes)
        refused.build((None, 5, 3, 6))
        with self.assertRaisesRegex(
            NotImplementedError, "must override `contraction_view`"
        ):
            refused.quantize(mode, config=_config(mode, group_size=-1))
        assert_float(self, refused)

        rng = np.random.default_rng(seed=37)
        layer = AxesProjection(
            equation, (3, 6, 4), kernel_axes, input_batch_axes=(2,)
        )
        layer.build((None, 5, 3, 6))
        reference = _einsum(
            equation,
            (None, 3, 4),
            (None, 5, 3, 6),
            kernel=ops.convert_to_numpy(layer.kernel),
        )
        x = rng.standard_normal((2, 5, 3, 6)).astype("float32")
        _calibrate(layer, mode, x, group_size=-1)
        _calibrate(reference, mode, x, group_size=-1)
        self.assertAllClose(_dequantized(layer), _dequantized(reference))
        self.assertAllClose(layer(x), reference(x))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_view_batch_axes_must_match_the_inputs(self, mode):
        # The override names the token axis (5) for the 3 experts.
        layer = AxesProjection(
            "btei,eid->bted",
            (3, 6, 4),
            KernelAxes(contracted=(1,), free=(2,), batch=(0,)),
            input_batch_axes=(1,),
        )
        layer.build((None, 5, 3, 6))
        x = np.ones((2, 5, 3, 6), dtype="float32")
        with self.assertRaisesRegex(ValueError, "has 3 problems"):
            _calibrate(layer, mode, x)
        assert_float(self, layer)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_mis_declared_axes_raise(self, mode):
        # Axis 0 of the kernel has no role. The author's mistake reaches
        # the caller instead of reading as a layer without support.
        layer = AxesProjection(
            "btd,ndh->btnh", (2, 8, 4), KernelAxes(contracted=(1,), free=(2,))
        )
        layer.build((None, 5, 8))
        with self.assertRaisesRegex(ValueError, "partition the kernel axes"):
            layer.quantize(mode, config=_config(mode))
        assert_float(self, layer)


# Dense, a 2-D einsum kernel, the keras-hub Gemma query projection (whose
# contraction view permutes the kernel), a mixture-of-experts down
# projection (whose expert axis is a batch of calibration problems), and a
# kernel whose last axis is such a batch axis.
KINDS = ["dense", "einsum", "permuted", "batched", "batch_last"]


def _make_layer(kind, rng, **kwargs):
    if kind == "dense":
        layer = layers.Dense(16, **kwargs)
        input_shape = (None, 12)
        x = rng.standard_normal((32, 12))
    elif kind == "einsum":
        layer = layers.EinsumDense(
            "abc,cd->abd", output_shape=(None, 16), bias_axes="d", **kwargs
        )
        input_shape = (None, 3, 12)
        x = rng.standard_normal((8, 3, 12))
    elif kind == "permuted":
        layer = layers.EinsumDense(
            "btd,ndh->btnh", output_shape=(None, 4, 6), bias_axes="nh", **kwargs
        )
        input_shape = (None, 3, 12)
        x = rng.standard_normal((8, 3, 12))
    elif kind == "batched":
        layer = layers.EinsumDense(
            "bec,ecd->bed", output_shape=(4, 6), bias_axes="d", **kwargs
        )
        input_shape = (None, 4, 12)
        x = rng.standard_normal((8, 4, 12))
    else:
        layer = layers.EinsumDense(
            "bce,dce->bde", output_shape=(6, 4), bias_axes="d", **kwargs
        )
        input_shape = (None, 12, 4)
        x = rng.standard_normal((8, 12, 4))
    layer.build(input_shape)
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    return layer, x.astype("float32")


def _set_adapter(layer, rng, scale):
    """Random LoRA factors whose product is about `scale` of the kernel."""
    a = rng.standard_normal(layer.lora_kernel_a.shape).astype("float32")
    b = rng.standard_normal(layer.lora_kernel_b.shape).astype("float32")
    layer.lora_kernel_a.assign(a)
    layer.lora_kernel_b.assign(b * scale)


def _set_adapter_by_norm(layer, rng, fraction):
    """Random LoRA factors whose update has `fraction` of the weight's norm."""
    _set_adapter(layer, rng, scale=1.0)
    scale = fraction * np.linalg.norm(_dequantized(layer))
    scale /= np.linalg.norm(_delta(layer))
    b = ops.convert_to_numpy(layer.lora_kernel_b)
    layer.lora_kernel_b.assign(b * scale)


def _delta(layer):
    a = ops.convert_to_numpy(layer.lora_kernel_a)
    b = ops.convert_to_numpy(layer.lora_kernel_b)
    return (layer.lora_alpha / layer.lora_rank) * (a @ b)


def _save(layer):
    store = {}
    layer.save_own_variables(store)
    return store


def _stored(layer, store):
    """The saved store keyed by serialization-spec name."""
    spec = layer.variable_serialization_spec[layer.quantization_mode]
    return {name: store[str(i)] for i, name in enumerate(spec)}


def _save_with_warnings(layer):
    """The saved store by spec name, and the merge warnings of the save."""
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        store = _stored(layer, _save(layer))
    messages = [
        str(w.message) for w in warned if "Merging the LoRA" in str(w.message)
    ]
    return store, messages


def _policy_built(layer, config):
    """A fresh layer of `layer`'s kind, built under `config`'s policy."""
    kwargs = layer.get_config()
    for key in (
        "name",
        "dtype",
        "quantization_config",
        "lora_rank",
        "lora_alpha",
    ):
        kwargs.pop(key, None)
    kwargs["dtype"] = f"{config.dtype_policy_string()}_from_float32"
    new = type(layer).from_config(kwargs)
    new.build(layer._build_shapes_dict["input_shape"])
    return new


class CalibrationLoRATest(testing.TestCase):
    """LoRA on the calibration modes: forward, calibration, merged save."""

    @parameterized.named_parameters(named_product(mode=MODES, kind=KINDS))
    def test_lora_delta_matches_float(self, mode, kind):
        # The LoRA update rides on the dequantized contraction, so its
        # contribution equals the float layer's for every kernel layout.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        float_layer, _ = _make_layer(kind, rng)
        float_layer.kernel.assign(layer.kernel)
        _calibrate(layer, mode, x, group_size=4)

        def lora_delta(layer):
            before = ops.convert_to_numpy(layer(x))
            layer.enable_lora(2)
            _set_adapter(layer, np.random.default_rng(1), scale=0.05)
            return ops.convert_to_numpy(layer(x)) - before

        self.assertAllClose(
            lora_delta(layer),
            lora_delta(float_layer),
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
        # `kernel` keeps the kernel's shape, as for int8 and int4.
        self.assertEqual(
            tuple(layer.kernel.shape), tuple(float_layer.kernel.shape)
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    @pytest.mark.requires_trainable_backend
    def test_lora_fit_updates_factors(self, mode):
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        a_before = ops.convert_to_numpy(layer.lora_kernel_a).copy()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, rng.standard_normal((32, 16)), epochs=2, verbose=0)
        self.assertGreater(
            np.abs(ops.convert_to_numpy(layer.lora_kernel_a) - a_before).max(),
            0.0,
        )
        self.assertGreater(
            np.abs(ops.convert_to_numpy(layer.lora_kernel_b)).max(), 0.0
        )

    @parameterized.named_parameters(
        ("gptq_2bit", "gptq", dict(weight_bits=2, group_size=8), "float32"),
        ("gptq_3bit", "gptq", dict(weight_bits=3, group_size=4), "float32"),
        (
            "gptq_4bit_act_order",
            "gptq",
            dict(weight_bits=4, group_size=4, activation_order=True),
            "float32",
        ),
        (
            "gptq_8bit_symmetric",
            "gptq",
            dict(weight_bits=8, group_size=-1, symmetric=True),
            "float32",
        ),
        (
            "gptq_8bit_mixed_bfloat16",
            "gptq",
            dict(weight_bits=8, group_size=8),
            "mixed_bfloat16",
        ),
        ("awq", "awq", dict(group_size=4), "float32"),
        ("awq_mixed_bfloat16", "awq", dict(group_size=4), "mixed_bfloat16"),
    )
    def test_lora_merge_with_zero_delta_is_identity(self, mode, kwargs, dtype):
        # A merged save rounds the merged weight onto the calibrated grid,
        # so with nothing to merge every stored variable comes back as it
        # is, whatever the compute dtype, and the save does not warn.
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng, dtype=dtype)
        _calibrate(layer, mode, x, **kwargs)
        layer.enable_lora(2)
        store, messages = _save_with_warnings(layer)
        self.assertEqual(messages, [])
        for name, value in store.items():
            self.assertAllEqual(value, getattr(layer, name), msg=name)

    @parameterized.named_parameters(
        named_product(mode=MODES, dtype=["float32", "mixed_bfloat16"])
    )
    def test_lora_merge_keeps_calibrated_parameters(self, mode, dtype):
        # The scale, zero point, group index (here an activation-order
        # permutation) and AWQ scales are the calibration's result; a
        # merged save changes only the codes, whatever the compute dtype.
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng, dtype=dtype)
        kwargs = dict(group_size=8)
        if mode == "gptq":
            kwargs["activation_order"] = True
        _calibrate(layer, mode, x, **kwargs)
        g_idx = ops.convert_to_numpy(layer.g_idx)
        if mode == "gptq":
            self.assertFalse(np.all(np.diff(g_idx) >= 0))
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        store = _stored(layer, _save(layer))
        for name in ("kernel_scale", "kernel_zero", "g_idx", "awq_scales"):
            if name in store:
                self.assertAllEqual(store[name], getattr(layer, name), msg=name)
        self.assertFalse(
            np.array_equal(
                ops.convert_to_numpy(store["quantized_kernel"]),
                ops.convert_to_numpy(layer.quantized_kernel),
            )
        )

    @parameterized.named_parameters(
        named_product(
            mode=MODES, kind=["dense", "permuted", "batched", "batch_last"]
        )
    )
    def test_lora_merged_weight_is_within_half_a_step(self, mode, kind):
        # Round-to-nearest under the calibrated parameters: every merged
        # weight the grid covers lands within half a code of the sum of
        # the dequantized weight and the delta, and a layer built under
        # the mode's policy reads the merged store back to that weight.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        config = _calibrate(layer, mode, x, group_size=4).config
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        quantized_weight = layer._quantized_weight()
        merged = ops.convert_to_numpy(
            quantized_weight.dequantize("float32")
        ) + _delta(layer)

        new = _policy_built(layer, config)
        new.load_own_variables(_save(layer))
        reloaded = ops.convert_to_numpy(
            new._quantized_weight().dequantize("float32")
        )

        image = ops.convert_to_numpy(quantized_weight.code_image(merged))
        low, high = quantized_weight.scheme.code_range
        covered = (image >= low - 0.5) & (image <= high + 0.5)
        # One real-valued unit per position, in the stored layout.
        step = ops.convert_to_numpy(
            quantized_weight.code_image(
                np.ones(quantized_weight.shape, "float32")
            )
        ) - ops.convert_to_numpy(
            quantized_weight.code_image(
                np.zeros(quantized_weight.shape, "float32")
            )
        )
        error = np.abs(
            ops.convert_to_numpy(quantized_weight._as_stored(reloaded - merged))
        )
        bound = 0.5 / np.abs(step) + 1e-6
        self.assertGreater(covered.mean(), 0.9)
        self.assertTrue(np.all(error[covered] <= bound[covered]))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_lora_merge_clips_outside_the_calibrated_range(self, mode):
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        layer.lora_kernel_a.assign(np.ones(layer.lora_kernel_a.shape))
        layer.lora_kernel_b.assign(np.full(layer.lora_kernel_b.shape, 5.0))
        store, messages = _save_with_warnings(layer)
        self.assertLen(messages, 1)
        self.assertIn("clipped", messages[0])
        self.assertIn(f"layer '{layer.name}'", messages[0])
        self.assertIn(mode.upper(), messages[0])
        quantized_weight = layer._quantized_weight()
        codes = quantized_weight.layout.unpack(store["quantized_kernel"])
        high = quantized_weight.scheme.code_range[1]
        self.assertAllEqual(codes, np.full(layer.kernel_shape, high, "uint8"))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_lora_merge_warns_when_the_update_is_rounded_away(self, mode):
        # An update at 1% of the weight's norm is smaller than half a code
        # step of every group: the merged save keeps the base codes and
        # warns, though it clips nothing.
        rng = np.random.default_rng(0)
        layer, x = _make_layer("dense", rng)
        _calibrate(layer, mode, x, group_size=8)
        layer.enable_lora(2)
        _set_adapter_by_norm(layer, rng, fraction=0.01)
        store, messages = _save_with_warnings(layer)
        self.assertLen(messages, 1)
        self.assertIn("rounded away", messages[0])
        self.assertIn(f"layer '{layer.name}'", messages[0])
        self.assertIn(mode.upper(), messages[0])
        self.assertAllEqual(store["quantized_kernel"], layer.quantized_kernel)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_lora_merge_that_keeps_the_update_does_not_warn(self, mode):
        # At 20% of the weight's norm, the update moves most of the codes
        # of a per-channel grid over 256 rows and pushes few past it.
        rng = np.random.default_rng(0)
        layer = layers.Dense(64)
        layer.build((None, 256))
        layer.kernel.assign(rng.standard_normal((256, 64)) * 0.1)
        x = rng.standard_normal((64, 256)).astype("float32")
        _calibrate(layer, mode, x, group_size=-1)
        layer.enable_lora(2)
        _set_adapter_by_norm(layer, rng, fraction=0.2)
        store, messages = _save_with_warnings(layer)
        self.assertEqual(messages, [])
        self.assertFalse(
            np.array_equal(
                ops.convert_to_numpy(store["quantized_kernel"]),
                ops.convert_to_numpy(layer.quantized_kernel),
            )
        )

    @parameterized.named_parameters(named_product(mode=MODES, kind=KINDS))
    def test_enable_lora_before_calibration(self, mode, kind):
        # The calibrators quantize the base kernel of the float layer, so
        # the update stays a separate term, and the swap keeps the factors.
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        twin, _ = _make_layer(kind, rng)
        twin.kernel.assign(layer.kernel)
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.05)
        factors = [
            ops.convert_to_numpy(layer.lora_kernel_a),
            ops.convert_to_numpy(layer.lora_kernel_b),
        ]

        for target in (layer, twin):
            _calibrate(target, mode, x, group_size=4)
        self.assertFalse(hasattr(layer, "_kernel"))
        self.assertTrue(layer.lora_enabled)
        self.assertAllEqual(layer.lora_kernel_a, factors[0])
        self.assertAllEqual(layer.lora_kernel_b, factors[1])
        self.assertAllEqual(layer.quantized_kernel, twin.quantized_kernel)
        twin.enable_lora(2)
        twin.lora_kernel_a.assign(layer.lora_kernel_a)
        twin.lora_kernel_b.assign(layer.lora_kernel_b)
        self.assertAllClose(
            layer(x), twin(x), atol=1e-6, tpu_atol=1e-2, tpu_rtol=1e-2
        )

    @parameterized.named_parameters(
        named_product(mode=MODES, kind=["dense", "permuted"])
    )
    def test_lora_model_round_trip(self, mode, kind):
        rng = np.random.default_rng(0)
        layer, x = _make_layer(kind, rng)
        config = _calibrate(layer, mode, x, group_size=8).config
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.002)
        model = models.Sequential([layer])
        model(x)
        # What every reload must reproduce: the merged codes.
        merged = _policy_built(layer, config)
        merged.load_own_variables(_save(layer))
        expected = ops.convert_to_numpy(merged(x))
        self.assertNotAllClose(expected, model(x))

        # A `.keras` file re-creates LoRA from the config with zero factors.
        path = os.path.join(self.get_temp_dir(), "lora.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        self.assertTrue(reloaded.layers[0].lora_enabled)
        self.assertAllEqual(
            reloaded.layers[0].lora_kernel_b,
            np.zeros(layer.lora_kernel_b.shape, "float32"),
        )
        self.assertAllClose(reloaded(x), expected, atol=1e-6)

        # Weights only, into a model built under the mode's policy.
        path = os.path.join(self.get_temp_dir(), "lora.weights.h5")
        model.save_weights(path)
        fresh = models.Sequential([_policy_built(layer, config)])
        fresh.build((None,) + x.shape[1:])
        fresh.load_weights(path)
        self.assertFalse(fresh.layers[0].lora_enabled)
        self.assertAllClose(fresh(x), expected, atol=1e-6)

        # A plain checkpoint back into the LoRA model zeroes its factors.
        fresh.save_weights(path)
        model.load_weights(path)
        self.assertAllClose(model(x), expected, atol=1e-6)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_policy_built_layer_with_lora_rank(self, mode):
        # A layer built from a saved config carries `lora_rank`; it has
        # codes and no float kernel, so `enable_lora` has nothing to freeze.
        layer = layers.Dense(16, dtype=f"{mode}/4/8_from_float32", lora_rank=2)
        layer.build((None, 12))
        self.assertTrue(layer.lora_enabled)
        self.assertFalse(hasattr(layer, "_kernel"))
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 4 if mode == "gptq" else 5)


# name -> (equation, or `None` for `Dense`; output shape or units; inputs).
# Each equation is reshape-compatible, so a forced route serves it. A
# `_lora` case also has a bias, an activation and LoRA.
ROUTED = {
    "dense": (None, 24, (3, 128)),
    "dense_lora": (None, 24, (2, 3, 128)),
    "qkv_dnh_lora": ("btd,dnh->btnh", (None, 2, 8), (2, 3, 128)),
    "query_ndh": ("btd,ndh->btnh", (None, 2, 8), (2, 3, 128)),
    "output_nhd": ("btnh,nhd->btd", (None, 24), (2, 3, 2, 64)),
}
# name -> (mode, group size, activation order).
ROUTED_MODES = {
    "gptq": ("gptq", 64, False),
    "gptq_per_channel": ("gptq", -1, False),
    "gptq_act_order": ("gptq", 64, True),
    "awq": ("awq", 64, False),
}


def _routed_layer(case, mode, dtype="float32"):
    """A layer of `case` calibrated in `mode`, and inputs for it."""
    equation, output, input_shape = ROUTED[case]
    extras = case.endswith("_lora")
    activation = "relu" if extras else None
    if equation is None:
        layer = layers.Dense(
            output, activation=activation, use_bias=extras, dtype=dtype
        )
    else:
        layer = layers.EinsumDense(
            equation,
            output,
            activation=activation,
            bias_axes=equation[-1] if extras else None,
            dtype=dtype,
        )
    layer.build(input_shape)
    rng = np.random.default_rng(0)
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    if extras:
        layer.bias.assign(rng.standard_normal(layer.bias.shape))
    mode, group_size, activation_order = ROUTED_MODES[mode]
    kwargs = {"activation_order": True} if activation_order else {}
    x = rng.standard_normal((16,) + input_shape[1:]).astype("float32")
    _calibrate(layer, mode, x, group_size=group_size, **kwargs)
    if extras:
        layer.enable_lora(2)
        _set_adapter(layer, rng, scale=0.1)
    return layer, x[: input_shape[0]]


def _vjp(layer, x, upstream, compiled):
    """The outputs of `layer` at `x`, and the input gradient for the output
    gradient `upstream`, eagerly or compiled."""
    if backend.backend() == "jax":

        def vjp(v, u):
            y, pullback = jax.vjp(layer, v)
            return y, pullback(ops.cast(u, y.dtype))[0]

        y, grad = jax.jit(vjp)(x, upstream) if compiled else vjp(x, upstream)
        return _numpy(y), _numpy(grad)
    if backend.backend() == "torch":
        call = layer
        if compiled:
            torch._dynamo.reset()
            call = torch.compile(lambda v: layer(v), backend="aot_eager")
        v = torch.tensor(x, requires_grad=True)
        y = call(v)
        y.backward(ops.cast(upstream, y.dtype))
        return _numpy(y.detach()), _numpy(v.grad)

    def vjp(v, u):
        with tf.GradientTape() as tape:
            tape.watch(v)
            y = layer(v)
        return y, tape.gradient(y, v, output_gradients=ops.cast(u, y.dtype))

    if compiled:
        vjp = tf.function(vjp, jit_compile=True)
    y, grad = vjp(tf.constant(x), tf.constant(upstream))
    return _numpy(y), _numpy(grad)


class CalibrationRouteTest(testing.TestCase):
    """GPTQ and AWQ through a route.

    `force_route("reference_wrapped")` runs the reference path as a route,
    inside the custom gradient, and `matrix` (a route of `kernels_test.py`)
    contracts the `(M, K)` inputs with the stored matrix. `kernels_test.py`
    runs N5 on GPTQ and AWQ.
    """

    @parameterized.named_parameters(
        named_product(
            case=sorted(ROUTED), mode=sorted(ROUTED_MODES), dtype=["float32"]
        )
        + named_product(
            case=sorted(ROUTED), mode=MODES, dtype=["mixed_bfloat16"]
        )
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="NumPy has no gradients."
    )
    def test_wrapped_reference(self, case, mode, dtype):
        """The outputs equal the reference path's bit for bit. The input
        gradient is the backward of the custom gradient: for `Dense` it
        equals autodiff's bit for bit, and for an einsum it can sum in
        another order, within 8 ulps of the largest entry."""
        layer, x = _routed_layer(case, mode, dtype)
        rng = np.random.default_rng(1)
        upstream = rng.standard_normal(_numpy(layer(x)).shape)
        upstream = upstream.astype("float32")
        mantissa = ml_dtypes.finfo(layer.compute_dtype).nmant
        for compiled in (False, True):
            with kernels.policy("off"):
                y, expected = _vjp(layer, x, upstream, compiled)
            kernels.route_counts.clear()
            with kernels.force_route("reference_wrapped"):
                outputs, gradient = _vjp(layer, x, upstream, compiled)
            if not compiled:
                counts = dict(kernels.route_counts)
                self.assertEqual(counts, {"reference_wrapped": 1})
            self.assertAllEqual(outputs, y)
            if ROUTED[case][0] is None:
                self.assertAllEqual(gradient, expected)
            else:
                peak = np.max(np.abs(expected))
                ulp = 2.0 ** (np.floor(np.log2(peak)) - mantissa)
                self.assertLessEqual(
                    np.max(np.abs(gradient - expected)), 8 * ulp
                )
            # A route replaces the forward only.
            with _routes(), kernels.force_route("matrix"):
                _, routed = _vjp(layer, x, upstream, compiled)
            self.assertAllEqual(routed, gradient)

    @parameterized.named_parameters(
        named_product(case=["dense_lora", "query_ndh"], mode=MODES)
    )
    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Reads the residuals of `jax.vjp`."
    )
    def test_jax_residuals_hold_the_stored_tensors(self, case, mode):
        """Autodiff keeps the float kernel for the backward pass. The custom
        gradient keeps the stored tensors and fewer bytes."""
        layer, x = _routed_layer(case, mode)
        kernel_size = math.prod(layer.kernel.shape)

        def residuals(scope):
            with scope:
                _, pullback = jax.vjp(layer, x)
            leaves = jax.tree_util.tree_leaves(pullback)
            return [leaf for leaf in leaves if hasattr(leaf, "nbytes")]

        def float_kernels(leaves):
            return [
                leaf
                for leaf in leaves
                if leaf.size == kernel_size
                and jax.numpy.issubdtype(leaf.dtype, jax.numpy.floating)
            ]

        expected = residuals(kernels.policy("off"))
        wrapped = residuals(kernels.force_route("reference_wrapped"))
        self.assertNotEmpty(float_kernels(expected))
        # The default policy has no route here, so autodiff keeps the
        # float kernel too.
        self.assertNotEmpty(float_kernels(residuals(kernels.policy("auto"))))
        self.assertEmpty(float_kernels(wrapped))
        codes = layer.quantized_kernel
        self.assertTrue(
            any(
                leaf.shape == codes.shape and leaf.dtype == codes.dtype
                for leaf in wrapped
            )
        )
        self.assertLess(
            sum(leaf.nbytes for leaf in wrapped),
            sum(leaf.nbytes for leaf in expected),
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_reference_path_constructs_no_custom_gradient(self, mode):
        """NumPy warns when it constructs a custom gradient. The reference
        path (`"auto"` with no row, and `"off"`) constructs none, and the
        wrapped reference does."""
        layer, x = _routed_layer("dense", mode)
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        for scope, warns in (
            (kernels.policy("auto"), False),
            (kernels.policy("off"), False),
            (
                kernels.force_route("reference_wrapped"),
                backend.backend() == "numpy",
            ),
        ):
            with (
                warnings.catch_warnings(record=True) as caught,
                _routes(),
                scope,
            ):
                warnings.simplefilter("always")
                y = layer(x)
            self.assertAllEqual(_numpy(y), expected)
            messages = [str(w.message) for w in caught]
            self.assertEqual(
                any("`custom_gradient`" in m for m in messages), warns
            )

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_calibration_capture_runs_once_per_call(self, mode):
        layer, x = _routed_layer("dense", mode)
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

    @parameterized.named_parameters(
        ("two_bit", "dense", 2),
        ("three_bit", "dense", 3),
        ("eight_bit", "dense", 8),
        ("batch_axis", "batched", 4),
    )
    def test_refusals_run_the_reference_path(self, kind, bits):
        """Codes that do not pack two per byte and a batch axis keep the
        reference path under a forced route."""
        layer, x = _make_layer(kind, np.random.default_rng(0))
        _calibrate(layer, "gptq", x, weight_bits=bits, group_size=4)
        with kernels.policy("off"):
            expected = _numpy(layer(x))
        kernels.route_counts.clear()
        with kernels.force_route("reference_wrapped"):
            self.assertAllEqual(_numpy(layer(x)), expected)
        self.assertEqual(dict(kernels.route_counts), {"reference": 1})
