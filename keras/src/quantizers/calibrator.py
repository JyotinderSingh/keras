"""The per-layer object of a calibration mode.

A `CalibrationRun` creates one `Calibrator` per layer of a block, passes
every input the layer sees during the calibration sweeps to `observe`,
then calls `quantize`, which solves for the layer's codes and writes them
back through the mode's strategy. `GPTQCalibrator` (a Hessian) and
`AWQCalibrator` (activation magnitudes) are the calibrators of the
built-in modes.
"""

from keras.src import ops
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.geometry import ProjectionGeometry


class Calibrator:
    """Per-layer statistics and solve of one calibration mode.

    The constructor resolves the layer's `ContractionView`. `rows` is the
    number of contracted features of one input sample, `columns` the
    number of outputs, and `batch` the number of independent problems
    that a kernel axis shared with the inputs splits the kernel into. A
    statistic has a leading problem axis only when `batch > 1`.
    `num_samples` counts the input rows observed so far, per problem.
    Subclasses implement `observe` and `_solve`, and a subclass that sets
    `warn_tokens_per_row` also words the warning in its classmethod
    `undersampling_warning(layers)`.

    Args:
        layer: A layer with a projection geometry (`Dense`, `EinsumDense`)
            that supports the calibrator's mode.
        config: The mode's config object.
    """

    # The quantization mode the calibrator solves for.
    mode = None
    # Warn after the run when a layer saw fewer input rows than this per
    # input feature; `None` never warns.
    warn_tokens_per_row = None

    def __init__(self, layer, config):
        self.layer = layer
        self.config = config
        self.num_samples = 0
        self.strategy = strategy_registry.get_strategy(self.mode)
        geometry = layer._quantization_geometry()
        if not isinstance(
            geometry, ProjectionGeometry
        ) or not layer._supports_quantization_mode(self.strategy):
            raise TypeError(
                f"Unsupported layer type for {self.mode.upper()}: {type(layer)}"
            )
        self.view = geometry.contraction_view()
        self.batch = self.view.batch
        self.rows = self.view.rows
        self.columns = self.view.columns

    def observe(self, inputs):
        """Accumulates statistics from one batch of the layer's inputs."""
        raise NotImplementedError

    def _inputs_view(self, inputs):
        """Validates `inputs` and lays them out through the view.

        Returns `float32` `(samples, rows)`, or `(batch, samples, rows)`
        when the view has several problems.
        """
        if inputs is None:
            raise ValueError("Input tensor cannot be None.")
        if len(inputs.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(inputs.shape)})."
            )
        if ops.size(inputs) == 0:
            raise ValueError("Input tensor cannot be empty.")
        input_features = self.view.input_features(inputs)
        if input_features != self.rows:
            raise ValueError(
                f"Calibration statistics ({self.rows}) do not match input "
                f"features ({input_features})."
            )
        return ops.cast(self.view.inputs_to_view(inputs), "float32")

    def _per_problem(self, shape):
        """`shape` with a leading problem axis when there are several."""
        return shape if self.batch == 1 else (self.batch,) + shape

    def _problem(self, statistic, index):
        """Problem `index`'s slice of a per-problem statistic."""
        return statistic if self.batch == 1 else statistic[index]

    def quantize(self):
        """Solves each problem for its codes and writes them back."""
        # The base kernel, in float32 whatever the layer's variable dtype. A
        # LoRA update stays a separate term of the forward pass, as for int8
        # and int4.
        kernel = ops.cast(self.layer._kernel, "float32")
        kernel = self.view.kernel_to_view(kernel)
        results = [
            self._solve(ops.transpose(kernel[index]), index)
            for index in range(self.batch)
        ]
        codes, scale, zero, g_idx, extra = self._stack_problems(results)
        self.strategy.write_back(self.layer, codes, scale, zero, g_idx, **extra)

    def _solve(self, weights, index):
        """Quantizes `weights`, problem `index`'s `[out, in]` kernel.

        Returns:
            `(codes, scale, zero, g_idx, extra)`: the codes `[out, in]`, the
            scale and zero point `[out, n_groups]`, the group index `[in]`,
            and a dict of the mode's extra calibrated values, one per input
            row, which `write_back` receives as keyword arguments.
        """
        raise NotImplementedError

    @staticmethod
    def _stack_problems(results):
        """Stacks the problems' `_solve` results along the stored rows.

        The solvers work on `[out, in]`; the layer stores the view's
        `[in, out]` orientation with the group parameters as
        `[n_groups, out]`, so the forward pass never transposes. Each
        problem's groups are numbered after the previous problem's.
        """
        codes, scales, zeros, group_indices = [], [], [], []
        extras = {}
        for index, (code, scale, zero, g_idx, extra) in enumerate(results):
            n_groups = ops.shape(scale)[1]
            codes.append(ops.transpose(code))
            scales.append(ops.transpose(scale))
            zeros.append(ops.transpose(zero))
            group_indices.append(ops.add(g_idx, index * n_groups))
            for name, value in extra.items():
                extras.setdefault(name, []).append(value)
        return (
            ops.concatenate(codes, axis=0),
            ops.concatenate(scales, axis=0),
            ops.concatenate(zeros, axis=0),
            ops.concatenate(group_indices, axis=0),
            {name: ops.concatenate(v, axis=0) for name, v in extras.items()},
        )
