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

    The constructor resolves the 2-D `(rows, columns)` view of the kernel
    from the layer's geometry: `rows` is the number of input features.
    `num_samples` counts the input rows observed so far. Subclasses
    implement `observe` and `_solve`, and a subclass that sets
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
        self.rows, self.columns = geometry.calibration_rows_columns(
            tuple(layer.kernel.shape)
        )

    def observe(self, inputs):
        """Accumulates statistics from one batch of the layer's inputs."""
        raise NotImplementedError

    def _flatten_inputs(self, inputs):
        """Validates `inputs` and lays them out as `float32` `[-1, rows]`."""
        if inputs is None:
            raise ValueError("Input tensor cannot be None.")
        if len(inputs.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(inputs.shape)})."
            )
        if ops.size(inputs) == 0:
            raise ValueError("Input tensor cannot be empty.")
        if len(inputs.shape) > 2:
            inputs = ops.reshape(inputs, (-1, inputs.shape[-1]))
        return ops.cast(inputs, "float32")

    def quantize(self):
        """Solves for the layer's codes and writes them back."""
        # The solve runs in float32 whatever the layer's variable dtype.
        kernel = ops.cast(self.layer.kernel, "float32")
        if len(kernel.shape) != 2:
            kernel = ops.reshape(kernel, (self.rows, self.columns))
        codes, scale, zero, g_idx, extra = self._solve(ops.transpose(kernel))
        # The solvers work on `[out, in]`; the layer stores the kernel's own
        # `[in, out]` orientation with the group parameters as
        # `[n_groups, out]`, so the forward pass never transposes.
        self.strategy.write_back(
            self.layer,
            ops.transpose(codes),
            ops.transpose(scale),
            ops.transpose(zero),
            g_idx,
            **extra,
        )

    def _solve(self, weights):
        """Quantizes `weights`, the `[out, in]` kernel, from the statistics.

        Returns:
            `(codes, scale, zero, g_idx, extra)`: the codes `[out, in]`, the
            scale and zero point `[out, n_groups]`, the group index `[in]`,
            and a dict of the mode's extra calibrated values, which
            `write_back` receives as keyword arguments.
        """
        raise NotImplementedError
