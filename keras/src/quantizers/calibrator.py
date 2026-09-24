"""The per-layer object of a calibration mode.

A `CalibrationRun` creates one `Calibrator` per layer of a block, feeds it
every input the layer sees during the calibration sweeps through
`observe`, then calls `quantize` to solve for the layer's codes and write
them back through the mode's strategy, and `release` to drop the
statistics. `GPTQCalibrator` (a Hessian) and `AWQCalibrator` (activation
magnitudes) are the two calibrators of the built-in modes.
"""

from keras.src import ops
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.geometry import ProjectionGeometry


class Calibrator:
    """Per-layer calibration state and solve for one calibration mode.

    The constructor resolves the 2-D `(rows, columns)` view the algorithm
    works on from the layer's geometry (`calibration_rows_columns`).
    `rows` is the number of input features a sample of the layer's
    inputs carries; `num_samples` counts the input rows observed so far.

    Args:
        layer: A layer with a projection geometry (`Dense`, `EinsumDense`)
            that supports the calibrator's mode.
        config: The mode's config object.
    """

    # The quantization mode the calibrator solves for; it names the
    # strategy the constructor validates against and the write-back.
    mode = None
    # Warn after the run when a layer saw fewer than this many input rows
    # per input feature; `None` never warns. A calibrator that sets it
    # also implements `undersampling_warning`.
    warn_tokens_per_row = None

    def __init__(self, layer, config):
        self.original_layer = layer
        self.config = config
        self.num_samples = 0
        strategy = strategy_registry.get_strategy(self.mode)
        geometry = layer._quantization_geometry()
        if not isinstance(
            geometry, ProjectionGeometry
        ) or not layer._supports_quantization_mode(strategy):
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

    def _kernel_view(self):
        """The layer's kernel in the `(rows, columns)` view.

        The variable itself for a 2-D kernel, a reshaped copy for a 3-D
        one.
        """
        kernel = self.original_layer.kernel
        if len(kernel.shape) != 2:
            kernel = ops.reshape(kernel, (self.rows, self.columns))
        return kernel

    def quantize(self):
        """Solves for the layer's codes and writes them back."""
        raise NotImplementedError

    def release(self):
        """Drops the statistics."""
        raise NotImplementedError

    @classmethod
    def undersampling_warning(cls, layers):
        """The warning for layers calibrated on too few input rows.

        Args:
            layers: List of `(name, tokens, rows)` for every layer that saw
                fewer than `warn_tokens_per_row` input rows per input
                feature, as tallied by `CalibrationRun`.
        """
        raise NotImplementedError
