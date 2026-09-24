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

    The constructor resolves the layer's `CalibrationView`: the 2-D
    `(rows, columns)` matrix the algorithm works on and `batch`, the
    number of independent problems a kernel axis shared with the inputs
    stacks. `rows` is the number of contracted features a sample of the
    layer's inputs carries; `num_samples` counts the input rows observed
    so far, per problem.

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
        self.view = geometry.calibration_view()
        self.batch = self.view.batch
        self.rows = self.view.rows
        self.columns = self.view.columns

    def observe(self, inputs):
        """Accumulates statistics from one batch of the layer's inputs."""
        raise NotImplementedError

    def _check_inputs(self, inputs):
        """Validates one batch of the layer's inputs before `observe`."""
        if inputs is None:
            raise ValueError("Input tensor cannot be None.")
        if len(inputs.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(inputs.shape)})."
            )
        if ops.size(inputs) == 0:
            raise ValueError("Input tensor cannot be empty.")

    def _kernel_view(self):
        """The layer's kernel laid out through the calibration view."""
        return self.view.kernel_to_view(self.original_layer.kernel)

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
