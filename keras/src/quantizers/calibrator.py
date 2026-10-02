"""The per-layer object of a calibration mode.

A `CalibrationRun` creates one `Calibrator` per float layer of a block,
passes every input the layer sees during the calibration sweeps to
`observe`, then calls `quantize`, which solves for the layer's codes and
swaps them in through the mode's strategy. Every calibrator accumulates
the Hessian of the layer's inputs. `GPTQCalibrator` solves with it;
`AWQCalibrator` also accumulates activation magnitudes and scores its
searches with the Hessian.
"""

from keras.src import ops
from keras.src.quantizers.geometry import ProjectionGeometry


def accumulate_hessian(hessian, x, num_samples):
    """Adds a batch of inputs to a running Hessian `2 X^T X / N`.

    GPTQ solves with this Hessian and AWQ scores its searches with it.

    Args:
        hessian: The Hessian of the `num_samples` samples seen so far, with
            a leading problem axis when `x` has one.
        x: The batch laid out by the contraction view, `(samples, rows)`
            or `(batch, samples, rows)`.
        num_samples: The number of samples `hessian` covers.

    Returns:
        The Hessian of the `num_samples` samples and the samples of `x`.
    """
    total_samples = num_samples + int(ops.shape(x)[-2])
    gram_matrix = ops.matmul(ops.swapaxes(x, -1, -2), x)
    # Ensures numerical stability and symmetry in case of large floating
    # point activations.
    gram_matrix = ops.divide(
        ops.add(gram_matrix, ops.swapaxes(gram_matrix, -1, -2)), 2.0
    )
    # Decay the previous mean and add the batch's contribution (2 / N).
    if num_samples > 0:
        hessian = ops.multiply(hessian, ops.divide(num_samples, total_samples))
    return ops.add(
        hessian, ops.multiply(ops.divide(2.0, total_samples), gram_matrix)
    )


class Calibrator:
    """Per-layer statistics and solve of one calibration mode.

    The constructor resolves the layer's `ContractionView`. `rows` is the
    number of contracted features of one input sample, and `batch` the
    number of independent problems that a kernel axis shared with the
    inputs splits the kernel into. A statistic has a leading problem axis
    only when `batch > 1`. `num_samples` counts the input samples observed
    so far, per problem, and `hessian` is their `2 mean(x x^T)`.
    Subclasses implement `_solve` and can accumulate statistics of their
    own in `_observe`. A subclass that sets `warn_tokens_per_row` also
    words the warning in its classmethod `undersampling_warning(layers)`.

    Args:
        strategy: The `CalibrationStrategy` of the mode the calibrator
            solves for.
        layer: A float layer with a projection geometry (`Dense`,
            `EinsumDense`) that supports the mode.
        config: The mode's config object. The solve reads its parameters,
            and the swap builds the layer's variables from it.
    """

    # Warn after the run when a layer saw fewer calibration tokens than
    # this per kernel row (input feature); `None` never warns.
    warn_tokens_per_row = None

    def __init__(self, strategy, layer, config):
        self.strategy = strategy
        self.layer = layer
        self.config = config
        self.num_samples = 0
        geometry = layer._quantization_geometry()
        if not isinstance(
            geometry, ProjectionGeometry
        ) or not layer._supports_quantization_mode(strategy):
            raise TypeError(
                f"Unsupported layer type for {strategy.name.upper()}: "
                f"{type(layer)}"
            )
        self.view = geometry.contraction_view()
        self.batch = self.view.batch
        self.rows = self.view.rows
        self.hessian = ops.zeros(
            self._per_problem((self.rows, self.rows)), dtype="float32"
        )

    def observe(self, inputs):
        """Accumulates statistics from one batch of the layer's inputs."""
        x = self._inputs_view(inputs)
        self._observe(x)
        self.hessian = accumulate_hessian(self.hessian, x, self.num_samples)
        self.num_samples += int(ops.shape(x)[-2])

    def _observe(self, x):
        """Accumulates the mode's own statistics from a laid-out batch.

        `x` is the batch as `_inputs_view` returns it; `num_samples` does
        not count it yet.
        """

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
        """Solves each problem for its codes and swaps them in."""
        # The base kernel, in float32 whatever the layer's variable dtype. A
        # LoRA update stays a separate term of the forward pass, as for int8
        # and int4.
        kernel = ops.cast(self.layer._kernel, "float32")
        kernel = self.view.kernel_to_view(kernel)
        results = [
            self._solve(ops.transpose(kernel[index]), index)
            for index in range(self.batch)
        ]
        self.strategy.write_back(
            self.layer, self.config, *self._stack_problems(results)
        )

    def _solve(self, weights, index):
        """Quantizes `weights`, problem `index`'s `[out, in]` kernel.

        Returns:
            `(codes, scale, zero, g_idx, input_scales)`: the codes
            `[out, in]`, the scale and zero point `[out, n_groups]`, the
            group index `[in]`, and the per-input-row scales `[in]` that
            multiplied the weights before quantization, or `None`.
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
        codes, scales, zeros, group_indices, input_scales = [], [], [], [], []
        for index, (code, scale, zero, g_idx, in_scales) in enumerate(results):
            n_groups = ops.shape(scale)[1]
            codes.append(ops.transpose(code))
            scales.append(ops.transpose(scale))
            zeros.append(ops.transpose(zero))
            group_indices.append(ops.add(g_idx, index * n_groups))
            if in_scales is not None:
                input_scales.append(in_scales)
        return (
            ops.concatenate(codes, axis=0),
            ops.concatenate(scales, axis=0),
            ops.concatenate(zeros, axis=0),
            ops.concatenate(group_indices, axis=0),
            ops.concatenate(input_scales, axis=0) if input_scales else None,
        )
