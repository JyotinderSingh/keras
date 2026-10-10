"""The run scope of the calibration modes: one pass over one dataset.

GPTQ and AWQ share one driver, `CalibrationRun`. What differs between the
modes is declared, not coded twice: the calibrator class comes from the
mode's `CalibrationStrategy`, the forward batch size from its config, and
the undersampling threshold from the calibrator.
"""

import math
import warnings
from contextlib import nullcontext

import numpy as np
from absl import logging

from keras.src import backend
from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.quantizers.capture import calibration_scope


def calibration_no_grad_scope():
    """Returns a context manager that disables gradient tracking.

    On torch the calibration forwards would otherwise build autograd
    graphs, which the activations retained across the run keep alive.
    JAX and TensorFlow build none in eager mode.
    """
    if backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return nullcontext()


# Seed of the offset of the first calibration window.
_WINDOW_OFFSET_SEED = 42


def get_dataloader(tokenizer, sequence_length, dataset, num_samples=128):
    """Cuts `num_samples` token windows out of the calibration dataset.

    The dataset is tokenized into one token stream, repeated if it is too
    short, and windows of `sequence_length` tokens are taken at a regular
    stride from a fixed offset. All processing happens on the CPU.

    Args:
        tokenizer: The tokenizer to use for text splitting.
        sequence_length: The length of each input sequence.
        dataset: The dataset to sample from.
        num_samples: The number of samples to generate.

    Returns:
        np.ndarray of shape (num_samples, 1, sequence_length), dtype int32.
    """
    if not hasattr(dataset, "__iter__") or isinstance(dataset, (str, bytes)):
        raise TypeError(
            "The `dataset` argument must be an iterable (e.g., a list of "
            "strings, a generator, or a NumPy array). Got type: "
            f"{type(dataset).__name__}. Please pass the loaded dataset "
            "directly."
        )

    dataset_list = list(dataset)
    if not dataset_list:
        raise ValueError("Provided dataset is empty.")

    if isinstance(dataset_list[0], str):
        pieces = [
            ops.convert_to_numpy(tokenizer.tokenize(s)).reshape(-1)
            for s in dataset_list
        ]
    else:
        pieces = [
            ops.convert_to_numpy(s).reshape(-1).astype(np.int32, copy=False)
            for s in dataset_list
        ]
    all_tokens = np.concatenate(pieces, axis=0).astype(np.int32, copy=False)

    required_tokens = num_samples * sequence_length
    if all_tokens.size < required_tokens:
        repeats = math.ceil(required_tokens / max(1, all_tokens.size))
        all_tokens = np.tile(all_tokens, repeats)

    max_start = all_tokens.size - sequence_length
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens to form one sample of length {sequence_length} "
            f"(have {all_tokens.size})."
        )

    # A stride that covers the stream roughly uniformly, from an offset
    # derived from a fixed seed. Python's `hash()` must not be used here:
    # it is randomized per process, so the windows, and every quantization
    # result, would differ between runs.
    stride = max(1, (max_start + 1) // num_samples)
    offset = (
        int(
            np.random.default_rng(_WINDOW_OFFSET_SEED).integers(
                0, max_start + 1
            )
        )
        if max_start > 0
        else 0
    )
    starts = (offset + np.arange(num_samples, dtype=np.int64) * stride) % (
        max_start + 1
    )
    # `sliding_window_view` avoids building a big index matrix.
    windows = np.lib.stride_tricks.sliding_window_view(
        all_tokens, sequence_length
    )
    return windows[starts].astype(np.int32)[:, None, :]


def _execution_stages(block, layers, batch):
    """Groups a block's layers into sequential quantization stages.

    Reference GPTQ implementations quantize a block's sub-layers in
    topological order ("true sequential"): the statistics of a layer are
    taken after the layers upstream of it are quantized, so its solve is
    computed against the activations it sees at inference.

    One forward pass of `batch` records the order the layers first run in
    and the input each consumes. Consecutive layers that consume the same
    input tensor (the query/key/value projections, an MLP's gate/up pair)
    share a stage, since quantizing one cannot change the others' inputs.
    The layers that do not run in that pass form the last stage.

    Args:
        block: The block the layers belong to.
        layers: The layers to group.
        batch: One batch of the block's inputs.

    Returns:
        List of lists of layers, one list per stage, in execution order.
    """
    trace = {}

    def recorder(layer):
        def record(inputs):
            # The first call only. The trace keeps the input alive, so its
            # identity cannot be reused within the pass.
            trace.setdefault(id(layer), (layer, inputs))

        return record

    with calibration_scope({layer: recorder(layer) for layer in layers}):
        block(batch)
    stages, stage_input = [], None
    for layer, inputs in trace.values():
        if not stages or inputs is not stage_input:
            stages.append([])
        stages[-1].append(layer)
        stage_input = inputs
    untraced = [layer for layer in layers if id(layer) not in trace]
    if untraced:
        stages.append(untraced)
    return stages


class CalibrationRun:
    """One calibration pass of a model's sequential blocks over a dataset.

    The mode's `CalibrationStrategy` creates the run for `Model.quantize`
    (`model_run`) before any layer changes, so a missing structure,
    dataset or tokenizer refuses the call while the model is untouched.
    The walk of `Model.quantize` asks the run whether it `covers` each
    layer and hands it the float layers to quantize (`add`).

    `run()` materializes the activations behind the prefix layers once,
    then walks the blocks in order. For each block it groups the layers
    it was handed into execution-order stages ("true sequential"). For
    each stage in turn it builds a calibrator per layer, from the run's
    config, passes the block's inputs through the block to them, and
    quantizes the stage's layers, so a later stage observes the quantized
    output of the stages before it. A calibrator's write-back swaps its
    float layer for the quantized one in one step. The run then runs the
    calibrated block to produce the next block's inputs. The activations
    are kept as the batches the blocks run on. `quantized` lists the
    layers swapped so far, also when `run()` raises.

    Args:
        strategy: The `CalibrationStrategy` of the mode. It supplies the
            calibrator class.
        config: The mode's config. It holds the dataset and the
            tokenizer, the solver parameters, and the number of samples
            and the batch size of the run.
        structure: Dict with keys `"pre_block_layers"` and
            `"sequential_blocks"`.

    Each block is calibrated on the quantized outputs of the block
    before it, for both modes: the statistics describe the activations
    the block sees at inference. This is the GPTQ references' rule; the
    AWQ references calibrate every block on the float model's
    activations. On SmolLM2-135M the two gave the same held-out
    perplexity over three calibration seeds.
    """

    def __init__(self, strategy, config, structure):
        self.strategy = strategy
        self.config = config
        self.pre_block_layers = structure.get("pre_block_layers", [])
        self.blocks = structure.get("sequential_blocks", [])
        if not self.blocks:
            raise ValueError(
                "No sequential blocks found in the provided structure to "
                "quantize."
            )
        if config.dataset is None or config.tokenizer is None:
            raise ValueError(
                f"{strategy.name.upper()} quantization requires a dataset "
                "and a tokenizer. Please provide them in the "
                f"`{strategy.config_cls.__name__}`."
            )
        self.dataloader = get_dataloader(
            config.tokenizer,
            config.sequence_length,
            config.dataset,
            num_samples=config.num_samples,
        )
        self.batch_size = int(config.calibration_batch_size)
        # A block layer is covered when it declares support for the mode
        # as `Layer.quantize` reads it, so a third-party layer that lists
        # the mode in its `variable_serialization_spec` is covered, as
        # `Dense` and `EinsumDense` are. A quantizable layer may own
        # sub-layers (a `Layer` activation), so this does not filter
        # leaves.
        self._covered = {
            id(layer)
            for block in self.blocks
            for layer in block._flatten_layers()
            if layer._supports_quantization_mode(strategy)
        }
        # The float layers the walk handed over, and the ones swapped so
        # far.
        self._targets = set()
        self.quantized = []
        # Layers whose statistics saw too few calibration tokens relative
        # to their input width, and layers that observed no input,
        # collected across all blocks for one warning each.
        self.undersampled = []
        self.unreached = []

    def covers(self, layer):
        """Whether `layer` is a block layer that supports the mode."""
        return id(layer) in self._covered

    def add(self, layer, type_check=True):
        """Queues a float `layer`, or raises as `Layer.quantize` would."""
        layer._check_quantizable(self.config, type_check)
        self._targets.add(id(layer))

    def run(self):
        """Calibrates and quantizes every block, in order."""
        if not self._targets:
            logging.info("No layers to calibrate. Skipping.")
            return
        logging.info("Starting model quantization...")
        with calibration_no_grad_scope():
            inputs = self._prefix_outputs(self.dataloader)
            progbar = keras_utils.Progbar(target=len(self.blocks))
            for block_idx, block in enumerate(self.blocks):
                logging.info(f"Quantizing Block {block_idx}")
                self._calibrate_block(block_idx, block, inputs)
                if block_idx < len(self.blocks) - 1:
                    logging.info(
                        f"Generating inputs for block {block_idx + 1}..."
                    )
                    inputs = self._next_inputs(block, inputs)
                progbar.update(current=block_idx + 1)
        self._warn_undersampled()
        self._warn_unreached()
        logging.info("Quantization process complete.")

    def _prefix_outputs(self, dataloader):
        # The pre-block layers run one sample at a time; their outputs are
        # stacked into the batches the blocks run on.
        outputs = []
        for batch in dataloader:
            batch = ops.convert_to_tensor(batch, dtype="int32")
            for layer in self.pre_block_layers:
                batch = layer(batch)
            outputs.append(batch)
        return [
            ops.concatenate(outputs[start : start + self.batch_size], axis=0)
            for start in range(0, len(outputs), self.batch_size)
        ]

    def _float_layers(self, block):
        # The layers the walk handed over that are still float. A layer
        # that two blocks share is solved in the first.
        return {
            layer.path: layer
            for layer in block._flatten_layers()
            if id(layer) in self._targets and not layer._is_quantized
        }

    def _calibrator(self, layer):
        # The run's config: the swap builds the variables from it, so the
        # solve and the packing agree.
        return self.strategy.calibrator_cls(self.strategy, layer, self.config)

    def _calibrate_block(self, block_idx, block, inputs):
        layers = self._float_layers(block)
        if not layers:
            logging.info(
                f"  No quantizable layers found in block {block_idx}. Skipping."
            )
            return
        logging.info(f"Found layers: {list(layers)}")
        # Quantize the block's layers in execution-order stages ("true
        # sequential", as in reference GPTQ): a stage observes the block
        # after the stages before it are quantized, so its solves are
        # computed against the activations the layers see at inference.
        stages = _execution_stages(block, list(layers.values()), inputs[0])
        for stage in stages:
            self._calibrate_stage(block, stage, inputs)

    def _calibrate_stage(self, block, layers, inputs):
        # The statistics live for this call only, so the run holds those
        # of one stage at a time.
        calibrators = [self._calibrator(layer) for layer in layers]
        with calibration_scope({c.layer: c.observe for c in calibrators}):
            for batch in inputs:
                block(batch)
        for calibrator in calibrators:
            name = calibrator.layer.path
            if calibrator.num_samples:
                self._tally_undersampling(name, calibrator)
            else:
                self.unreached.append(name)
            logging.info(f"Quantizing {name}...")
            calibrator.quantize()
            self.quantized.append(calibrator.layer)

    def _tally_undersampling(self, name, calibrator):
        threshold = calibrator.warn_tokens_per_row
        if threshold is None:
            return
        tokens = int(calibrator.num_samples)
        rows = int(calibrator.rows)
        if tokens < threshold * rows:
            self.undersampled.append((name, tokens, rows))

    def _next_inputs(self, block, inputs):
        # Each batch's output goes to the next block as it is: the first
        # output of a block that returns several.
        next_inputs = []
        for batch in inputs:
            output = block(batch)
            if isinstance(output, (list, tuple)):
                output = output[0]
            next_inputs.append(output)
        return next_inputs

    def _warn_undersampled(self):
        if not self.undersampled:
            return
        warnings.warn(
            self.strategy.calibrator_cls.undersampling_warning(
                self.undersampled
            ),
            stacklevel=2,
        )

    def _warn_unreached(self):
        if not self.unreached:
            return
        warnings.warn(
            f"{self.strategy.name.upper()} calibration observed no input "
            f"for {len(self.unreached)} layer(s), so their weights are "
            "quantized without calibration statistics: "
            f"{', '.join(self.unreached)}. A layer that the blocks do not "
            "run on the calibration data, such as a layer that runs only "
            "in training, is never observed. To keep such a layer in "
            "float, exclude it with `filters`.",
            stacklevel=2,
        )
