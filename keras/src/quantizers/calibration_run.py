"""The run scope of the calibration modes: one pass over one dataset.

GPTQ and AWQ share one driver. A `CalibrationRun` materializes the
activations behind the model's prefix layers once, then walks the
sequential blocks: for each block it finds the quantizable layers,
builds one `Calibrator` per layer, streams the block's inputs into them
through `calibration_scope`, quantizes the layers in execution-order
stages, and forwards the activations through the calibrated block for
the next one. What differs between the modes is declared, not coded
twice: the calibrator class and the forward batch size come from the
mode's `CalibrationStrategy`, and the undersampling threshold from the
calibrator.
"""

import math
import warnings
from contextlib import contextmanager
from contextlib import nullcontext

import numpy as np
from absl import logging

from keras.src import backend
from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.utils import should_quantize_layer


def calibration_no_grad_scope():
    """Returns a context manager that disables gradient tracking.

    Calibration is inference-only: it runs forward passes to accumulate
    statistics and then assigns quantized values to variables. On the torch
    backend those forwards would otherwise build autograd graphs, and
    because per-sample activations are retained across the whole
    calibration loop (and AWQ stashes activation samples for its clipping
    search), the graphs and every intermediate activation stay alive until
    the block completes - enough to exhaust GPU memory on models that fit
    comfortably otherwise. JAX and TensorFlow build no such graphs in eager
    mode, so this is a no-op there.
    """
    if backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return nullcontext()


def get_dataloader(
    tokenizer,
    sequence_length,
    dataset,
    num_samples=128,
    *,
    sampling="strided",
    seed=42,
    stride=None,
    eos_id=None,
):
    """
    Prepares and chunks the calibration dataloader, repeating short datasets.
    All processing happens on the CPU.

    Args:
        tokenizer: The tokenizer to use for text splitting.
        sequence_length: The length of each input sequence.
        dataset: The dataset to sample from.
        num_samples: The number of samples to generate.
        sampling: The sampling strategy to use. Possible values are
         1. "strided": Samples are taken at regular intervals.
         2. "linspace": Samples are taken at evenly spaced intervals.
         3. "random": Samples are taken at random positions.
        seed: The random seed for reproducibility. Used only if
         sampling="random"
        stride: The stride length for "strided" sampling.
        eos_id: The end-of-sequence token ID.

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

    pieces = []
    if isinstance(dataset_list[0], str):
        for i, s in enumerate(dataset_list):
            toks = ops.convert_to_numpy(tokenizer.tokenize(s)).reshape(-1)
            pieces.append(toks)
            # avoid windows that span document boundaries
            if eos_id is not None and i < len(dataset_list) - 1:
                pieces.append(np.array([eos_id], dtype=np.int32))
    else:
        for s in dataset_list:
            toks = ops.convert_to_numpy(s).reshape(-1)
            pieces.append(toks.astype(np.int32, copy=False))

    all_tokens = (
        pieces[0].astype(np.int32, copy=False)
        if len(pieces) == 1
        else np.concatenate(pieces, axis=0).astype(np.int32, copy=False)
    )

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

    # Choose deterministic, well-spread starts by default
    if sampling == "random":
        rng = np.random.default_rng(seed)
        starts = rng.integers(
            0, max_start + 1, size=num_samples, dtype=np.int64
        )
    elif sampling == "linspace":
        # even coverage with no RNG
        starts = np.linspace(0, max_start, num_samples, dtype=np.int64)
    elif sampling == "strided":
        # stride chosen to cover the space roughly uniformly
        if stride is None:
            stride = max(1, (max_start + 1) // num_samples)
        # Offset derived deterministically from the seed. Python's
        # built-in `hash()` must not be used here: string/tuple hashes are
        # randomized per process (PYTHONHASHSEED), which silently made
        # calibration windows - and therefore every quantization result -
        # unreproducible across runs despite the fixed seed.
        offset = (
            int(np.random.default_rng(seed).integers(0, max_start + 1))
            if max_start > 0
            else 0
        )
        starts = (offset + np.arange(num_samples, dtype=np.int64) * stride) % (
            max_start + 1
        )
    else:
        raise ValueError(f"Unknown sampling: {sampling}")

    # Gather contiguous windows
    # sliding_window_view avoids building a big index matrix
    windows = np.lib.stride_tricks.sliding_window_view(
        all_tokens, sequence_length
    )
    samples = windows[starts]  # (num_samples, sequence_length)
    return samples.astype(np.int32)[:, None, :]


def _stack_calibration_batch(samples):
    """Stacks a list of per-sample calibration activations into a single batch.

    Each element may be a 2D `[sequence, features]` or 3D
    `[1, sequence, features]` tensor. Every element is normalized to a leading
    batch axis of size 1 and concatenated along axis 0, producing a
    `[batch, sequence, features]` tensor that can be run through a block in a
    single forward pass.

    Args:
        samples: List of per-sample activation tensors.

    Returns:
        A single `[batch, sequence, features]` tensor.
    """
    normalized = []
    for sample in samples:
        if ops.ndim(sample) == 2:
            sample = ops.expand_dims(sample, axis=0)
        normalized.append(sample)
    if len(normalized) == 1:
        return normalized[0]
    return ops.concatenate(normalized, axis=0)


def find_layers_in_block(block):
    """
    Finds all Dense and EinsumDense layers in a transformer block.

    Args:
        block: A Keras layer representing a transformer block.
    Returns:
        A dict mapping layer paths to the corresponding Dense or EinsumDense
    """
    found_layers = {}
    for sub_layer in block._flatten_layers():
        # A quantizable layer may own sub-layers (e.g. a `Layer`
        # activation), so no leaf filtering here — collect every Dense/
        # EinsumDense reachable inside the block.
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[sub_layer.path] = sub_layer
    return found_layers


def _execution_stages(layer_names, execution_trace):
    """Groups a block's layers into sequential quantization stages.

    Reference GPTQ implementations quantize a block's sub-layers in
    topological order ("true sequential"): once an upstream sub-layer is
    quantized, downstream statistics are re-estimated on the quantized
    activations. Without this, e.g. an MLP's statistics are captured while the
    attention sub-layers are still full-precision, and the error
    corrections it derives are tuned to activations that no longer exist
    once attention is quantized too — which measurably degrades quality
    below plain round-to-nearest.

    Stages are derived from the calibration trace: layers are ordered by
    first invocation, and layers that consumed the *same* input tensor
    (e.g. the query/key/value projections, or an MLP's gate/up pair) share
    a stage since quantizing one cannot affect the others' inputs. Layers
    that never fired during tracing are placed in the first stage,
    preserving their (empty) statistics.

    Args:
        layer_names: Iterable of layer names in the block.
        execution_trace: Dict of `{name: (call_index, input_tensor)}`
            recorded by `stream_inputs`.

    Returns:
        List of lists of layer names, one list per stage, in execution
        order.
    """
    traced = [name for name in layer_names if name in execution_trace]
    untraced = [name for name in layer_names if name not in execution_trace]
    traced.sort(key=lambda name: execution_trace[name][0])

    stages = []
    current_stage, current_input_id = [], None
    for name in traced:
        input_id = id(execution_trace[name][1])
        if current_stage and input_id != current_input_id:
            stages.append(current_stage)
            current_stage = []
        current_stage.append(name)
        current_input_id = input_id
    if current_stage:
        stages.append(current_stage)

    if untraced:
        if stages:
            stages[0] = untraced + stages[0]
        else:
            stages = [untraced]
    return stages


@contextmanager
def stream_inputs(layers_map, calibrators, execution_trace=None):
    """Streams every target layer's inputs into its calibrator.

    Registers a calibration capture (`keras.src.quantizers.capture`) on
    each layer, which the dispatch machinery runs before each of the
    layer's forward passes, whichever forward that is. The capture lays
    the input out as the 2-D `[-1, rows]` matrix the calibrator's
    statistics describe and passes it to `calibrators[name].observe`.
    Every capture is removed on exit, even if an exception occurs;
    nothing on the layers is rebound.

    Args:
        layers_map: Dict[str, Layer]. Mapping from logical layer names to
            the layers to observe. Keys must match `calibrators`.
        calibrators: Dict[str, Calibrator]. Mapping from names to the
            calibrators that receive the inputs.
        execution_trace: Optional dict. When provided, each layer's FIRST
            capture records `{name: (call_index, input_tensor)}` into it:
            the block-level execution order and the identity of the input
            each layer consumes, from which `CalibrationRun` derives the
            within-block quantization stages. The recorded tensors are
            only kept alive for identity comparison; callers should drop
            the trace after use.

    Yields:
        None: The captures are active only within the `with` block.
    """
    call_counter = [0]

    def create_capture(name):
        def capture(inputs):
            if execution_trace is not None and name not in execution_trace:
                # Record block-level execution order and the input tensor's
                # identity on the first call (a live reference is kept so the
                # id cannot be recycled while tracing).
                execution_trace[name] = (call_counter[0], inputs)
                call_counter[0] += 1
            # Explicitly reshape the input tensor to be 2D, with the
            # second dimension matching the number of input features
            # expected by the layer's kernel.
            # This correctly handles inputs of any dimensionality
            # (e.g., 3D or 4D).
            calibrator = calibrators[name]
            calibrator.observe(ops.reshape(inputs, (-1, calibrator.rows)))

        return capture

    with calibration_scope(
        {layer: create_capture(name) for name, layer in layers_map.items()}
    ):
        yield


class CalibrationRun:
    """One calibration pass of a model's sequential blocks over a dataset.

    `Model.quantize` resolves the layer structure and the mode's
    `CalibrationStrategy` creates the run for it (`calibrate`). The run
    materializes the activations behind the prefix layers once, then
    walks the blocks in order. For each block it finds the quantizable
    layers, builds a calibrator per layer, streams the block's inputs into
    them, quantizes the layers in execution-order stages ("true
    sequential": after a stage is quantized, the later stages' statistics
    are re-estimated on the quantized upstream activations), and forwards
    the activations through the calibrated block for the next one.

    Args:
        strategy: The `CalibrationStrategy` of the mode. It supplies the
            calibrator class and the forward batch size.
        config: The mode's config.
        structure: Dict with keys `"pre_block_layers"` and
            `"sequential_blocks"`.
        filters: Optional filters that exclude layers from quantization.
    """

    def __init__(self, strategy, config, structure, filters=None):
        self.strategy = strategy
        self.config = config
        self.filters = filters
        self.pre_block_layers = structure.get("pre_block_layers", [])
        self.blocks = structure.get("sequential_blocks", [])
        if not self.blocks:
            raise ValueError(
                "No sequential blocks found in the provided structure to "
                "quantize."
            )
        self.num_samples = config.num_samples
        # Forward batch size of the sweeps and of the block-to-block
        # handoff. Batching changes the order in which the statistics
        # accumulate, so each mode declares its own.
        self.batch_size = strategy.calibration_batch_size(config)
        # Layers whose statistics saw too few calibration tokens relative
        # to their input width, collected across all blocks for a single
        # summary warning.
        self.undersampled = []

    def run(self, dataloader):
        """Calibrates and quantizes every block, in order.

        Args:
            dataloader: An iterable of token batches for the prefix layers.
        """
        logging.info("Starting model quantization...")
        inputs = self._prefix_outputs(dataloader)
        progbar = keras_utils.Progbar(target=len(self.blocks))
        for block_idx, block in enumerate(self.blocks):
            logging.info(f"Quantizing Block {block_idx}")
            self._calibrate_block(block_idx, block, inputs)
            if block_idx < len(self.blocks) - 1:
                logging.info(f"Generating inputs for block {block_idx + 1}...")
                inputs = self._next_inputs(block, inputs)
            progbar.update(current=block_idx + 1)
        self._warn_undersampled()
        logging.info("Quantization process complete.")

    def _prefix_outputs(self, dataloader):
        # The initial inputs are the outputs of the pre-block layers, one
        # activation per sample.
        inputs = []
        for batch in dataloader:
            batch = ops.convert_to_tensor(batch, dtype="int32")
            for layer in self.pre_block_layers:
                batch = layer(batch)
            inputs.append(batch)
        self.num_samples = min(self.num_samples, len(inputs))
        return inputs[: self.num_samples]

    def _batches(self, inputs):
        for start in range(0, self.num_samples, self.batch_size):
            yield _stack_calibration_batch(
                inputs[start : start + self.batch_size]
            )

    def _sweep(self, block, layers, calibrators, inputs, execution_trace=None):
        with stream_inputs(layers, calibrators, execution_trace):
            for batch in self._batches(inputs):
                _ = block(batch)

    def _calibrate_block(self, block_idx, block, inputs):
        layers = {
            name: layer
            for name, layer in find_layers_in_block(block).items()
            if should_quantize_layer(layer, self.filters)
        }
        if not layers:
            logging.info(
                f"  No quantizable layers found in block {block_idx}. Skipping."
            )
            return
        logging.info(f"Found layers: {list(layers)}")
        calibrators = {
            name: self.strategy.calibrator_cls(layer, self.config)
            for name, layer in layers.items()
        }
        execution_trace = {}
        self._sweep(block, layers, calibrators, inputs, execution_trace)

        # Quantize the block's layers in execution-order stages ("true
        # sequential", as in reference GPTQ): after each stage is
        # quantized, downstream stages' statistics are re-estimated so
        # their solves are computed against the quantized upstream
        # activations they will actually see at inference.
        stages = _execution_stages(layers, execution_trace)
        del execution_trace
        for stage_idx, stage_names in enumerate(stages):
            if stage_idx > 0:
                for name in stage_names:
                    calibrators[name].release()
                    calibrators[name] = self.strategy.calibrator_cls(
                        layers[name], self.config
                    )
                self._sweep(
                    block,
                    {name: layers[name] for name in stage_names},
                    {name: calibrators[name] for name in stage_names},
                    inputs,
                )
            for name in stage_names:
                calibrator = calibrators[name]
                self._tally_undersampling(name, calibrator)
                logging.info(f"Quantizing {name}...")
                calibrator.quantize()
                calibrator.release()

    def _tally_undersampling(self, name, calibrator):
        threshold = calibrator.warn_tokens_per_row
        if threshold is None:
            return
        tokens = int(calibrator.num_samples)
        rows = int(calibrator.rows)
        if tokens < threshold * rows:
            self.undersampled.append((name, tokens, rows))

    def _next_inputs(self, block, inputs):
        next_inputs = []
        for batch in self._batches(inputs):
            output = block(batch)
            if isinstance(output, (list, tuple)):
                output = output[0]
            # Split the batched output back into per-sample activations so
            # the next block can be calibrated identically.
            for sample_idx in range(ops.shape(output)[0]):
                next_inputs.append(output[sample_idx])
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
