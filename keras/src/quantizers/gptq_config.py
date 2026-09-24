from keras.src.api_export import keras_export
from keras.src.quantizers.quantization_config import QuantizationConfig


@keras_export("keras.quantizers.GPTQConfig")
class GPTQConfig(QuantizationConfig):
    """Configuration class for GPTQ (Accurate Post-Training Quantization).

    GPTQ is a post-training quantization method that quantizes weights to
    2, 3, 4 or 8 bits while minimizing the impact on model accuracy. It
    accumulates the Hessian of each layer's inputs over calibration data,
    quantizes the weights one column at a time, and corrects the columns
    still to come for the quantization error already made.

    Methodology:
    1. Collects the Hessian of each layer's inputs from calibration data
    2. Quantizes the weights column by column with error correction
    3. Reorders the columns by importance (optional)
    4. Quantizes a block's layers in execution order, re-estimating the
       Hessians of later layers on the quantized activations

    References:
    - Original GPTQ paper: "GPTQ: Accurate Post-Training Quantization for
      Generative Pre-trained Transformers" (https://arxiv.org/abs/2210.17323)
    - Reference implementation: https://github.com/IST-DASLab/gptq

    Args:
        dataset: The calibration dataset. It can be an iterable that yields
            strings or pre-tokenized numerical tensors (e.g., a list of
            strings, a generator, or a NumPy array). This data is used to
            analyze the model's activations.
        tokenizer: A tokenizer instance (or a similar callable) that is used
            to process the `dataset` if it contains strings.
        weight_bits: The number of bits to quantize weights to. Supported
            values are 2, 3, 4 and 8. Defaults to 4.
        num_samples: The number of calibration data samples to use from the
            dataset. Defaults to 128.
        calibration_batch_size: The number of calibration samples to run
            through each block per forward pass during calibration. Larger
            values reduce the number of forward passes (and therefore
            wall-clock calibration time) at the cost of higher peak
            activation memory. The Hessian accumulates over the observed
            rows, so the result is the same up to floating-point
            accumulation order. Defaults to 8.
        per_channel: Whether the scale and zero point are computed per
            output channel. If `False`, one scale covers the whole kernel.
            Defaults to `True`.
        sequence_length: The sequence length to use for each calibration
            sample. Defaults to 512.
        hessian_damping: The fraction of the mean Hessian diagonal added to
            the diagonal for stabilization before inversion. Defaults to
            0.01.
        group_size: The size of weight groups to quantize together. A
            `group_size` of -1 means one group spanning all input features:
            per-channel, or whole-tensor when `per_channel=False`.
            Defaults to 128.
        symmetric: If `True`, uses symmetric quantization. If `False`, uses
            asymmetric quantization. Defaults to `False`.
        activation_order: If `True`, reorders weight columns by the Hessian
            diagonal so the most salient columns are quantized first, which
            can improve quantization accuracy. Defaults to `False`.
        quantization_layer_structure: A dictionary defining the model's
            quantization structure. It should contain:
            - "pre_block_layers": list of layers to run before the first
              block (e.g., embedding layer).
            - "sequential_blocks": list of transformer blocks to quantize
              sequentially.
            If not provided, the model must implement
            `get_quantization_layer_structure`.

    Example:
    ```python
    from keras.quantizers import GPTQConfig

    # Create configuration for 4-bit GPTQ quantization
    config = GPTQConfig(
        dataset=calibration_data,          # Your calibration dataset
        tokenizer=your_tokenizer,          # Tokenizer for text data
        weight_bits=4,                     # Quantize to 4 bits
        num_samples=128,                   # Number of calibration samples
        sequence_length=512,               # Sequence length for each sample
        group_size=128,                    # Weight grouping for quantization
        activation_order=True,             # Reorder columns by importance
    )

    # Apply quantization to your model
    model.quantize("gptq", config=config)
    ```

    """

    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        weight_bits: int = 4,
        num_samples: int = 128,
        calibration_batch_size: int = 8,
        per_channel: bool = True,
        sequence_length: int = 512,
        hessian_damping: float = 0.01,
        group_size: int = 128,
        symmetric: bool = False,
        activation_order: bool = False,
        quantization_layer_structure: dict = None,
    ):
        super().__init__()
        if weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                f"Unsupported weight_bits {weight_bits}. "
                "Supported values are 2, 3, 4, and 8."
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if calibration_batch_size <= 0:
            raise ValueError(
                "calibration_batch_size must be a positive integer."
            )
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        if hessian_damping < 0 or hessian_damping > 1:
            raise ValueError("hessian_damping must be between 0 and 1.")
        if group_size < -1 or group_size == 0:
            raise ValueError(
                "Invalid group_size. Supported values are -1 (whole-tensor) "
                "or a positive integer, "
                f"but got {group_size}."
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.calibration_batch_size = calibration_batch_size
        self.per_channel = per_channel
        self.sequence_length = sequence_length
        self.hessian_damping = hessian_damping
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.activation_order = activation_order
        self.quantization_layer_structure = quantization_layer_structure

    @property
    def mode(self):
        return "gptq"

    def dtype_policy_string(self):
        """Returns the dtype policy string for this configuration.

        Returns:
            A string representing the dtype policy, e.g. "gptq/4/128".
        """
        return f"gptq/{self.weight_bits}/{self.group_size}"

    def get_config(self):
        return {
            # Dataset, tokenizer and quantization layer structure are only
            # required for one-time calibration and are not saved in the
            # config. The structure also holds references to live layer
            # objects, which cannot be serialized.
            "dataset": None,
            "tokenizer": None,
            "quantization_layer_structure": None,
            "weight_bits": self.weight_bits,
            "num_samples": self.num_samples,
            "calibration_batch_size": self.calibration_batch_size,
            "per_channel": self.per_channel,
            "sequence_length": self.sequence_length,
            "hessian_damping": self.hessian_damping,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "activation_order": self.activation_order,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
