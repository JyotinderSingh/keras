from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class Int8Strategy(QuantizationStrategy):
    """W8A8 dynamic quantization (int8 weights times int8 activations)."""

    name = "int8"
    config_cls = Int8QuantizationConfig
