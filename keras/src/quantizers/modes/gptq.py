from keras.src.dtype_policies.dtype_policy import GPTQDTypePolicy
from keras.src.quantizers.gptq import GPTQCalibrator
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.calibration import CalibrationStrategy


class GPTQStrategy(CalibrationStrategy):
    """GPTQ post-training quantization (calibration-based, 2/3/4/8-bit).

    GPTQ quantizes the kernel one column at a time and corrects the columns
    still to come with the inverse Hessian of the layer's inputs.
    """

    name = "gptq"
    config_cls = GPTQConfig
    policy_cls = GPTQDTypePolicy
    calibrator_cls = GPTQCalibrator
