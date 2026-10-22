import re

from absl.testing import parameterized

from keras.src import testing
from keras.src.backend.torch.version_check import MIN_TORCH_VERSION
from keras.src.backend.torch.version_check import check_torch_version


class CheckTorchVersionTest(testing.TestCase):
    @parameterized.named_parameters(
        ("minimum", "2.6.0"),
        ("minimum_local_build", "2.6.0+cpu"),
        ("minimum_pre_release", "2.6.0a0+git1eba9b3"),
        ("minimum_dev_build", "2.6.0.dev20241112"),
        ("newer", "2.14.1"),
    )
    def test_supported_version_passes(self, version):
        self.assertIsNone(check_torch_version(version))

    @parameterized.named_parameters(
        ("without_public_dtensor", "2.4.1"),
        ("previous_minor", "2.5.1"),
        ("previous_minor_local_build", "2.5.1+cu124"),
    )
    def test_old_version_raises(self, version):
        with self.assertRaisesRegex(
            ImportError,
            f"requires torch {re.escape(MIN_TORCH_VERSION)} or newer, but "
            f"torch {re.escape(version)} is installed",
        ):
            check_torch_version(version)
