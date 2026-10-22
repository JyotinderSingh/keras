"""Raises an `ImportError` when the installed torch is too old.

`keras/src/backend/torch/__init__.py` imports this module before the
modules that use newer torch APIs, such as `DTensor` from
`torch.distributed.tensor`. An older torch then fails with this message,
not with an `ImportError` for one of those APIs.
"""

import torch
from packaging.version import Version

# Keep in sync with the PyTorch row of the "Backend Compatibility Table" in
# `README.md`.
MIN_TORCH_VERSION = "2.6.0"


def check_torch_version(version):
    """Raises an `ImportError` if torch `version` is below the minimum.

    Only the release part of `version` is compared, so pre-release, dev
    and local builds of the minimum version pass, for example
    `2.6.0a0+git1eba9b3` or `2.6.0+cpu`.

    Args:
        version: A torch version string, such as `torch.__version__`.
    """
    release = Version(Version(version).base_version)
    if release < Version(MIN_TORCH_VERSION):
        raise ImportError(
            f"The Keras torch backend requires torch {MIN_TORCH_VERSION} or "
            f"newer, but torch {version} is installed. Upgrade torch with "
            "`pip install --upgrade torch`, or select another backend with "
            "the `KERAS_BACKEND` environment variable."
        )


check_torch_version(torch.__version__)
