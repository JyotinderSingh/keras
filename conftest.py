try:
    # When using torch and tensorflow, torch needs to be imported first,
    # otherwise it will segfault upon import. This should force the torch
    # import to happen first for all tests.
    import torch  # noqa: F401
except ImportError:
    torch = None

import contextlib  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402

from keras.src.backend import SUPPORTS_GRADIENT  # noqa: E402
from keras.src.backend import config as backend_config  # noqa: E402
from keras.src.backend.config import PLUGGABLE_BACKENDS
from keras.src.backend.config import backend  # noqa: E402
from keras.src.quantizers import kernels  # noqa: E402
from keras.src.utils.module_utils import get_pluggable_backend_module


def pytest_addoption(parser):
    parser.addoption(
        "--kernel-policy",
        choices=("auto", "off", "require"),
        default="auto",
        help=(
            "The policy of the quantized kernels for the session. `auto` "
            "leaves the policy as it is. Under `off` or `require`, a test "
            "that leaves another policy fails. `require` fails at the "
            "start on a platform with no kernel."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )
    config.addinivalue_line(
        "markers",
        "multi_device: mark test for running with multiple devices only",
    )
    config.addinivalue_line(
        "markers",
        "no_pytest_xdist: mark test that cannot run under pytest-xdist "
        "workers (e.g. tests that bind to a fixed port)",
    )

    # Disable CUDA TF32 to get higher numerical accuracy for correctness tests.
    if backend() == "jax":
        import jax

        if jax.default_backend() == "gpu":
            jax.config.update("jax_default_matmul_precision", "float32")
    elif backend() == "tensorflow":
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            tf.config.experimental.enable_tensor_float_32_execution(False)
    elif backend() == "torch":
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False

    # `require` on a platform with no kernel fails here, once, with the
    # requirements of each kernel.
    try:
        kernels._check_policy(config.getoption("--kernel-policy"))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error


@pytest.fixture(scope="session", autouse=True)
def kernel_policy(request):
    """Runs the session under `--kernel-policy`."""
    value = request.config.getoption("--kernel-policy")
    if value == "auto":
        context = contextlib.nullcontext()
    else:
        context = kernels.policy(value)
    with context:
        yield


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Under `--kernel-policy=off` or `require`, fails a test that leaves
    another kernel policy, and sets the option again for the next test.

    `trylast`: this runs after the teardown of the fixtures. After the
    last test, the session fixture has put the previous policy back.
    """
    expected = item.config.getoption("--kernel-policy")
    if expected == "auto" or nextitem is None:
        return
    policy = backend_config._quantized_kernels_policy()
    if policy != expected:
        backend_config._set_quantized_kernels_policy(expected)
        pytest.fail(
            f'The test leaves the kernel policy "{policy}", not "{expected}".',
            pytrace=False,
        )


def pytest_collection_modifyitems(config, items):
    has_multiple_devices = False

    backend_skipped_tests = set()
    if backend() in PLUGGABLE_BACKENDS:
        backend_module_file = get_pluggable_backend_module().__file__
        exclusions_path = os.path.join(
            # Remove `src/__init__.py`.
            os.path.dirname(os.path.dirname(backend_module_file)),
            "excluded_tests.txt",
        )
        # An installed backend package has no exclusion list.
        if os.path.exists(exclusions_path):
            with open(exclusions_path, "r") as file:
                # Exclude empty lines and comments.
                backend_skipped_tests = {
                    stripped
                    for line in file.readlines()
                    if (stripped := line.strip())
                    and not stripped.startswith("#")
                }

    if backend() == "jax":
        import jax

        has_multiple_devices = jax.device_count() > 1
    elif backend() == "torch":
        from keras.src.backend.torch import distribution_lib

        has_multiple_devices = distribution_lib.get_device_count() > 1

    requires_trainable_backend = pytest.mark.skipif(
        not SUPPORTS_GRADIENT,
        reason="Trainer not implemented for this backend.",
    )
    requires_multiple_devices = (
        None
        if has_multiple_devices
        else pytest.mark.skip(reason="Requires multiple devices")
    )
    # Distributed tests (e.g. torch.distributed) cannot run under
    # pytest-xdist workers because process group init deadlocks when
    # multiple workers attempt to bind to the same port.
    is_xdist_worker = "PYTEST_XDIST_WORKER" in os.environ

    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        if requires_multiple_devices and "multi_device" in item.keywords:
            item.add_marker(requires_multiple_devices)
        if is_xdist_worker and "no_pytest_xdist" in item.keywords:
            item.add_marker(
                pytest.mark.skip(
                    reason="This test cannot run under pytest-xdist workers"
                )
            )

        # Skip concrete tests listed in the backend specific file.
        if item.nodeid in backend_skipped_tests:
            item.add_marker(
                skip_if_backend(
                    backend(),
                    f"Not supported operation by {backend()} backend",
                )
            )


def skip_if_backend(given_backend, reason):
    return pytest.mark.skipif(backend() == given_backend, reason=reason)
