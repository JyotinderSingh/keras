"""Runs pytest under the kernel policy `"require"`.

The policy is private and process-global, and library code reads no
environment variable for it. So this runner sets it, then runs pytest in
the same process. `set_policy("require")` raises when the platform has
no kernel, for example when Triton does not import on a CUDA host. Each
routable quantized call then runs the platform's kernel, and no call
falls back to the reference path because a version or an allow-list row
is missing.

Usage, from the repository root:

    PYTHONPATH=. python .github/workflows/scripts/pytest_require_kernels.py \
        <pytest arguments>
"""

import sys

import pytest

from keras.src.quantizers import kernels

kernels.set_policy("require")
sys.exit(pytest.main(sys.argv[1:]))
