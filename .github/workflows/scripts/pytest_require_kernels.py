"""Runs pytest under the kernel policy `"require"`.

The policy is private and process-global, and library code reads no
environment variable for it. So this runner sets it, then runs pytest in
the same process. `set_policy("require")` raises when the platform has
no kernel, for example when Triton does not import on a CUDA host. Each
routable quantized call then runs the platform's kernel, and no call
falls back to the reference path because a version or an allow-list row
is missing.

A test that sets another policy and does not put `"require"` back would
run the tests after it under that policy, with no failure. So the
plugin `PolicyLeakCheck` fails each test whose teardown finds a policy
other than `"require"`, and sets `"require"` again for the next test.

Usage, from the repository root:

    PYTHONPATH=. python .github/workflows/scripts/pytest_require_kernels.py \
        <pytest arguments>
"""

import sys

import pytest

from keras.src.quantizers import kernels


class PolicyLeakCheck:
    """Fails a test that leaves a kernel policy other than `"require"`."""

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item):
        # `trylast`: this runs after the teardown of the fixtures.
        policy = kernels._state.policy
        if policy != "require":
            kernels.set_policy("require")
            pytest.fail(
                f'The test leaves the kernel policy "{policy}", not "require".',
                pytrace=False,
            )


kernels.set_policy("require")
sys.exit(pytest.main(sys.argv[1:], plugins=[PolicyLeakCheck()]))
