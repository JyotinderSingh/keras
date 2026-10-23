"""A pytest plugin that runs one part of the selected tests. Fork only;
not part of a PR (SESSION.md section 8, the row sessions).

    KSHARD=2/4 python -m pytest -p benchmarks.quantized_matmul.pod.shard ...

The selected tests, in collection order, go to the parts in turn: test
`i` to part `i % N + 1`. So each part gets about 1/N of the tests of each
file, and the slow files spread over the parts. The plugin runs after the
other selections (`-k`, `--deselect`) and deselects the tests of the
other parts. With no `KSHARD`, it changes nothing. The "require" runner
(`pytest_require_kernels.py`) passes `-p` on to pytest.
"""

import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    spec = os.environ.get("KSHARD", "")
    if not spec:
        return
    part, parts = (int(value) for value in spec.split("/"))
    if not 1 <= part <= parts:
        raise ValueError(f"KSHARD must be <part>/<parts>. Received: {spec}")
    keep = [item for i, item in enumerate(items) if i % parts == part - 1]
    drop = [item for i, item in enumerate(items) if i % parts != part - 1]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
