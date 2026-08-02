"""Shared test fixtures and markers.

`requires_cifar10` exists because ten tests in `test_data.py` exercise the real
CIFAR-10 split, and `build_datasets` is `download=True`. On a developer machine
the data is already there and nothing is skipped. On a fresh CI runner it is
not, and the download comes from `www.cs.toronto.edu`, which is slow and
sometimes unreachable from cloud IPs -- the first run of this project's workflow
sat in `pytest` for nineteen minutes with no output and was killed by the job
timeout.

Skipping is the right failure mode there: a dataset that cannot be fetched is an
availability problem, not a defect in this repository, and a bounded skip that
names itself in the test output is far more useful than a job that hangs. The CI
workflow fetches the dataset in its own step first, so a green build normally
runs all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# torchvision extracts the python pickles here under `cfg.data.root`.
CIFAR10_DIR = Path(__file__).resolve().parents[1] / "data" / "cifar-10-batches-py"

requires_cifar10 = pytest.mark.skipif(
    not CIFAR10_DIR.exists(),
    reason=(
        f"CIFAR-10 not found at {CIFAR10_DIR}. Fetch it once with "
        '`python -c "from torchvision.datasets import CIFAR10; '
        "CIFAR10(root='data', download=True)\"`."
    ),
)
