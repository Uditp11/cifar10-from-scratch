"""CIFAR-10 dataset construction. See ARCHITECTURE.md section 3 and ADR-001.

`build_datasets` needs train and val to carry *different* transforms despite
both being drawn from the same underlying CIFAR-10 train split. A `Subset`
shares its parent's transform, so a single `CIFAR10` instance can't serve
both. We construct two separate `CIFAR10` instances (one with the train
transform, one with the eval transform) and subset each with the same index
partition. The expensive part -- unpickling the split from disk -- is cached
per `(root, train)` and shallow-copied per transform, so the two instances
are genuinely distinct objects without a redundant disk read.
"""

from __future__ import annotations

import copy
import json
import random
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from cifar.config import Config
from cifar.data.transforms import build_eval_transform, build_train_transform

__all__ = ["build_datasets", "generate_val_indices", "REPO_ROOT", "VAL_INDICES_PATH"]

# Anchored to the repo root (not CWD) so build_datasets works regardless of
# where the caller was invoked from. Later sprints needing a repo-relative
# path (checkpoints, run dirs, etc.) should reuse this rather than
# reinventing the anchor.
REPO_ROOT = Path(__file__).resolve().parents[3]
VAL_INDICES_PATH = REPO_ROOT / "configs" / "val_indices.json"
_NUM_CLASSES = 10  # CIFAR-10


@lru_cache(maxsize=None)
def _load_cifar10(root: str, train: bool) -> CIFAR10:
    """Load and cache the raw CIFAR-10 split for `(root, train)`, transform-free.

    `CIFAR10.__init__` unpickles the full split from disk; callers that need
    the same split under different transforms shallow-copy this and set
    `.transform`, so repeated `build_datasets` calls (e.g. across tests)
    don't re-read the files.
    """
    return CIFAR10(root=root, train=train, download=True)


def _dataset_with_transform(root: Path, train: bool, transform) -> CIFAR10:
    ds = copy.copy(_load_cifar10(str(root), train))
    ds.transform = transform
    return ds


def generate_val_indices(root: Path, val_size: int, seed: int = 0) -> list[int]:
    """One-time stratified sample of `val_size` indices from the CIFAR-10 train split.

    Run once (see `scripts/make_val_split.py`) to produce the committed
    `configs/val_indices.json` (ADR-001). Never called at training time --
    the committed file is the source of truth regardless of `cfg.repro.seed`.

    Args:
        root: CIFAR-10 data directory, downloaded if missing.
        val_size: Total validation size; must be divisible by 10 for an
            even per-class split.
        seed: Seed for the one-time sampling draw. Irrelevant after the
            file is committed.

    Returns:
        Sorted list of `val_size` indices into the 50,000-image train split.
    """
    if val_size % _NUM_CLASSES != 0:
        raise ValueError(f"val_size ({val_size}) must be divisible by {_NUM_CLASSES}")
    per_class = val_size // _NUM_CLASSES

    base = CIFAR10(root=root, train=True, download=True)
    by_class: dict[int, list[int]] = {c: [] for c in range(_NUM_CLASSES)}
    for idx, target in enumerate(base.targets):
        by_class[target].append(idx)

    rng = random.Random(seed)
    indices: list[int] = []
    for c in range(_NUM_CLASSES):
        indices.extend(rng.sample(by_class[c], per_class))
    indices.sort()
    return indices


def build_datasets(cfg: Config) -> tuple[Dataset, Dataset, Dataset]:
    """Build (train, val, test).

    Train gets train transforms; val and test get eval transforms. The val
    split comes from the fixed index file at `configs/val_indices.json` --
    never a random split (ADR-001).

    Args:
        cfg: Full config; reads `cfg.data`.

    Returns:
        (train_ds, val_ds, test_ds): disjoint `Subset`s of the CIFAR-10
        train split for train/val, and the untouched 10,000-image test
        split for test.
    """
    if not VAL_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"{VAL_INDICES_PATH} not found. Generate it once with "
            "scripts/make_val_split.py and commit it (see ADR-001)."
        )
    val_indices: list[int] = json.loads(VAL_INDICES_PATH.read_text())
    if len(val_indices) != cfg.data.val_size:
        raise ValueError(
            f"configs/val_indices.json has {len(val_indices)} indices but "
            f"cfg.data.val_size is {cfg.data.val_size} -- regenerate the split "
            "file if val_size genuinely changed."
        )

    train_base = _dataset_with_transform(cfg.data.root, True, build_train_transform(cfg))
    val_base = _dataset_with_transform(cfg.data.root, True, build_eval_transform(cfg))
    test_ds = _dataset_with_transform(cfg.data.root, False, build_eval_transform(cfg))

    val_set = set(val_indices)
    train_indices = [i for i in range(len(train_base)) if i not in val_set]

    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)
    return train_ds, val_ds, test_ds
