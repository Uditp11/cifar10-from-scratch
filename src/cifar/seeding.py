"""Global seeding for reproducibility. See `docs/ARCHITECTURE.md` §2."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "worker_init_fn", "make_generator"]


def seed_everything(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    """Seed random, numpy, torch, torch.cuda; set cudnn flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding. Derives from the base seed + worker_id."""
    base_seed = torch.initial_seed() % 2**32
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
