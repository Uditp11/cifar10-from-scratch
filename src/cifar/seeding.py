"""Global seeding for reproducibility. See `docs/ARCHITECTURE.md` §2."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "worker_init_fn",
    "make_generator",
    "capture_rng_state",
    "restore_rng_state",
]


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


def capture_rng_state() -> dict:
    """Snapshot every RNG this project draws from.

    Checkpointed so that a resumed run continues the same trajectory an
    uninterrupted one would have taken -- without this, resume re-draws
    augmentations and shuffles from a fresh stream.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from `capture_rng_state`.

    CUDA state is skipped when the checkpoint was written on a machine with a
    different GPU count, since `set_rng_state_all` requires one state per
    device -- a cross-machine resume loses GPU-side stream position but stays
    otherwise faithful.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    cuda_state = state.get("torch_cuda")
    if cuda_state and torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_state)
