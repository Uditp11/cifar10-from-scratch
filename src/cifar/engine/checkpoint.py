"""Checkpoint save/load. See `docs/ARCHITECTURE.md` §5.

A checkpoint holds everything needed to make a resumed run indistinguishable
from an uninterrupted one: model, optimizer, scheduler and scaler state, plus
the RNG state of every stream (`cifar.seeding.capture_rng_state`).

Writes go to `<name>.tmp` and are then `os.replace`d into place, which is
atomic on both POSIX and Windows -- a crash mid-save can therefore never leave
a truncated file where the only good checkpoint used to be.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from cifar.config import Config, config_hash

__all__ = ["save_checkpoint", "load_checkpoint"]


def _state_dict(obj: Any) -> Any:
    return None if obj is None else obj.state_dict()


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    ema: Any,
    epoch: int,
    best_metric: float,
    cfg: Config,
    rng_state: dict,
) -> None:
    """Write a full training checkpoint atomically.

    Args:
        path: Destination file. Parent directories are created if missing.
        model: Model whose weights are saved.
        optimizer: Optimizer; momentum buffers matter for a faithful resume.
        scheduler: LR scheduler, or None.
        scaler: AMP `GradScaler`, or None.
        ema: `ModelEMA`, or None until sprint 7.
        epoch: Index of the epoch that just completed.
        best_metric: Best validation accuracy seen so far.
        cfg: Full config; only its hash is stored, for the resume mismatch warning.
        rng_state: From `cifar.seeding.capture_rng_state`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": _state_dict(optimizer),
        "scheduler": _state_dict(scheduler),
        "scaler": _state_dict(scaler),
        "ema": _state_dict(ema),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg_hash": config_hash(cfg),
        "rng_state": rng_state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema: Any = None,
    map_location: str = "cpu",
) -> dict:
    """Restore a checkpoint into the objects passed in.

    Args:
        path: Checkpoint file written by `save_checkpoint`.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional `GradScaler` to restore.
        ema: Optional `ModelEMA` to restore.
        map_location: Passed to `torch.load`.

    Returns:
        `{"epoch", "best_metric", "cfg_hash", "rng_state"}`. Callers compare
        `cfg_hash` against their own config -- `Trainer.resume_from` warns on a
        mismatch, since resuming into a changed config must never be silent.
    """
    # weights_only=False: the payload carries RNG state (numpy arrays, python
    # tuples), not just tensors. These are our own files, written above.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model"])
    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if obj is not None and ckpt.get(key) is not None:
            obj.load_state_dict(ckpt[key])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])

    return {
        "epoch": ckpt["epoch"],
        "best_metric": ckpt["best_metric"],
        "cfg_hash": ckpt["cfg_hash"],
        "rng_state": ckpt["rng_state"],
    }
