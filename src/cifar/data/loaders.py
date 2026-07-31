"""DataLoader construction. See ARCHITECTURE.md section 3.

On Windows, `num_workers > 0` requires callers to build these loaders behind
an `if __name__ == "__main__":` guard (spawn-based multiprocessing re-imports
the launching module). See scripts/inspect_data.py for the pattern.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from cifar.config import Config
from cifar.data.dataset import build_datasets
from cifar.seeding import make_generator, worker_init_fn

__all__ = ["build_dataloaders"]


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders.

    Train shuffles with a seeded generator and `drop_last=True`. Val/test
    never shuffle and never drop the last batch. `persistent_workers` is
    forced off when `num_workers == 0`, since torch requires that.

    Args:
        cfg: Full config; reads `cfg.data` and `cfg.repro.seed`.

    Returns:
        (train_loader, val_loader, test_loader).
    """
    train_ds, val_ds, test_ds = build_datasets(cfg)
    persistent = cfg.data.persistent_workers and cfg.data.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init_fn if cfg.data.num_workers > 0 else None,
        generator=make_generator(cfg.repro.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader
