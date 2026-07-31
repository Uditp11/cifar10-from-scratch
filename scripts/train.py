"""Train a model.

Usage:
    python scripts/train.py --config configs/simplecnn.yaml
    python scripts/train.py --config configs/resnet18.yaml --override optim.learning_rate=0.05
    python scripts/train.py --config configs/resnet18.yaml --resume runs/foo/checkpoints/last.pth

Prints the run directory as its last line of stdout so runs can be chained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cifar.config import load_config
from cifar.data.loaders import build_dataloaders
from cifar.engine.trainer import Trainer
from cifar.models import build_model
from cifar.seeding import seed_everything
from cifar.tracking.run import Run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. train.epochs=30. Repeatable.",
    )
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume from.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    overrides = list(args.override)
    if args.device:
        overrides.append(f"device={args.device}")
    cfg = load_config(args.config, overrides)

    seed_everything(cfg.repro.seed, cfg.repro.deterministic, cfg.repro.cudnn_benchmark)
    # The test loader is built and dropped here; Trainer is handed two loaders,
    # so it is structurally incapable of touching the test set.
    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg)

    run = Run(cfg)
    trainer = Trainer(model, (train_loader, val_loader), cfg, run)
    if args.resume:
        trainer.resume_from(args.resume)

    summary = trainer.fit()
    print(
        f"best val acc {summary['best_val_acc']:.2f}% @ epoch "
        f"{int(summary['best_epoch']) + 1} | device {cfg.device}"
    )
    print(run.dir)


if __name__ == "__main__":
    main()
