"""Evaluate a finished run on the CIFAR-10 test set.

Usage:
    python scripts/eval.py --run runs/simple_cnn_20260801_120000 --checkpoint best

This is the only place in the codebase that touches the test set. Nothing here
feeds back into training: no threshold, no epoch choice, no hyperparameter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cifar.config import load_config
from cifar.data.loaders import build_dataloaders
from cifar.engine.checkpoint import load_checkpoint
from cifar.engine.evaluator import evaluate
from cifar.models import build_model
from cifar.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="A run directory.")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--tta", action="store_true", help="Sprint 7.")
    parser.add_argument("--ema", action="store_true", help="Sprint 7.")
    args = parser.parse_args()

    if args.ema:
        raise NotImplementedError("EMA weights land in sprint 7")

    cfg = load_config(args.run / "config.yaml")
    seed_everything(cfg.repro.seed, cfg.repro.deterministic, cfg.repro.cudnn_benchmark)

    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    ckpt_path = args.run / "checkpoints" / f"{args.checkpoint}.pth"
    meta = load_checkpoint(ckpt_path, model=model, map_location=cfg.device)

    result = evaluate(model, test_loader, cfg, tta=args.tta)

    print(f"checkpoint  {ckpt_path} (epoch {meta['epoch'] + 1}, val {meta['best_metric']:.2f}%)")
    print(f"test loss   {result.loss:.4f}")
    print(f"test top-1  {result.top1:.2f}%   top-5 {result.top5:.2f}%   n={result.num_samples}")
    print("\nper-class accuracy")
    for name, acc in result.per_class_acc.items():
        print(f"  {name:<12} {acc:6.2f}%")
    print("\nconfusion matrix (rows = true)")
    for name, row in zip(result.per_class_acc, result.confusion_matrix):
        print(f"  {name:<12} " + " ".join(f"{v:4d}" for v in row))
    print(args.run)


if __name__ == "__main__":
    main()
