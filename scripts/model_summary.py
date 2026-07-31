"""Print a model's per-stage shapes and parameter counts.

Usage: python scripts/model_summary.py --model resnet18_cifar
"""

from __future__ import annotations

import argparse

from cifar.config import load_config
from cifar.models import build_model, list_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list_models())
    args = parser.parse_args()

    cfg = load_config("configs/base.yaml", [f"model.name={args.model}", "device=cpu"])
    model = build_model(cfg)
    print(model.summary())


if __name__ == "__main__":
    main()
