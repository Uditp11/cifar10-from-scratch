"""Visual sanity check for the data pipeline. See docs/sprints/01-data.md C1.6.

Saves an 8x8 grid of augmented training images (denormalized) plus a
per-class count table to docs/figures/data_samples.png. Augmentation bugs
are usually obvious to the eye and invisible in the metrics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar.config import load_config
from cifar.data.dataset import build_datasets
from cifar.seeding import seed_everything

_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
_GRID = 8
_OUT_PATH = Path("docs/figures/data_samples.png")


def _denormalize(img: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = (img * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main() -> None:
    cfg = load_config("configs/resnet18.yaml")
    seed_everything(cfg.repro.seed)
    train_ds, val_ds, _ = build_datasets(cfg)

    fig = plt.figure(figsize=(_GRID * 1.6, _GRID * 1.6 + 2.5))
    gs = gridspec.GridSpec(_GRID + 3, _GRID, figure=fig)

    for i in range(_GRID * _GRID):
        img, label = train_ds[i]
        ax = fig.add_subplot(gs[i // _GRID, i % _GRID])
        ax.imshow(_denormalize(img, cfg.data.mean, cfg.data.std))
        ax.set_title(_CLASS_NAMES[label], fontsize=7)
        ax.axis("off")

    train_counts = Counter(train_ds.dataset.targets[i] for i in train_ds.indices)
    val_counts = Counter(val_ds.dataset.targets[i] for i in val_ds.indices)
    table_ax = fig.add_subplot(gs[_GRID:, :])
    table_ax.axis("off")
    rows = [[_CLASS_NAMES[c], train_counts[c], val_counts[c]] for c in range(10)]
    table_ax.table(cellText=rows, colLabels=["class", "train", "val"], loc="center")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(_OUT_PATH, dpi=150)
    print(f"wrote {_OUT_PATH}")


if __name__ == "__main__":
    main()
