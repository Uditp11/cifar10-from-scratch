"""Visual sanity check for the data pipeline. See sprints 01 (C1.6) and 06 (C6.6).

Writes two figures:

* `docs/figures/data_samples.png` — an 8x8 grid of augmented training images
  (denormalized) plus a per-class count table.
* `docs/figures/augmentations.png` — one row per augmentation, the same eight
  images throughout, so each stage can be compared against the row above it.

Augmentation bugs are usually obvious to the eye and invisible in the metrics.
CutMix in particular: a box that clips at the border and is *not* recompensated
in `lam` produces a perfectly smooth loss curve and a wrong gradient. In an
image grid you can see whether the pasted rectangle is contiguous and whether
the printed `lam` matches how much of the image it covers.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar.config import Config, load_config
from cifar.data.batch_aug import apply_batch_aug
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
_AUG_OUT_PATH = Path("docs/figures/augmentations.png")

# Transform-level panels, each expressed as a delta on the sprint-5 recipe so
# the figure shows what a config change actually does rather than a hand-rolled
# reconstruction of it.
_OFF = {"random_crop_padding": 0, "horizontal_flip_p": 0.0}
_TRANSFORM_PANELS: list[tuple[str, dict]] = [
    ("no aug", _OFF),
    ("crop4 + flip", {}),
    ("+ cutout(1,16)", {"cutout_holes": 1, "cutout_length": 16}),
    ("cutout(1,20)", {**_OFF, "cutout_holes": 1, "cutout_length": 20}),
    ("randaugment(2,9)", {**_OFF, "randaugment_n": 2, "randaugment_m": 9}),
]
# Batch-level panels run on the collated tensor, after the transform pipeline.
# `lam` is drawn once per batch, so each row is a single draw. a=0.2 is the
# recipe value and Beta(0.2, 0.2) is strongly bimodal, so that row usually shows
# a near-pure permutation rather than a blend -- which is what mixup at this
# alpha actually does most of the time. a=1.0 (uniform lam) is included so the
# blend itself is visible at least once.
_BATCH_PANELS: list[tuple[str, dict]] = [
    ("mixup a=0.2", {"mixup_alpha": 0.2, "batch_aug_prob": 1.0}),
    ("mixup a=1.0", {"mixup_alpha": 1.0, "batch_aug_prob": 1.0}),
    ("cutmix a=1.0", {"cutmix_alpha": 1.0, "batch_aug_prob": 1.0}),
]


def _with_aug(cfg: Config, **overrides) -> Config:
    """Copy `cfg` with `cfg.aug` fields replaced. Skips `_validate` by design:
    these panels are illustrative and need combinations a real run would not."""
    return dataclasses.replace(cfg, aug=dataclasses.replace(cfg.aug, **overrides))


def _denormalize(img: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = (img * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def write_sample_grid(cfg: Config, out_path: Path = _OUT_PATH) -> Path:
    """Write the 8x8 augmented-sample grid and the per-class count table.

    Args:
        cfg: Full config; the train transform is used as configured.
        out_path: PNG to write; parent directories are created.

    Returns:
        The path written.
    """
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _row(axes, images: torch.Tensor, label: str, cfg: Config) -> None:
    """Draw one labelled row of denormalized images."""
    for ax, img in zip(axes, images):
        ax.imshow(_denormalize(img, cfg.data.mean, cfg.data.std))
        # Ticks off rather than axis off: `axis("off")` would take the row
        # label with it, and the label is the whole point of the figure.
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)


def write_augmentation_grid(cfg: Config, out_path: Path = _AUG_OUT_PATH) -> Path:
    """Write one row per augmentation, over a fixed set of eight images.

    Every row is produced by the real pipeline under a modified `cfg.aug`, so a
    row that looks wrong is a bug in the transform, not in this script.

    Args:
        cfg: Full config; each panel overrides `cfg.aug` and rebuilds.
        out_path: PNG to write; parent directories are created.

    Returns:
        The path written.
    """
    panels = _TRANSFORM_PANELS + _BATCH_PANELS
    fig, axes = plt.subplots(
        len(panels), _GRID, figsize=(_GRID * 1.3, len(panels) * 1.5), squeeze=False
    )

    for row, (label, overrides) in enumerate(_TRANSFORM_PANELS):
        # Re-seeded per row so crop/flip draw the same way in each: a difference
        # between two rows is then the augmentation under test, not the RNG.
        seed_everything(cfg.repro.seed)
        train_ds, _, _ = build_datasets(_with_aug(cfg, **overrides))
        images = torch.stack([train_ds[i][0] for i in range(_GRID)])
        _row(axes[row], images, label, cfg)

    # The batch panels share one unmixed source batch, so mixup's blend and
    # cutmix's paste are visibly against the same originals as row 2.
    seed_everything(cfg.repro.seed)
    train_ds, _, _ = build_datasets(cfg)
    source = torch.stack([train_ds[i][0] for i in range(_GRID)])
    targets = torch.tensor([train_ds[i][1] for i in range(_GRID)])

    for offset, (label, overrides) in enumerate(_BATCH_PANELS):
        batch = apply_batch_aug(source, targets, _with_aug(cfg, **overrides))
        pasted = (batch.inputs != source).float().mean().item()
        # lam is the *reported* weight and `pasted` the fraction of pixels that
        # actually changed; for cutmix these must agree, or the clipped-box bug
        # is back and the loss is being weighted for a box nobody pasted.
        _row(
            axes[len(_TRANSFORM_PANELS) + offset],
            batch.inputs,
            f"{label}\nlam={batch.lam:.3f}\nchanged={pasted:.3f}",
            cfg,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    cfg = load_config("configs/resnet18.yaml")
    seed_everything(cfg.repro.seed)
    for path in (write_sample_grid(cfg), write_augmentation_grid(cfg)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
