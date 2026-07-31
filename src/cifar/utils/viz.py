"""Figures. See `docs/sprints/04-tracking.md` C4.3.

Every function returns a `Figure` and saves nothing; `Logger.log_figure` owns
where figures land and closes them. That keeps the plotting testable without a
run directory.

`matplotlib.use("Agg")` at import: training runs headless and under
`DataLoader` worker spawn on Windows, where an interactive backend either fails
outright or opens a window nobody closes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from cifar.config import Config  # noqa: E402
from cifar.engine.optim import build_scheduler  # noqa: E402

__all__ = [
    "plot_curves",
    "plot_confusion_matrix",
    "plot_lr_schedule",
    "plot_sample_predictions",
]


def _read_metrics(metrics_path: Path) -> list[dict]:
    lines = Path(metrics_path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def plot_curves(metrics_path: Path) -> Figure:
    """Plot loss and accuracy against epoch, train vs val, best epoch marked.

    Reads `metrics.jsonl` rather than in-memory history, so the same figure can
    be regenerated for any finished run without re-training it.

    Args:
        metrics_path: Path to a run's `metrics.jsonl`.

    Returns:
        A two-panel figure. Epochs whose val metrics are NaN (a skipped eval
        under `eval_every > 1`) are dropped from the val series rather than
        plotted at zero.

    Raises:
        ValueError: If the file has no rows -- an empty plot hides the problem.
    """
    rows = _read_metrics(metrics_path)
    if not rows:
        raise ValueError(f"{metrics_path} has no epochs to plot")

    epochs = [r["epoch"] + 1 for r in rows]
    val = [
        (r["epoch"] + 1, r["val_loss"], r["val_acc"]) for r in rows if r["val_acc"] == r["val_acc"]
    ]
    best = max(val, key=lambda v: v[2]) if val else None

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_acc.plot(epochs, [r["train_acc"] for r in rows], label="train")
    if val:
        ax_loss.plot([v[0] for v in val], [v[1] for v in val], label="val")
        ax_acc.plot([v[0] for v in val], [v[2] for v in val], label="val")
    if best is not None:
        ax_acc.axvline(best[0], color="gray", linestyle="--", linewidth=1)
        ax_acc.annotate(
            f"best {best[2]:.2f}% @{best[0]}",
            xy=(best[0], best[2]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
        )

    for ax, ylabel in ((ax_loss, "loss"), (ax_acc, "accuracy (%)")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(Path(metrics_path).parent.name)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str] | tuple[str, ...], normalize: bool = True
) -> Figure:
    """Annotated confusion-matrix heatmap, rows = true class.

    Args:
        cm: `(C, C)` integer counts, as produced by `evaluate`.
        class_names: Labels for both axes, in class-index order.
        normalize: Show row-normalized percentages instead of raw counts.

    Returns:
        A square heatmap with every cell annotated.
    """
    data = np.asarray(cm, dtype=float)
    if normalize:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data * 100.0, totals, out=np.zeros_like(data), where=totals > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, label="% of true class" if normalize else "count")

    ticks = range(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    # White text on the dark end of the colormap, black on the light end.
    threshold = data.max() / 2 if data.max() else 0.5
    fmt = "{:.1f}" if normalize else "{:.0f}"
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(data[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig


def plot_lr_schedule(cfg: Config, steps_per_epoch: int) -> Figure:
    """Plot the LR the scheduler will actually produce, before training starts.

    Built by stepping a throwaway optimizer over a single dummy parameter, so
    the curve comes from `build_scheduler` itself rather than from a second
    implementation of the same maths that could drift away from it. One second
    of compute that catches a broken schedule before an hour of training does.

    Args:
        cfg: Full config; reads `cfg.optim` and `cfg.train.epochs`.
        steps_per_epoch: `len(train_loader)`.

    Returns:
        A figure of LR against epoch (fractional), log-scaled on y.
    """
    param = torch.zeros(1, requires_grad=True)
    opt = SGD([param], lr=cfg.optim.learning_rate)
    scheduler = build_scheduler(opt, cfg, steps_per_epoch)

    lrs = []
    for _ in range(cfg.train.epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()  # no grad, so a no-op -- but it silences the step-order warning
        scheduler.step()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([i / steps_per_epoch for i in range(len(lrs))], lrs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title(f"{cfg.optim.scheduler} schedule, {cfg.optim.warmup_epochs} warmup epochs")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_sample_predictions(model: nn.Module, loader: DataLoader, n: int = 16) -> Figure:
    """Grid of predictions from the loader's first batch: green right, red wrong.

    Images are min-max stretched per-image for display rather than un-normalized
    with the dataset statistics -- the frozen signature carries no `cfg`, and for
    eyeballing which bird got called a plane the exact tone is not the point.

    Args:
        model: Model to run. Its device is taken from its parameters.
        loader: Any loader yielding `(images, labels)`.
        n: How many images to show; capped at the batch size.

    Returns:
        A square-ish grid figure, titled with predicted (and true, when wrong).
    """
    from cifar.data.dataset import CLASS_NAMES

    device = next(model.parameters()).device
    images, labels = next(iter(loader))
    images, labels = images[:n], labels[:n]

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = model(images.to(device)).argmax(1).cpu()
    finally:
        model.train(was_training)

    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    for ax, img, true, pred in zip(np.ravel(axes), images, labels, preds):
        chw = img.float()
        chw = (chw - chw.min()) / (chw.max() - chw.min() + 1e-8)
        ax.imshow(chw.permute(1, 2, 0).numpy())
        correct = int(true) == int(pred)
        ax.set_title(
            CLASS_NAMES[int(pred)]
            if correct
            else f"{CLASS_NAMES[int(pred)]}\n({CLASS_NAMES[int(true)]})",
            fontsize=7,
            color="green" if correct else "red",
        )
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.tight_layout()
    return fig
