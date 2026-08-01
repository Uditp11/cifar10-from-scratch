"""LR range test (sprint 5, C5.1).

Usage:
    python scripts/lr_finder.py --config configs/resnet18.yaml
    python scripts/lr_finder.py --config configs/resnet18.yaml --steps 400

Trains for a few hundred steps while ramping the learning rate exponentially
from `--min-lr` to `--max-lr`, and plots smoothed loss against LR on a log x
axis. The point is evidence, not ceremony: it turns "0.1 because everyone uses
0.1" into "0.1 because here is the curve".

Weights are thrown away -- this writes no run directory and no checkpoint. It
prints the steepest-descent bracket and whether the config's own
`optim.learning_rate` falls inside it, then the figure path as its last line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cifar.config import Config, load_config
from cifar.data.loaders import build_dataloaders
from cifar.engine.criterion import build_criterion
from cifar.engine.optim import build_optimizer
from cifar.models import build_model
from cifar.seeding import seed_everything

# Imported through viz so the Agg backend is selected in exactly one place.
from cifar.utils.viz import plt

# Stop early once the run has clearly blown up. Beyond this the curve carries no
# information and its scale ruins the plot.
_DIVERGE_FACTOR = 4.0


def sweep_lr(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Config,
    min_lr: float,
    max_lr: float,
    steps: int,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Train `steps` batches with the LR ramped exponentially, recording loss.

    The optimizer comes from `build_optimizer` so the param-group split and
    momentum under test are the ones training will actually use; only the LR is
    overwritten each step.

    Args:
        model: Model on `cfg.device`, left in an arbitrary trained state after.
        loader: Training loader. Cycled if it is shorter than `steps`.
        cfg: Full config; reads `cfg.optim`, `cfg.train.amp` and `cfg.device`.
        min_lr: LR at step 0.
        max_lr: LR at the final step.
        steps: Number of optimizer steps to take.
        smooth: EMA coefficient for the loss trace, bias-corrected so the first
            points are not dragged toward zero. Raw per-batch loss at batch 128
            is far too noisy to differentiate, but the EMA's ~1/(1-smooth)-step
            window also *lags* the ramp and shifts the curve to the right; at
            0.98 that is ~50 steps, which is a third of a decade at 200 steps
            per decade. Worth varying before believing any LR read off the plot.

    Returns:
        `(lrs, smoothed_losses)`, truncated at the point the loss diverges.
    """
    device = torch.device(cfg.device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")

    schedule = np.logspace(np.log10(min_lr), np.log10(max_lr), steps)
    lrs: list[float] = []
    losses: list[float] = []
    avg, best = 0.0, np.inf

    model.train()
    batches = iter(loader)
    for i, lr in enumerate(schedule):
        try:
            x, y = next(batches)
        except StopIteration:  # fewer batches than steps -- start another pass
            batches = iter(loader)
            x, y = next(batches)

        for group in optimizer.param_groups:
            group["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=cfg.train.amp):
            loss = criterion(model(x), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        raw = loss.item()
        avg = smooth * avg + (1 - smooth) * raw
        smoothed = avg / (1 - smooth ** (i + 1))
        best = min(best, smoothed)
        if not np.isfinite(smoothed) or smoothed > _DIVERGE_FACTOR * best:
            break
        lrs.append(float(lr))
        losses.append(smoothed)

    return np.asarray(lrs), np.asarray(losses)


def steepest_region(
    lrs: np.ndarray, losses: np.ndarray, fraction: float = 0.5
) -> tuple[float, float, float]:
    """Find the LR band over which loss falls fastest.

    Steepness is d(loss)/d(log10 LR), which is the slope the eye reads off the
    log-x plot. The band is the contiguous run around the steepest point where
    the slope stays at least `fraction` as steep -- a single argmin on a noisy
    curve is a coin flip, an interval is a claim.

    Only the curve *up to the loss minimum* is considered. That is the descent
    region by definition, and searching past it is not a refinement but a bug:
    the post-divergence tail is violently noisy, and one downward jitter between
    two spikes there outscores the entire genuine descent. Measured directly --
    searching the whole curve put the "steepest" point at lr 4.5 and 6.4 for two
    smoothing settings whose loss minimum was 1.1, i.e. deep inside divergence.

    Slope is measured across a fixed window a quarter-decade wide rather than
    between adjacent points. Adjacent-point `np.gradient` on this curve is pure
    noise amplification -- it made the half-peak band collapse to a single
    sample (0.464..0.471) because one point happened to sit twice as steep as
    its neighbours. A window turns the derivative back into the loss-drop-per-
    decade a reader takes off the plot.

    Args:
        lrs: Learning rates, ascending.
        losses: Smoothed loss at each LR.
        fraction: How much of the peak steepness a point must retain to stay in
            the band.

    Returns:
        `(low_lr, high_lr, steepest_lr)`.

    Raises:
        ValueError: If the descent is too short to fit a window across -- a
            curve that short cannot be differentiated, and guessing a bracket
            from it would be worse than saying nothing.
    """
    stop = int(np.argmin(losses)) + 1
    lrs, losses = lrs[:stop], losses[:stop]

    decades = np.log10(lrs[-1] / lrs[0]) if len(lrs) > 1 else 0.0
    half_window = max(1, round(0.125 * len(lrs) / decades)) if decades > 0 else 1
    if len(lrs) < 2 * half_window + 3:
        raise ValueError(f"LR range test produced only {len(lrs)} descending points; need more")

    log_lr = np.log10(lrs)
    lo_i, hi_i = np.arange(len(lrs) - 2 * half_window), np.arange(2 * half_window, len(lrs))
    slope = (losses[hi_i] - losses[lo_i]) / (log_lr[hi_i] - log_lr[lo_i])
    centres = lrs[half_window : len(lrs) - half_window]

    peak = int(np.argmin(slope))
    threshold = fraction * slope[peak]

    low = peak
    while low > 0 and slope[low - 1] <= threshold:
        low -= 1
    high = peak
    while high < len(slope) - 1 and slope[high + 1] <= threshold:
        high += 1
    return float(centres[low]), float(centres[high]), float(centres[peak])


def plot_lr_range(
    lrs: np.ndarray, losses: np.ndarray, band: tuple[float, float, float], configured_lr: float
) -> plt.Figure:
    """Plot loss against LR with the steepest-descent band and the config's LR marked.

    Args:
        lrs: Learning rates, ascending.
        losses: Smoothed loss at each LR.
        band: `(low, high, steepest)` from `steepest_region`.
        configured_lr: `cfg.optim.learning_rate`, drawn as a vertical line so
            the figure answers the question it was run to answer.

    Returns:
        A single-axis figure, log-scaled on x.
    """
    low, high, steepest = band
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses, linewidth=1.5)
    ax.axvspan(low, high, color="tab:green", alpha=0.15, label=f"steepest {low:.2g}-{high:.2g}")
    ax.axvline(steepest, color="tab:green", linestyle=":", linewidth=1)
    ax.axvline(
        configured_lr, color="tab:red", linestyle="--", linewidth=1, label=f"lr {configured_lr:g}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("configs/resnet18.yaml"), type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. data.batch_size=256. Repeatable.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-lr", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1000)
    # 0.9, not the customary 0.98: at 1000 steps the two agree on the steepest
    # point (0.33 vs 0.36) but 0.98's extra lag smears the half-peak band from
    # 0.31-0.39 out to 0.26-1.16. Measured, not assumed -- vary it and see.
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="Loss EMA coefficient. Lower lags the ramp less but reads noisier.",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/figures/lr_finder.png"))
    args = parser.parse_args()

    if not 0 < args.min_lr < args.max_lr:
        raise SystemExit(f"need 0 < min-lr ({args.min_lr}) < max-lr ({args.max_lr})")
    if not 0 <= args.smooth < 1:
        raise SystemExit(f"need 0 <= smooth ({args.smooth}) < 1")

    cfg = load_config(args.config, args.override)
    seed_everything(cfg.repro.seed, cfg.repro.deterministic, cfg.repro.cudnn_benchmark)
    train_loader, _, _ = build_dataloaders(cfg)
    model = build_model(cfg)

    lrs, losses = sweep_lr(
        model, train_loader, cfg, args.min_lr, args.max_lr, args.steps, args.smooth
    )
    low, high, steepest = steepest_region(lrs, losses)
    min_loss_lr = float(lrs[int(np.argmin(losses))])
    configured = cfg.optim.learning_rate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_lr_range(lrs, losses, (low, high, steepest), configured)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(
        f"steps used      {len(lrs)}/{args.steps}"
        + ("" if len(lrs) == args.steps else " (diverged)")
    )
    print(f"steepest slope  at lr {steepest:.3g}")
    print(f"steepest band   {low:.3g} .. {high:.3g}  (>= half peak steepness)")
    # The two anchors the plot is conventionally read from: the loss minimum is
    # an upper bound (past it, training is unstable), and the usual pick is an
    # order of magnitude below it.
    print(f"loss minimum    at lr {min_loss_lr:.3g}  -> conventional pick {min_loss_lr / 10:.3g}")
    brackets = low <= configured <= high
    print(f"configured lr   {configured:g} -- {'INSIDE' if brackets else 'OUTSIDE'} the band")
    print(args.out)


if __name__ == "__main__":
    main()
