"""One logging call, four destinations. See `docs/ARCHITECTURE.md` §6.

`log_scalars` writes console, `train.log`, TensorBoard and `metrics.jsonl`.
Nothing in `src/` writes to only one of them: a number that exists in exactly
one place is a number nobody can reconstruct a claim from.

Two ordering/encoding rules here are load-bearing, both learned by crashing a
GPU run at the end of epoch 1:

- The JSONL row is written *before* the console line. Console output is the
  fragile destination; a run that dies while printing must still have its
  completed epochs on disk. The write order is jsonl -> train.log -> TensorBoard
  -> console, cheapest-to-lose last.
- Console output stays ASCII. Windows' legacy console encodes as cp1252, and a
  box-drawing separator raises `UnicodeEncodeError` mid-run.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from cifar.config import Config
from cifar.tracking.run import Run

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from cifar.engine.trainer import EpochMetrics

__all__ = ["Logger"]


class Logger:
    """Fans one logging call out to every destination a run records."""

    def __init__(self, run: Run, cfg: Config) -> None:
        """Open the run's log file and TensorBoard writer.

        Args:
            run: The run directory being written into.
            cfg: Full config; reads `cfg.log` and `cfg.train.epochs`.
        """
        self._run = run
        self._cfg = cfg
        # soft_wrap: the epoch line is one record. Wrapping it at the terminal
        # width turns every epoch into two ragged lines and makes the log
        # unreadable in a narrow console.
        self._console = Console(soft_wrap=True)
        self._metrics_path = run.dir / "metrics.jsonl"
        self._log_path = run.dir / "train.log"
        self._writer = SummaryWriter(str(run.tb_dir)) if cfg.log.tensorboard else None
        self._best_acc = 0.0
        self._best_epoch = -1

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        # JSONL, not JSON: a run killed mid-epoch still leaves every completed
        # epoch readable.
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_tb(self, tags: dict[str, float], step: int) -> None:
        """Write already-namespaced TensorBoard tags, e.g. `val/acc`."""
        if self._writer is None:
            return
        for tag, value in tags.items():
            # A skipped eval logs NaN (see `eval_every`); TensorBoard renders it
            # as a real point at 0, so leave the gap instead.
            if isinstance(value, (int, float)) and math.isfinite(value):
                self._writer.add_scalar(tag, value, step)

    def _emit(self, line: str) -> None:
        """Mirror one line to `train.log`, then to the console."""
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._console.print(line)

    def info(self, message: str) -> None:
        """Print a one-off console line. Never the only record of a number."""
        self._emit(message)

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Log a group of scalars to every destination.

        Args:
            metrics: Scalar name -> value.
            step: Epoch (or global step) the values belong to.
            prefix: Namespace, e.g. `"val"` -> `val_loss` in JSONL, `val/loss`
                in TensorBoard.
        """
        tagged = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
        self._append_jsonl({"epoch": step, **tagged, "timestamp": _utc_now()})
        self._write_tb({f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}, step)
        self._emit(f"step {step:>4} | " + " | ".join(f"{k} {v:.4g}" for k, v in tagged.items()))

    def log_epoch(self, m: EpochMetrics) -> None:
        """Log one epoch's metrics: a JSONL row first, then the other three."""
        self._append_jsonl({**asdict(m), "timestamp": _utc_now()})
        if m.val_acc > self._best_acc:
            self._best_acc, self._best_epoch = m.val_acc, m.epoch
        self._write_tb(
            {
                "train/loss": m.train_loss,
                "train/acc": m.train_acc,
                "val/loss": m.val_loss,
                "val/acc": m.val_acc,
                "train/lr": m.learning_rate,
                "train/epoch_seconds": m.epoch_seconds,
            },
            m.epoch,
        )
        best = (
            f" | best {self._best_acc:.1f}% @{self._best_epoch + 1}"
            if self._best_epoch >= 0
            else ""
        )
        self._emit(
            f"epoch {m.epoch + 1:>4}/{self._cfg.train.epochs} | "
            f"train {m.train_loss:.3f} / {m.train_acc:.1f}% | "
            f"val {m.val_loss:.3f} / {m.val_acc:.1f}% | "
            f"lr {m.learning_rate:.2e} | {m.epoch_seconds:.1f}s{best}"
        )

    def log_figure(self, name: str, fig: Figure, step: int = 0) -> None:
        """Save a figure to `figures/{name}.png` and push it to TensorBoard.

        Closes `fig` — every caller builds a figure solely to hand it here, and
        an un-closed matplotlib figure per epoch is a slow memory leak.

        Args:
            name: Basename, no extension.
            fig: The figure to record.
            step: Step to file it under in TensorBoard.
        """
        import matplotlib.pyplot as plt

        fig.savefig(self._run.figures_dir / f"{name}.png", dpi=120, bbox_inches="tight")
        if self._writer is not None:
            self._writer.add_figure(f"figures/{name}", fig, step, close=False)
        plt.close(fig)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
