"""Console + `metrics.jsonl` logging. See `docs/ARCHITECTURE.md` §6.

Sprint 3 scope: two of the three destinations. TensorBoard and figure logging
land in sprint 4 behind these same call sites, so nothing in `engine/` changes
when they do.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console

from cifar.config import Config
from cifar.tracking.run import Run

if TYPE_CHECKING:
    from cifar.engine.trainer import EpochMetrics

__all__ = ["Logger"]


class Logger:
    """Fans one logging call out to every destination a run records."""

    def __init__(self, run: Run, cfg: Config) -> None:
        self._run = run
        self._cfg = cfg
        self._console = Console()
        self._metrics_path = run.dir / "metrics.jsonl"

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        # JSONL, not JSON: a run killed mid-epoch still leaves every completed
        # epoch readable.
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def info(self, message: str) -> None:
        """Print a one-off console line. Never the only record of a number."""
        self._console.print(message)

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Log a group of scalars to console and `metrics.jsonl`.

        Args:
            metrics: Scalar name -> value.
            step: Epoch (or global step) the values belong to.
            prefix: Namespace, e.g. `"val"` -> `val_loss`, `val_acc`.
        """
        tagged = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
        self._console.print(
            f"step {step:>4} │ " + " │ ".join(f"{k} {v:.4g}" for k, v in tagged.items())
        )
        self._append_jsonl({"epoch": step, **tagged, "timestamp": _utc_now()})

    def log_epoch(self, m: EpochMetrics) -> None:
        """Log one epoch's metrics: a scannable console line plus a JSONL row."""
        self._console.print(
            f"epoch {m.epoch + 1:>4}/{self._cfg.train.epochs} │ "
            f"train {m.train_loss:.3f} / {m.train_acc:.1f}% │ "
            f"val {m.val_loss:.3f} / {m.val_acc:.1f}% │ "
            f"lr {m.learning_rate:.2e} │ {m.epoch_seconds:.1f}s"
        )
        self._append_jsonl({**asdict(m), "timestamp": _utc_now()})

    def close(self) -> None:
        """Release logging resources. A no-op until sprint 4 adds TensorBoard."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
