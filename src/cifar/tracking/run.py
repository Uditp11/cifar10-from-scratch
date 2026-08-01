"""The run directory. See `docs/ARCHITECTURE.md` §6.

A run directory is meant to be self-describing months later: the manifest says
what code, what hardware and what config produced the numbers, and `config.yaml`
re-loads to a `Config` that hashes identically.

Three fields sit in `runtime` rather than being left implicit in `config`: the
*resolved* device (ADR-005 -- `"auto"` never reaches the manifest), the
`channels_last` memory format (ADR-007 makes it a first-class experimental
variable, not a detail), and the measured steady-state seconds per epoch.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from cifar.config import Config, config_hash, config_to_dict

__all__ = ["Run"]

_REPO = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_info() -> dict[str, Any]:
    """Describe the working tree the run started from.

    Degrades to nulls rather than raising: git may be absent, or the project may
    have been copied out of its repository. A run without provenance is worse
    than a run with none, but it is not a reason to refuse to train.
    """

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    try:
        return {
            "sha": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "branch": None, "dirty": None}


def _gpu_temp_c() -> int | None:
    """Current GPU temperature in Celsius, or None if it cannot be read.

    Recorded at run start and end because `sweep.py` runs jobs back to back:
    the ninth run of an ablation starts on a card the first eight have been
    heating, so run *order* silently confounds every wall-clock number in the
    sweep. One `nvidia-smi` call per run makes that visible instead of
    mysterious. Never raises -- no GPU, no driver, or no `nvidia-smi` on PATH
    are all ordinary, and none of them is a reason to fail a training run.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _env_info() -> dict[str, Any]:
    """Versions and hardware, so a timing number can be read years later."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


class Run:
    """Owns one run directory: `runs/{run_name}/`."""

    def __init__(self, cfg: Config) -> None:
        """Create the run directory and write the resolved config into it.

        Args:
            cfg: Full config; reads `cfg.log.run_root` and `cfg.log.run_name`.
                A `None` run_name auto-names as `{model}_{timestamp}`, with
                `_2`, `_3`, ... appended on collision.
        """
        self._cfg = cfg
        self._started = time.perf_counter()
        self._started_at = _utc_now()
        self._gpu_temp_start = _gpu_temp_c()
        self._gpu_temp_end: int | None = None
        base = cfg.log.run_name or f"{cfg.model.name}_{datetime.now():%Y%m%d_%H%M%S}"
        root = Path(cfg.log.run_root)

        name, suffix = base, 1
        while (root / name).exists():
            suffix += 1
            name = f"{base}_{suffix}"

        self.name = name
        self._dir = root / name
        self.checkpoint_dir.mkdir(parents=True)
        self.figures_dir.mkdir()
        self.tb_dir.mkdir()
        (self._dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        return self._dir / "figures"

    @property
    def tb_dir(self) -> Path:
        return self._dir / "tb"

    def _seconds_per_epoch(self) -> float | None:
        """Median steady-state epoch time from `metrics.jsonl`, or None.

        Computed over the **back half** of the run, not over everything-but-the-
        first-epoch. Epoch 1 is the obvious outlier -- 60-75 s of DataLoader
        worker spawn on Windows -- but it is not the only one: measured in
        sprint 4, SimpleCNN takes until roughly epoch 7 to settle (8.4, 7.0,
        5.2, 4.7, 4.3, ... 4.4) and ResNet-18 until epoch 3. Dropping only the
        first epoch reported SimpleCNN at 6.5 s/epoch against a true 4.4.

        Median, not mean, so one thermally-throttled epoch does not move it.

        Returns:
            Seconds per steady-state epoch, or None before any epoch is logged.
            **A run shorter than ~10 epochs has no steady state to find and the
            number it reports is ramp-inflated** -- read it against
            `summary.epochs_completed` before comparing it with anything.
        """
        path = self._dir / "metrics.jsonl"
        if not path.exists():
            return None
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        times = [row["epoch_seconds"] for row in rows if "epoch_seconds" in row]
        if not times:
            return None
        return round(statistics.median(times[len(times) // 2 :]), 2)

    def write_manifest(self, extra: dict | None = None) -> None:
        """Write `run_manifest.json` with `status: "running"`.

        Called before the first epoch so that a run killed at any point after
        this leaves a manifest behind, rather than an unexplained directory.

        Args:
            extra: Merged into the manifest; how `finalize` adds its end fields.
        """
        manifest = {
            "run_name": self.name,
            "config_hash": config_hash(self._cfg),
            "config": config_to_dict(self._cfg),
            "git": _git_info(),
            "env": _env_info(),
            "runtime": {
                "device": self._cfg.device,
                "channels_last": self._cfg.train.channels_last,
                "seconds_per_epoch": self._seconds_per_epoch(),
                "gpu_temp_start_c": self._gpu_temp_start,
                "gpu_temp_end_c": self._gpu_temp_end,
            },
            "started_at": self._started_at,
            "status": "running",
        }
        manifest.update(extra or {})
        (self._dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def finalize(self, summary: dict[str, float], status: str = "completed") -> None:
        """Stamp the manifest with end time, total seconds, status, and summary.

        Args:
            summary: Headline numbers, e.g. best val accuracy and best epoch.
            status: `completed` | `failed` | `interrupted` (ADR-006).
        """
        self._gpu_temp_end = _gpu_temp_c()
        self.write_manifest(
            {
                "finished_at": _utc_now(),
                "total_seconds": round(time.perf_counter() - self._started, 1),
                "status": status,
                "summary": summary,
            }
        )
