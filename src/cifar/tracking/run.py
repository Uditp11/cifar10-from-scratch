"""The run directory. See `docs/ARCHITECTURE.md` §6.

Sprint 3 scope: create the directory, save the resolved config so a run can be
re-loaded and evaluated, and keep a manifest whose `status` is honest about how
the run ended. Sprint 4 fills in the git/env capture and the figures directory.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cifar.config import Config, config_hash, config_to_dict

__all__ = ["Run"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        base = cfg.log.run_name or f"{cfg.model.name}_{datetime.now():%Y%m%d_%H%M%S}"
        root = Path(cfg.log.run_root)

        name, suffix = base, 1
        while (root / name).exists():
            suffix += 1
            name = f"{base}_{suffix}"

        self.name = name
        self._dir = root / name
        self.checkpoint_dir.mkdir(parents=True)
        (self._dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir / "checkpoints"

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
        self.write_manifest(
            {
                "finished_at": _utc_now(),
                "total_seconds": round(time.perf_counter() - self._started, 1),
                "status": status,
                "summary": summary,
            }
        )
