"""Run every config in a directory, one training each, and summarize.

Usage:
    python scripts/sweep.py --dir configs/ablations/
    python scripts/sweep.py --dir configs/ablations/ --dry-run

A failure does not stop the sweep: the point is to start a set of ablations and
come back later, and losing the remaining six runs because the second one OOM'd
is exactly the outcome that wastes an evening. Failures are reported in the
summary table with their exit code.

Child stdout is streamed as it arrives, so a long sweep is still watchable; the
last line of each child is its run directory (every script guarantees that),
which is how the summary finds the manifest to report from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_TRAIN = Path(__file__).resolve().parent / "train.py"


def run_one(config: Path, overrides: list[str]) -> tuple[int, Path | None]:
    """Train one config in a subprocess, streaming its output.

    A subprocess, not an in-process call: a run that segfaults in a CUDA kernel
    or exhausts VRAM takes its own process down and the sweep continues.

    Args:
        config: YAML config to train.
        overrides: Dotted overrides passed through to `train.py`.

    Returns:
        `(exit_code, run_dir)`. `run_dir` is None when the child printed nothing
        usable -- typically because it died before creating one.
    """
    cmd = [sys.executable, str(_TRAIN), "--config", str(config)]
    for override in overrides:
        cmd += ["--override", override]

    last_line = ""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if line.strip():
            last_line = line.strip()
    code = proc.wait()

    candidate = Path(last_line) if last_line else None
    return code, candidate if candidate and candidate.is_dir() else None


def summarize(results: list[tuple[Path, int, Path | None]]) -> str:
    """Render the end-of-sweep table.

    Args:
        results: `(config, exit_code, run_dir)` per attempted config.

    Returns:
        A markdown table, one row per config.
    """
    rows = []
    for config, code, run_dir in results:
        best, status = "-", f"FAILED (exit {code})" if code else "ok"
        if run_dir is not None:
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            status = manifest.get("status", status)
            if "best_val_acc" in manifest.get("summary", {}):
                best = f"{manifest['summary']['best_val_acc']:.2f}"
        rows.append(
            f"| {config.name} | {status} | {best} | {run_dir.as_posix() if run_dir else '-'} |"
        )
    return "\n".join(["| Config | Status | Best val acc | Run |", "|---|---|---|---|", *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="Directory of YAML configs.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Applied to every run, e.g. train.epochs=5. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would run and stop.")
    args = parser.parse_args()

    configs = sorted(p for p in args.dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No *.yaml configs in {args.dir}")

    if args.dry_run:
        for config in configs:
            print(
                f"would train {config}" + (f" ({' '.join(args.override)})" if args.override else "")
            )
        return

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {config}")
        code, run_dir = run_one(config, args.override)
        results.append((config, code, run_dir))

    print("\n" + summarize(results))
    if any(code for _, code, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
