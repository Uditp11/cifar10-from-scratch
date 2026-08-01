"""Compare finished runs: a markdown table plus an overlaid val-accuracy curve.

Usage:
    python scripts/make_report.py --runs runs/a runs/b --section sprint-5
    python scripts/make_report.py --all --section all-runs

Only hyperparameters that *differ* between the compared runs get a column. A
table with forty identical columns is a table nobody reads.

`EXPERIMENTS.md` is two documents in one file: generated tables and hand-written
findings. Each `--section` owns a sentinel-delimited region

    <!-- BEGIN GENERATED: sprint-5 -->
    ...rewritten wholesale on every run...
    <!-- END GENERATED: sprint-5 -->

and this script rewrites *only* between those markers, appending a fresh region
if the section is new. Everything outside every region -- the findings, the
hypotheses, the hardware table -- is untouchable by construction (ADR-010).

Prints the report path as its last line of stdout so runs can be chained.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Imported through viz so the Agg backend is selected in exactly one place.
from cifar.utils.viz import plt

# Config sections that describe bookkeeping, not the experiment.
_IGNORED_TOP_LEVEL = {"log", "notes"}


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, sub in value.items():
        flat.update(_flatten(f"{prefix}.{key}" if prefix else key, sub))
    return flat


def _pid_is_live(pid: int) -> bool:
    """Is a process with this PID currently running on this machine?

    ponytail: PID reuse is not detected -- a recycled PID reads as live and the
    run stays "running". Recording the process creation time alongside the PID
    would close that, and needs `psutil`; not worth a dependency for a reaper
    whose failure mode is being too cautious.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        # 0x1000 = PROCESS_QUERY_LIMITED_INFORMATION: succeeds for processes
        # this user cannot open for reading, which PROCESS_QUERY_INFORMATION
        # would fail on and be misread as "dead".
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        # ERROR_ACCESS_DENIED means it exists but belongs to someone else.
        return ctypes.windll.kernel32.GetLastError() == 5
    try:
        os.kill(pid, 0)  # signal 0 tests existence without delivering anything
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reaped_status(manifest: dict[str, Any]) -> str:
    """The manifest's status, with a stranded `running` resolved to `killed`.

    ADR-006 argued a run never sits at `running` once its process is gone,
    because `Trainer.fit` labels its own exit in a `finally`. That holds for
    every exit Python can observe, and not for the ones it cannot: a
    TerminateProcess, a console teardown, a power event. Sprint 5 lost three
    ablations to exactly this and left a manifest reading `running` fourteen
    hours later. The writer cannot fix it -- it is already gone -- so the reader
    does, by asking whether the recorded PID is still live.

    Conservative by construction: a run whose manifest predates the `process`
    block, or that was produced on another machine, is left alone. Reporting a
    live run as killed is a worse error than reporting a dead one as running.

    Args:
        manifest: A parsed `run_manifest.json`.

    Returns:
        `"killed"` for a stranded run, otherwise the recorded status.
    """
    if manifest.get("status") != "running":
        return manifest.get("status", "-")
    process = manifest.get("process") or {}
    pid, host = process.get("pid"), process.get("host")
    if not isinstance(pid, int) or host != platform.node():
        return "running"
    return "running" if _pid_is_live(pid) else "killed"


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read one run's manifest and per-epoch metrics.

    Args:
        run_dir: A directory written by `Run`.

    Returns:
        `{"dir", "manifest", "params", "epochs"}` where `params` is the flat,
        comparable view of the config.

    Raises:
        FileNotFoundError: If the directory has no manifest -- a run that never
            started is not something to silently drop from a comparison.
    """
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{run_dir} has no run_manifest.json")
    manifest = json.loads(manifest_path.read_text())

    config = {k: v for k, v in manifest.get("config", {}).items() if k not in _IGNORED_TOP_LEVEL}
    metrics_path = run_dir / "metrics.jsonl"
    epochs = (
        [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if metrics_path.exists()
        else []
    )
    return {
        "dir": run_dir,
        "manifest": manifest,
        "params": _flatten("", config),
        "epochs": epochs,
    }


def differing_params(runs: list[dict]) -> list[str]:
    """Names of the config fields that are not identical across every run."""
    keys = sorted({k for r in runs for k in r["params"]})
    return [k for k in keys if len({json.dumps(r["params"].get(k)) for r in runs}) > 1]


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_table(runs: list[dict]) -> str:
    """Render the comparison as a markdown table.

    Args:
        runs: Loaded runs, in the order they should appear.

    Returns:
        A markdown table. Hyperparameter columns cover only the fields that
        differ; with a single run there are none, and the table degrades to the
        summary columns.
    """
    varying = differing_params(runs)
    retracted = [r for r in runs if r["manifest"].get("retracted")]
    headers = ["Run", "Hash", *[k.split(".")[-1] for k in varying]]
    headers += ["Best val acc", "Best epoch", "s/epoch", "GPU C in/out", "Min", "Status"]
    if retracted:
        headers.append("Retracted")

    rows = []
    for run in runs:
        m = run["manifest"]
        summary = m.get("summary", {})
        runtime = m.get("runtime", {})
        best_epoch = summary.get("best_epoch")
        total = m.get("total_seconds")
        # A retracted row stays in the table, struck through, with its reason
        # visible. A run that quietly vanishes is worse than a wrong one in a
        # document whose whole job is being checkable (ADR-010).
        strike = "~~{}~~".format if m.get("retracted") else str
        temps = (runtime.get("gpu_temp_start_c"), runtime.get("gpu_temp_end_c"))
        row = [
            strike(f"`{run['dir'].as_posix()}`"),
            f"`{m.get('config_hash', '-')}`",
            *[_cell(run["params"].get(k)) for k in varying],
            strike(f"{summary['best_val_acc']:.2f}" if "best_val_acc" in summary else "-"),
            str(int(best_epoch) + 1) if best_epoch is not None and best_epoch >= 0 else "-",
            strike(_cell(runtime.get("seconds_per_epoch"))),
            "/".join(_cell(t) for t in temps),
            f"{total / 60:.1f}" if total is not None else "-",
            reaped_status(m),
        ]
        if retracted:
            note = m.get("retracted") or {}
            row.append(
                f"{note.get('date', '?')}: {note.get('reason', '(no reason given)')}"
                if note
                else ""
            )
        rows.append(row)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def plot_comparison(runs: list[dict], out_path: Path) -> None:
    """Overlay every run's val-accuracy curve on one axis.

    Args:
        runs: Loaded runs.
        out_path: PNG to write; parent directories are created.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        points = [
            (e["epoch"] + 1, e["val_acc"])
            for e in run["epochs"]
            if e.get("val_acc") == e.get("val_acc")  # drop NaN (skipped evals)
        ]
        if points:
            ax.plot(*zip(*points), label=run["dir"].name, marker=".", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def begin_marker(section: str) -> str:
    return f"<!-- BEGIN GENERATED: {section} -->"


def end_marker(section: str) -> str:
    return f"<!-- END GENERATED: {section} -->"


def write_section(doc_path: Path, section: str, body: str) -> None:
    """Replace `section`'s generated region in `doc_path`, or append it.

    The whole point of the sentinels: a rewrite is bounded. Text before the
    BEGIN marker and after the END marker is preserved byte for byte, so the
    hand-written findings in `EXPERIMENTS.md` cannot be clobbered by a
    regeneration no matter how many times it runs (ADR-010).

    Args:
        doc_path: Markdown file to update; created if absent.
        section: Stable section id, e.g. `"sprint-5"`. Not the heading -- the
            heading may be reworded without orphaning the region.
        body: Markdown to place between the markers.

    Raises:
        ValueError: If the file has a BEGIN marker with no matching END, or an
            END before its BEGIN. Guessing where a malformed region stops is how
            a generator eats the paragraph underneath it.
    """
    begin, end = begin_marker(section), end_marker(section)
    region = f"{begin}\n{body}\n{end}\n"
    text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

    start, stop = text.find(begin), text.find(end)
    if (start == -1) != (stop == -1) or (start != -1 and stop < start):
        raise ValueError(
            f"{doc_path} has a malformed generated region for section {section!r}: "
            f"BEGIN at {start}, END at {stop}. Fix the markers by hand before regenerating."
        )

    if start == -1:
        separator = "" if text.endswith("\n\n") or not text else "\n"
        doc_path.write_text(f"{text}{separator}{region}", encoding="utf-8")
    else:
        doc_path.write_text(
            text[:start] + region + text[stop + len(end) :].lstrip("\n"), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=[], help="Run directories.")
    parser.add_argument("--all", action="store_true", help="Every run under --run-root.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs"), help="Where the report lands.")
    parser.add_argument(
        "--section",
        default="comparison",
        help="Stable id of the generated region to rewrite, e.g. sprint-5. Regenerating a "
        "section replaces it in place and touches nothing else in the document.",
    )
    parser.add_argument("--title", default="", help="Heading shown inside the region.")
    args = parser.parse_args()

    run_dirs = list(args.runs)
    if args.all:
        run_dirs += sorted(d for d in args.run_root.iterdir() if (d / "run_manifest.json").exists())
    if not run_dirs:
        raise SystemExit("Nothing to compare: pass --runs or --all")

    runs = [load_run(d) for d in dict.fromkeys(run_dirs)]
    table = build_table(runs)

    # One figure per section, so regenerating sprint-5 cannot overwrite sprint-6's.
    figure_path = args.out / "figures" / f"comparison_{args.section}.png"
    plot_comparison(runs, figure_path)

    title = args.title or f"Comparison {datetime.now():%Y-%m-%d %H:%M}"
    body = (
        f"## {title}\n\n{table}\n\n"
        f"![val accuracy]({figure_path.relative_to(args.out).as_posix()})\n\n"
        f"*Generated by `make_report.py --section {args.section}`. Edits inside this region are "
        f"overwritten; write findings outside it.*"
    )
    experiments = args.out / "EXPERIMENTS.md"
    write_section(experiments, args.section, body)

    print(table)
    print(experiments)


if __name__ == "__main__":
    main()
