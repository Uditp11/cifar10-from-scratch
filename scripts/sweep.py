"""Run every config in a directory, one training each, and summarize.

Usage:
    python scripts/sweep.py --dir configs/ablations/ --detach
    python scripts/sweep.py --dir configs/ablations/
    python scripts/sweep.py --dir configs/ablations/ --dry-run

A failure does not stop the sweep: the point is to start a set of ablations and
come back later, and losing the remaining six runs because the second one OOM'd
is exactly the outcome that wastes an evening. Failures are reported in the
summary table with their exit code.

`--detach` is the answer to the *other* way an evening gets lost, and it is the
default way to launch anything longer than a few minutes. An attached sweep dies
with the console that started it: sprint 5 lost three of eight ablations when the
launching process exited, and the two that had not started yet could not even
load their DLLs (`STATUS_DLL_INIT_FAILED_LOGOFF`). Nothing about that is
recoverable from inside the sweep, because the sweep is what gets killed. So
`--detach` re-launches this script in a process with no parent to be killed by,
prints the PID and log path, and returns immediately.

Child stdout is streamed as it arrives, so a long sweep is still watchable; the
last line of each child is its run directory (every script guarantees that),
which is how the summary finds the manifest to report from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_TRAIN = Path(__file__).resolve().parent / "train.py"


def detach(argv: list[str], log_path: Path) -> int:
    """Re-launch this script detached from the current console, and return its PID.

    The child gets no inherited console and its own process group, so neither a
    Ctrl-C to the launcher nor the launcher's own teardown reaches it. Output
    goes to `log_path` because a detached process has nowhere else to write.

    Args:
        argv: Arguments to re-invoke this script with, `--detach` removed.
        log_path: File to receive the child's stdout and stderr.

    Returns:
        The detached process's PID.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # DETACHED_PROCESS: no console at all, so a console-teardown event has
        # nothing to deliver to. NEW_PROCESS_GROUP additionally spares it the
        # Ctrl-C that would otherwise fan out to the whole group.
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs: dict = {"creationflags": creationflags}
    else:
        kwargs = {"start_new_session": True}  # setsid: escapes the controlling tty

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **kwargs,
        )
    return proc.pid


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
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Re-launch detached so the sweep survives this console closing. Use for "
        "anything longer than a few minutes.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Where a detached sweep writes its output. Default: runs/sweep_{timestamp}.log",
    )
    args = parser.parse_args()

    configs = sorted(p for p in args.dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No *.yaml configs in {args.dir}")

    if args.detach:
        # Validate before detaching: a bad --dir should fail here, in front of
        # the operator, not silently into a log file nobody reads for an hour.
        log_path = args.log or Path("runs") / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.log"
        argv = [a for a in sys.argv[1:] if a != "--detach"]
        pid = detach(argv, log_path)
        print(f"detached sweep pid {pid}, {len(configs)} configs")
        print(f"tail -f {log_path.as_posix()}")
        print(log_path)
        return

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
