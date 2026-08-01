# CIFAR-10

See `PRD.md` for the project brief and `docs/ARCHITECTURE.md` for the frozen interfaces.

## Setup

```bash
pip install -e ".[dev]"
```

## Makefile targets

`make` may not be available on Windows; the raw commands are listed alongside each target.

| Target | Command |
|---|---|
| `install` | `pip install -e ".[dev]"` |
| `test` | `pytest -q` |
| `lint` | `ruff check src/` |
| `format` | `ruff format src/ tests/` |
| `train` | `python scripts/train.py` |
| `eval` | `python scripts/eval.py` |
| `clean` | remove `runs/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__` dirs |
| `repro-check` | `python scripts/train.py repro.deterministic=true` |

## What a run leaves behind

Every training writes one self-describing directory. `run_manifest.json` records the git SHA and
dirty flag, the python/torch/CUDA/GPU it ran on, the full resolved config and its hash, and a
`runtime` block with the *resolved* device, the `channels_last` memory format, and the measured
steady-state seconds per epoch. Nothing in the project reports a number that isn't reconstructable
from one of these directories.

```
runs/{run_name}/
├── run_manifest.json      # status: running -> completed | failed | interrupted
├── config.yaml            # re-loads to a Config with the same hash
├── metrics.jsonl          # one JSON object per epoch, written before the console line
├── train.log              # console mirror
├── checkpoints/           # last.pth, best.pth
├── tb/                    # TensorBoard event files
└── figures/               # lr_schedule.png, curves.png, confusion_matrix.png
```

`tensorboard --logdir runs/` overlays every run in one board.

## Comparing runs

```bash
python scripts/make_report.py --runs runs/r18_runlevel runs/r18_cl_on --out docs/
```

The table carries only the hyperparameters that actually *differ* between the compared runs, plus
the summary columns, and gets appended to `docs/EXPERIMENTS.md` alongside an overlaid val-accuracy
curve. The first real use of it was confirming ADR-007 against full training runs — ResNet-18,
12 epochs, identical in every setting but one:

| Run | Hash | channels_last | Best val acc | s/epoch | Min |
|---|---|---|---|---|---|
| `runs/r18_runlevel` | `408a9d82636a` | False | 89.68 | **13.84** | 4.2 |
| `runs/r18_cl_on` | `d60ad5395da4` | True | 89.34 | **38.78** | 8.8 |

NHWC costs 2.80× the wall clock at 32×32 on this GPU, against the 2.90× that ADR-007's per-step
benchmark predicted, and buys no accuracy. That 4% agreement is the useful part: a per-step proxy is
trustworthy exactly when the model saturates the card, and misleading when it doesn't.

Each `--section` owns a sentinel-delimited region of `EXPERIMENTS.md` and regenerating rewrites only
that region — the hand-written findings around it are safe by construction (ADR-010). A run that
turns out to be wrong is retracted in place, by adding `"retracted": {"reason": ..., "date": ...}`
to its `run_manifest.json`; the row stays, struck through, with the reason visible.

## Measuring performance: SimpleCNN is not an instrument

A model too small to pull the GPU out of a low power state has epoch times that depend on **what ran
before it**. Measured in Sprint 4, identical config and identical commit:

| Model | Session A | Session B | Spread |
|---|---|---|---|
| SimpleCNN | 4.4 s/epoch | 6.4 s/epoch | ±20%, not reproducible |
| ResNet-18-CIFAR | 13.2 s/epoch | 13.8 s/epoch | ±4% |

ResNet-18 saturates the card, so it always clocks up and its timings repeat. SimpleCNN never draws
enough to leave a lower power state, so its number is a fact about the preceding minute rather than
about the model. **Run performance A/Bs on the model that loads the hardware.**

Two more things distort a naive reading of any timing here:

- `s/epoch` is the median of the run's **back half**. Epoch times ramp for several epochs — about 3
  for ResNet-18, 7 for SimpleCNN — on top of the fixed 60–75 s of DataLoader worker spawn that every
  Windows run pays in epoch 1. **A run under ~10 epochs cannot measure a steady state at all.**
- `sweep.py` runs jobs back to back, so later runs start on a hotter card and run order becomes
  confounded with temperature. Every manifest records `gpu_temp_start_c` / `gpu_temp_end_c` to make
  that visible; within a sweep, only the accuracy columns compare across rows.

Sweeps run a directory of configs back to back, continuing past a failure. Check what will run:

```bash
python scripts/sweep.py --dir configs/ablations/ --dry-run
```

Then launch it **detached** — this is the standard way to start any sweep, not a recovery step:

```bash
python scripts/sweep.py --dir configs/ablations/ --detach
```

`--detach` prints a PID and a log path and returns immediately; the sweep keeps running with no
parent console to be killed by. `tail -f` the log to watch it.

An attached sweep dies with the terminal that started it, and the failure is expensive and silent:
sprint 5 lost three of eight ablations that way. The one that was mid-startup was killed with
`0x40010004` (`DBG_TERMINATE_PROCESS`); the two that had not begun exited `0xC000026B`
(`STATUS_DLL_INIT_FAILED_LOGOFF`) — Python could not even load its DLLs, because the session was
already shutting down. None of those exit codes says "training problem", and all three cost a
re-run. At sprint 6's ~8 hours of sweeps, launching attached is a bet on nothing closing a terminal
for a working day.

Two related habits for long unattended runs:

- **Stop the machine sleeping.** `powercfg /change standby-timeout-ac 0`. A GPU-only workload does
  not reset Windows' idle timer, so the default 3-hour AC standby will suspend a sweep that is
  working perfectly. Plugged in, "Best performance", hard surface.
- **A killed run is visible in the table.** Manifests record `process: {pid, host}`, and
  `make_report.py` renders a run still marked `running` whose PID is gone as `killed` (ADR-006
  addendum). A process destroyed by `TerminateProcess` runs no `finally` and cannot label its own
  exit, so the reader settles it instead of the writer.

## Windows notes

- `DataLoader(num_workers > 0)` uses spawn-based multiprocessing on Windows, which
  re-imports the launching module in each worker. Any script that builds one must
  guard the call behind `if __name__ == "__main__":` (see `scripts/inspect_data.py`).
