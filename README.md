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
python scripts/make_report.py --runs runs/cl_off runs/cl_on --out docs/
```

The table carries only the hyperparameters that actually *differ* between the compared runs, plus
the summary columns, and gets appended to `docs/EXPERIMENTS.md` alongside an overlaid val-accuracy
curve. The first real use of it was re-measuring ADR-007 — SimpleCNN, 6 epochs, identical in every
setting but one:

| Run | Hash | channels_last | Best val acc | s/epoch | Min |
|---|---|---|---|---|---|
| `runs/cl_off` | `e52f6720740b` | False | 86.38 | **6.54** | 1.8 |
| `runs/cl_on` | `b23f6d51ffa0` | True | 86.92 | **14.36** | 2.5 |

NHWC costs 2.2× the wall clock at 32×32 on this GPU and buys 0.54% of val accuracy that is well
inside seed noise at 6 epochs. `s/epoch` is the median of the steady-state epochs — the first epoch
of every Windows run carries a fixed 60–75 s of DataLoader worker spawn and is excluded.

Sweeps run a directory of configs back to back, continuing past a failure:

```bash
python scripts/sweep.py --dir configs/ablations/ --dry-run
```

## Windows notes

- `DataLoader(num_workers > 0)` uses spawn-based multiprocessing on Windows, which
  re-imports the launching module in each worker. Any script that builds one must
  guard the call behind `if __name__ == "__main__":` (see `scripts/inspect_data.py`).
