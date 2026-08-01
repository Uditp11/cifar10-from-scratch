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
benchmark predicted, and buys no accuracy. Two things make that number readable, both learned the
hard way in Sprint 4:

- `s/epoch` is the median of the run's **back half**. Epoch times ramp for several epochs — about 3
  for ResNet-18, 7 for SimpleCNN — on top of the fixed 60–75 s of DataLoader worker spawn that every
  Windows run pays in epoch 1. **A run under ~10 epochs cannot measure a steady state at all.**
- The A/B is on ResNet-18, not SimpleCNN, because SimpleCNN's epoch time is not reproducible across
  sessions (4.4 s vs 6.4 s, same config, same commit) while ResNet-18 holds ±4%. Measure performance
  on the model that actually loads the hardware — see `PRD.md` §5.

Sweeps run a directory of configs back to back, continuing past a failure:

```bash
python scripts/sweep.py --dir configs/ablations/ --dry-run
```

## Windows notes

- `DataLoader(num_workers > 0)` uses spawn-based multiprocessing on Windows, which
  re-imports the launching module in each worker. Any script that builds one must
  guard the call behind `if __name__ == "__main__":` (see `scripts/inspect_data.py`).
