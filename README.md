# CIFAR-10 from scratch

[![CI](https://github.com/Uditp11/cifar10-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Uditp11/cifar10-from-scratch/actions/workflows/ci.yml)

An image classification project in computer vision: a deep convolutional neural network (ResNet-18)
implemented from scratch in PyTorch and trained on CIFAR-10, a standard machine learning benchmark of
60,000 32×32 colour images across 10 categories. The residual blocks, the LR schedule, the
augmentations and the training loop are all written in this repository, and every recipe decision is
backed by a logged ablation you can re-run.

**96.06% test top-1** across two seeds (95.94 and 96.17). ResNet-18 (11,173,962 parameters), 200 epochs,
one laptop GPU. No pretrained weights, no `timm`, no `torchvision.models`, no Lightning.
`torchvision` supplies the CIFAR-10 download and `ToTensor`, nothing else.

| | |
|---|---|
| Test top-1 | **96.06%** (two seeds: 95.94, 96.17) |
| Test top-5 | 99.74% (two seeds: 99.73, 99.75) |
| Validation top-1 | 96.65% (two seeds: 96.62, 96.68) |
| val − test gap | 0.59 points |
| Training time | 43 to 49 min per run |
| Hardware | RTX 4060 Laptop GPU, torch 2.11.0+cu128 |

Two seeds is one degree of freedom, so these are reported as both values rather than as mean ± std.
A standard deviation from n=2 carries almost no information, and this repository declines to state
one anywhere (see ADR-012, which refuses a different number on the same grounds).

Test accuracy was measured once, after the configuration was chosen and
[written down first](docs/sprints/06-augmentation.md). `Trainer` takes a `(train, val)` 2-tuple, so
it is structurally incapable of seeing the test set; only `scripts/eval.py` ever loads it.

## Quickstart

```bash
git clone https://github.com/Uditp11/cifar10-from-scratch.git
cd cifar10-from-scratch
pip install -e .
python scripts/train.py --config configs/aug/e15_randaugment_cutout.yaml
```

Four commands including the clone, or five on a GPU, where torch has to be installed from the CUDA
index first (see below). The dataset fetches itself on the first run, but budget for that fetch. CIFAR-10 comes from `www.cs.toronto.edu`, which served this machine at roughly 100 KB/s,
so **the 170 MB download can take twenty minutes or more** on a connection where everything else
takes two. It lands in `data/` and is never fetched again.

Two things that bite on Windows in particular:

- **`pip install -e .` gets you CPU-only PyTorch.** PyPI's default `torch` wheel has no CUDA support,
  so a machine with a GPU trains on the CPU instead: `device: auto` resolves to `cpu`, everything
  still runs, roughly fifty times slower. For a GPU, install torch first:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- **Clone somewhere short.** Installing torch under a path deeper than about 200 characters fails
  with `OSError: [Errno 2]` unless Windows long-path support is on. `C:\projects\` is fine; a nested
  temp directory is not.

Evaluate a finished run:

```bash
python scripts/eval.py --run runs/<run-dir> --checkpoint best
```

## Where the accuracy actually came from

The most interesting result here is how lopsided the contributions are. Two lines of transform
config (pad-4 random crop and a horizontal flip) outweigh everything else by about four to one:

| Ingredient | Δ val accuracy | Status |
|---|---|---|
| Random crop + horizontal flip | **+5.80** | Established |
| Cutout(1, 16) | **+0.85** | Established, 4σ |
| 120 → 200 epochs | +0.29 | Weak, 1.4σ |
| Mixup *or* RandAugment on top of Cutout | +0.35 | Probably real, **not established** |

That last row is deliberately not rounded up. Two configurations reached it independently and in the
same direction, but it is 4.5σ under one noise estimate and 1.8σ under another, and the disagreement
between those estimates is [recorded as open](docs/EXPERIMENTS.md) rather than resolved in favour of
the convenient one. Mixup and RandAugment finished 0.03 apart, which is indistinguishable, so the one that
composes more simply was chosen and the reasoning written down.

Seven further ablations covering optimizer, schedule, warmup, weight decay, label smoothing and epoch
budget are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), along with the results that came out
against expectation and the claims this project has had to withdraw. They are the other seven of the
eight-config Sprint 5 sweep; the eighth is the crop-and-flip row above.

## Reproducibility

One testable claim per mode, both measured:

- **With `repro.deterministic: true`, runs reproduce bit-identically.** Not within a tolerance:
  identical `metrics.jsonl` scalars and identical weights. Three 10-epoch runs of
  `configs/repro_check.yaml` hashed to `866f747125539331`, one of them on battery under CPU
  contention, straddling a power-source change mid-run at 2.1× the wall clock. Machine state did not
  move the number in this mode (three runs; limits in ADR-012). Costs roughly **11 to 17%** throughput,
  a range on one degree of freedom, not a point estimate (ADR-012).
- **In default fast mode, run-to-run spread at a fixed seed is typically ~0.1 to 0.2, and 0.26 is the
  largest seen.** `cudnn.benchmark` times candidate kernels and picks a winner, so it can select a
  *different kernel* depending on machine state, and different kernels produce different
  floating-point results. `repro.seed` does not make a fast-mode run reproduce. That is the noise
  floor the table above is read against, and it is a property of the mode rather than a defect.

## Known failure modes

Cats. Per-class test accuracy runs from **90.0% (cat)** to 98.5% (automobile), and the largest single
error mode is 61 cat→dog and 50 dog→cat confusions. This is the expected CIFAR-10 result; the
augmentation work narrowed it without closing it.

## Working with runs

Every training writes one self-describing directory. `run_manifest.json` records the git SHA and
dirty flag, the python/torch/CUDA/GPU it ran on, the full resolved config and its hash, and a
`runtime` block with the resolved device and the measured steady-state seconds per epoch. No number
in this project is reported that cannot be rebuilt from one of these.

```
runs/{run_name}/
├── run_manifest.json      # status: running -> completed | failed | interrupted
├── config.yaml            # re-loads to a Config with the same hash
├── metrics.jsonl          # one JSON object per epoch
├── train.log              # console mirror
├── checkpoints/           # last.pth, best.pth
├── tb/                    # TensorBoard event files
└── figures/               # lr_schedule.png, curves.png, confusion_matrix.png
```

`tensorboard --logdir runs/` overlays every run in one board.
`python scripts/make_report.py --runs runs/a runs/b --out docs/` builds a comparison table carrying
only the hyperparameters that differ, into a sentinel-delimited region of `EXPERIMENTS.md` that
regeneration rewrites without touching the prose around it.

Sweeps run a directory of configs back to back, continuing past a failure, and should be launched
**detached**. This is the normal way to start one, not a recovery step:

```bash
python scripts/sweep.py --dir configs/aug/ --dry-run
python scripts/sweep.py --dir configs/aug/ --detach
```

An attached sweep dies with the terminal that started it; three of Sprint 5's eight ablations were
lost that way. Two related habits for long unattended runs: stop the machine sleeping
(`powercfg /change standby-timeout-ac 0`; a GPU-only workload does not reset Windows' idle timer),
and note that a run killed outright is rendered `killed` in the table by `make_report.py`, since a
process destroyed by `TerminateProcess` runs no `finally` and cannot label its own exit.

## Tests and CI

```bash
pytest -m "not gpu" -q
```

**123 tests**, mirroring the `src/` tree, covering config validation and inheritance, the data split
and normalization, augmentation correctness, model shapes and parameter counts, the training loop,
checkpoint/resume, and the run manifest. All run on CPU in about 90 seconds; two more are marked
`gpu` and deselected by default, and ten that need the real CIFAR-10 split skip themselves with a
named reason when it isn't present. CI runs the suite plus `ruff check` and `ruff format --check` on
Python 3.10 and 3.12.

The pipeline earned its place by finding three real defects rather than by existing:

- **A job-level hang.** `pytest` inherited the CIFAR-10 download, produced no output for nineteen
  minutes, and was killed by the job timeout with nothing naming the cause. The fetch now runs in its
  own bounded step.
- **A bad fetch bound.** Each bound was then set from the previous run's measured progress rather than
  guessed: 5 minutes reached 26.4%, 20 minutes reached 99.2%.
- **A test that passed on tolerance rather than correctness.** `test_resume_matches_uninterrupted_run`
  compared a 4-epoch config against a 2-epoch one, and both cosine and warmup scale to that field, so
  the two arms diverged *before* the resume it claimed to test. It passed on one platform on a ±0.5
  tolerance over a noisy val set and failed on the other. Pinned to a constant LR, the resumed arm now
  reproduces the straight arm's per-epoch losses exactly.

## Layout

```
src/cifar/           config, seeding, data, models, engine, tracking
scripts/             train, eval, sweep, make_report, inspect_data, lr_finder
configs/             base + recipe + ablation configs, all inheriting
tests/               pytest, CPU-only, mirroring the src tree
docs/EXPERIMENTS.md  every run that informed a decision, and what it taught
docs/ADR.md          every deviation from the frozen interfaces, and why
docs/ARCHITECTURE.md the frozen signatures every component was built against
```

`DataLoader(num_workers > 0)` uses spawn-based multiprocessing on Windows, which re-imports the
launching module in each worker, so any script building one guards the call behind
`if __name__ == "__main__":`.

---

Built by Udit Parihar: [GitHub](https://github.com/Uditp11) · [LinkedIn](https://www.linkedin.com/in/udit-parihar-518186191/).

## License

[MIT](LICENSE).
