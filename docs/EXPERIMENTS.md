# EXPERIMENTS.md — Ablation ledger

Every training run that informs a decision gets a row here. A result you can't point to a row for doesn't count.

**Rules**
- One row per run, added when the run *finishes*, not when it's planned.
- `val_acc` is best val accuracy across epochs. Test accuracy appears only for the final model.
- Every row links to its run directory and config hash so it can be rebuilt.
- Rows that fell outside their expected band get a hypothesis in the Notes column. A surprise is data.
- Regenerate the table with `python scripts/make_report.py --all` rather than hand-editing.

---

## Hardware

Fill in once, in Sprint 3, so every timing below is interpretable.

| GPU | VRAM | CPU | PyTorch | CUDA | OS |
|---|---|---|---|---|---|
| NVIDIA RTX 4060 Laptop | 8 GB | i7-14700HX | 2.11.0+cu128 | 12.8 | Windows 11 |

Laptop GPU — record whether the machine was plugged in and on the "Best performance" power profile, since thermal throttling on runs over an hour is the main source of timing variance between otherwise identical runs.

**Sprint 3 conditions:** plugged in, Best-performance profile. Under sustained load the GPU held 2580 MHz at 65–68 °C with no throttle flags set; an fp16 4096³ matmul benchmarked at 28.9 TFLOPS, so the card is delivering its clocks. No run in this sprint exceeded 4 minutes, so none of these numbers carry thermal decay.

### Measured epoch times

Batch 128, AMP on, `channels_last` off (ADR-007). Steady-state excludes the first epoch.

| Model | s/epoch | First epoch | Notes |
|---|---|---|---|
| SimpleCNN | 4.4 (4.27–4.63 mean) | 65.6 s | 351 steps/epoch |
| ResNet-18-CIFAR | 13.6 (13.5–13.7) | 79.9 s | measured over 5 epochs |

The first epoch costs an extra 60–75 s on Windows: `num_workers=4` on both the train and val loaders means eight worker processes spawn and each unpickles its own copy of the CIFAR-10 split. Fixed per run, so it matters at 30 epochs and disappears at 300.

---

## Sprint 3 — Baseline

| ID | Run | Config hash | Model | Epochs | Val acc | Test acc | Min | Notes |
|---|---|---|---|---|---|---|---|---|
| E0 | `runs/simple_cnn_20260801_004212` | `939482835ef3` | simple_cnn | 30 | **93.02** (best, ep 27) | 91.55 | 3.3 | Gate M4 (≥85%) passed by 8 points. Final-epoch val 92.60 / test 91.76. |

## Sprint 5 — The 94% recipe

| ID | Run | Hash | Change from full recipe | Epochs | Val acc | Δ vs F | Min | Notes |
|---|---|---|---|---|---|---|---|---|
| F | | | — full recipe — | 120 | | — | | **Gate: ≥94.0%** |
| E1 | | | AdamW 1e-3 instead of SGD | 120 | | | | |
| E2 | | | Step LR (60,90) instead of cosine | 120 | | | | |
| E3 | | | No warmup | 120 | | | | |
| E4 | | | Weight decay applied to BN/bias | 120 | | | | |
| E5 | | | No label smoothing | 120 | | | | |
| E6 | | | weight_decay 1e-4 | 120 | | | | |
| E7 | | | 30 epochs | 30 | | | | |
| E8 | | | No crop, no flip | 120 | | | | |

## Sprint 6 — Augmentation

| ID | Run | Hash | Augmentation | Epochs | Val acc | Δ vs E9 | Min | Notes |
|---|---|---|---|---|---|---|---|---|
| E9 | | | baseline (crop+flip) | 200 | | — | | Reference for this block |
| E10 | | | + Cutout(1,16) | 200 | | | | |
| E11 | | | + Mixup α=0.2 | 200 | | | | |
| E12 | | | + CutMix α=1.0 | 200 | | | | |
| E13 | | | + Cutout + Mixup | 200 | | | | |
| E14 | | | + Cutout + CutMix | 200 | | | | |
| E15 | | | + RandAugment(2,9) + Cutout | 200 | | | | |
| E16 | | | Cutout length 8 / 16 / 20 | 200 | | | | Sensitivity sweep |

## Sprint 7 — Final

| ID | Run | Hash | Setup | Epochs | Seed | Val acc | Test acc | ECE | Min |
|---|---|---|---|---|---|---|---|---|---|
| E17 | | | best aug + EMA | 300 | 42 | | | | |
| E18 | | | best aug + EMA | 300 | 1337 | | | | |
| E19 | | | best aug + EMA | 300 | 2024 | | | | |
| E20 | | | E17 + TTA (hflip) | — | 42 | | | | |
| E21 | | | E17, no label smoothing | 300 | 42 | | | | Calibration comparison |

**Headline result:** _mean ± std over seeds 42/1337/2024, to be filled in Sprint 7._

---

## Candidate ablations

- `random_crop_padding` uses `padding_mode="reflect"` (`build_train_transform`, sprint 1). Zero padding is the
  classic recipe and is worth testing in Sprint 5.
- `channels_last` is off by default (ADR-007) because it measured 3× slower here. Re-measure on any other GPU
  before assuming that still holds — it is one config flag.
- The val loader runs `num_workers=4` to serve ten batches of 512. Four worker spawns almost certainly cost more
  than they save, and they are half of the 60–75 s first-epoch penalty. Worth measuring against `num_workers=0`
  on the eval loaders.

## Findings

**Sprint 3 — the val/test gap is mostly checkpoint selection, not a split bug.** The gate run's best checkpoint scores 93.02% val and 91.55% test, a 1.47% gap wide enough to be worth chasing. Evaluating the *final* checkpoint instead gives 92.60% val and 91.76% test — a 0.84% gap. The difference between the two gaps is the price of picking the best of 30 epochs by val accuracy: roughly 0.6% of apparent accuracy that does not survive contact with data the selection never saw. The residual ~0.85% is the genuine val/test offset, consistent with held-out train-split data being slightly easier than CIFAR-10's official test split. Nothing here indicates a transform or split defect — but it does mean **every val number in this document is optimistic by roughly half a point** relative to what the test set will say, and cross-run comparisons on val are only fair because every run pays the same tax.

**Sprint 3 — NHWC is not free.** See ADR-007. The reflex to enable `channels_last` alongside AMP is ImageNet advice; at 32×32 with CIFAR channel counts it cost 3× the wall clock on this GPU. It is worth a single measured run before adopting any "standard" performance recipe at an unusual input size.

**Sprint 4 — ADR-007 re-measured end to end, and the penalty is 2.2×, not 3×.** The channels_last A/B below is the first comparison produced by `make_report.py` rather than by a hand-run benchmark, and it disagrees slightly with ADR-007's table: 14.36 vs 6.54 s/epoch, a 2.2× penalty where the sprint-3 micro-benchmark said 2.9×. The difference is what a full epoch contains that a `ms/step` loop does not — validation over 5,000 images, checkpoint writes, logging — all of which are format-independent and dilute the ratio. Both numbers are correct about their own question; the run-level one is the one that predicts wall clock, so it is the one the epoch budgets should be read against. The direction is unchanged and ADR-007 stands. Val accuracy differed by 0.54% in NHWC's favour, which at 6 epochs is well inside seed noise and is not evidence of anything.

**Sprint 4 — the gate found a resume bug that no accuracy number would have.** Ctrl-C used to write `last.pth` from a half-finished epoch under that epoch's number, so resuming skipped it and left the LR schedule short of `min_learning_rate` (ADR-009). Nothing in a metrics file looks wrong when this happens — the run reports its full epoch count and a plausible curve. It took an explicit "kill it and resume it" step in the gate to see it, which is an argument for gates that exercise the failure paths rather than only the happy one.

Write these as you go, in prose, one short paragraph each. The table holds the numbers; this section holds the understanding. Candidates that usually turn out to be worth writing up:

- How much basic augmentation (crop + flip) is actually worth. Expect the largest single delta in the whole document.
- Cosine vs step LR at equal epochs.
- The cost of applying weight decay to BN and bias parameters.
- Whether mixup/cutmix helped at 120 epochs vs 200 — and why the answer differs.
- Label smoothing's effect on accuracy vs its effect on calibration.
- Seed-to-seed variance, and what that implies about how precisely any single-run CIFAR number should be read.

---

*Everything below this line is appended by `scripts/make_report.py`. Don't hand-edit it.*

## Sprint 4 - channels_last A/B (SimpleCNN, 6 epochs)

| Run | Hash | channels_last | Best val acc | Best epoch | s/epoch | Min | Status |
|---|---|---|---|---|---|---|---|
| `runs/cl_off` | `e52f6720740b` | False | 86.38 | 6 | 6.54 | 1.8 | completed |
| `runs/cl_on` | `b23f6d51ffa0` | True | 86.92 | 6 | 14.36 | 2.5 | completed |

![val accuracy](figures/comparison.png)
