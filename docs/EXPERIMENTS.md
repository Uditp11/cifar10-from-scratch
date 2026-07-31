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

| GPU | VRAM | PyTorch | CUDA | OS |
|---|---|---|---|---|
| _TBD_ | | | | |

---

## Sprint 3 — Baseline

| ID | Run | Config hash | Model | Epochs | Val acc | Test acc | Min | Notes |
|---|---|---|---|---|---|---|---|---|
| E0 | | | simple_cnn | 30 | | | | Sanity baseline, gate M4 (≥85%) |

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

## Findings

Write these as you go, in prose, one short paragraph each. The table holds the numbers; this section holds the understanding. Candidates that usually turn out to be worth writing up:

- How much basic augmentation (crop + flip) is actually worth. Expect the largest single delta in the whole document.
- Cosine vs step LR at equal epochs.
- The cost of applying weight decay to BN and bias parameters.
- Whether mixup/cutmix helped at 120 epochs vs 200 — and why the answer differs.
- Label smoothing's effect on accuracy vs its effect on calibration.
- Seed-to-seed variance, and what that implies about how precisely any single-run CIFAR number should be read.
