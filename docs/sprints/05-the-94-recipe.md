# Sprint 5 — The 94% recipe

**Goal:** ResNet-18-CIFAR reaches **≥94.0% val accuracy**, with every ingredient's contribution measured and logged.

**Depends on:** Sprint 4. **Blocks:** Sprints 6–8.
**Estimated effort:** 1 session of coding + several hours of unattended training.

---

## Milestone

PRD metric **M1** lands here. Everything before this sprint was infrastructure; everything after is refinement.

## The recipe

Nothing exotic — this is the well-trodden path, and the point is to walk it deliberately and measure each step.

| Ingredient | Value | Why |
|---|---|---|
| Optimizer | SGD, momentum 0.9, **Nesterov** | Beats Adam by ~1% on CIFAR ResNets at equal budget |
| Base LR | 0.1 (at batch 128) | Standard; scale linearly if batch changes |
| Weight decay | 5e-4, **excluded from BN & bias** | Primary regularizer here |
| Schedule | 5-epoch linear warmup → cosine to 0 | Warmup stabilizes the high-LR phase; cosine beats step by ~0.5% |
| Epochs | 120 | Where cosine + this augmentation converges |
| Batch size | 128 | Fits comfortably; the LR above is tuned for it |
| Label smoothing | 0.1 | Small, reliable gain; also improves calibration |
| AMP + channels_last | on | ~2× throughput, no accuracy cost |
| Augmentation | pad-4 random crop + horizontal flip | The Sprint 1 baseline; more comes in Sprint 6 |

## Components

Most of this exists from Sprint 3. This sprint is mostly configuration, ablation, and verification.

### C5.1 — LR range test

`scripts/lr_finder.py` — train for ~200 steps while exponentially increasing LR from 1e-5 to 10, plot loss vs LR (log x). The steepest-descent region should bracket 0.1. Save to `docs/figures/lr_finder.png`.

This is evidence, not ceremony: it converts "0.1 because everyone uses 0.1" into "0.1 because here is the curve." Takes two minutes to run.

### C5.2 — Verify the optimizer details

Confirm from Sprint 3 and fix if wrong:
- `nesterov=True` reaches the optimizer.
- Param groups genuinely split — print both group sizes at startup.
- `grad_clip_norm` unscales before clipping under AMP.

### C5.3 — Recipe configs

- `configs/resnet18.yaml` — the full 94% recipe above. Note that `base.yaml` already carries most of it (SGD 0.1, Nesterov, cosine, 5-epoch warmup, wd 5e-4 with BN/bias excluded, 120 epochs). The only genuine delta is `label_smoothing: 0.1`. Don't restate inherited values — a config that repeats its parent is a config that silently drifts from it.
- `configs/ablations/` — one file per row of the table below, each inheriting from `resnet18.yaml` and changing exactly one thing.

### C5.4 — Run the ablation ladder

Each run is 120 epochs unless noted, ~29 min measured. Total ~4.4 hours. Launch with `scripts/sweep.py` and step away.

**Run F first, not last.** F is the gate; the eight ablations are only interesting if the recipe they vary actually works. Finding out at minute 29 that F missed 94.5% costs you half an hour. Finding out at hour four costs you the evening. If F misses, **stop the sweep** and work the debug checklist at the end of this file before spending GPU time on ablations of a broken recipe.

Run `eval.py` on F's best checkpoint the moment it lands, and confirm test ≥ 94.0% (M1). Do not defer that to Sprint 8.

**Unattended-run hygiene**, since this is a laptop and the sweep is 4.4 hours: set the Windows power plan so the machine never sleeps or hibernates on AC — a sleep event mid-sweep kills every remaining run. Plug in, use the "Best performance" profile, and keep the machine on a hard surface. GPU temperature is now recorded per run, so thermal drift will be visible after the fact rather than mysterious.

| ID | Config | Change from full recipe | Expected val acc |
|---|---|---|---|
| E1 | `abl_adam` | AdamW 1e-3 instead of SGD | 93.0–94.0% |
| E2 | `abl_step_lr` | Step LR (60, 90) instead of cosine | 94.0–94.7% |
| E3 | `abl_no_warmup` | `warmup_epochs=0` | 94.3–95.0% |
| E4 | `abl_wd_on_bn` | `no_wd_on_bn_bias=false` | 94.2–94.9% |
| E5 | `abl_no_smoothing` | `label_smoothing=0.0` | 94.5–95.1% |
| E6 | `abl_wd_1e4` | `weight_decay=1e-4` | 93.8–94.6% |
| E7 | `abl_30ep` | `epochs=30` | 92.5–93.5% |
| E8 | `abl_no_aug` | crop and flip off | 87–90% |
| **F** | `resnet18.yaml` | — full recipe — | **94.5–95.3%** |

Run E8 even though the result is unsurprising. Quantifying that basic augmentation is worth ~6 points is the most striking number in the whole table, and it anchors everything in Sprint 6.

### C5.5 — Log the findings

Append all nine rows to `docs/EXPERIMENTS.md` with run directory, config hash, best val acc, best epoch, and total minutes. Where a result falls outside its expected band, write a sentence of hypothesis. Bands are estimates from the literature, not contracts — a miss is information.

---

## Acceptance criteria

- Full recipe run reaches **≥ 94.5% best val accuracy** *(Gate)*. Raised from 94.0% after Sprint 3 measured a ~0.6% selection-bias inflation in best-of-N val numbers — M1 is a **test**-set target, so the val gate needs headroom. Confirm with `eval.py` that test lands ≥ 94.0%.
- Wall-clock ≤ 45 minutes for that run (PRD **M3**). Report actual s/epoch.
- All 9 ablation runs completed, logged, and present in `EXPERIMENTS.md`.
- `docs/figures/lr_finder.png` exists and supports the LR choice.
- Test accuracy from `eval.py` is within 0.5% of val accuracy. A larger gap means the val split is unrepresentative or the test set got contaminated — investigate before proceeding.

```python
def test_nesterov_reaches_optimizer():
    opt = build_optimizer(model, cfg(optimizer="sgd", nesterov=True))
    assert all(g["nesterov"] for g in opt.param_groups)

def test_cosine_reaches_min_lr_at_final_step():
    sched = build_scheduler(opt, cfg(epochs=10, warmup_epochs=2, min_learning_rate=0.0),
                            steps_per_epoch=10)
    for _ in range(100): opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)
```

## If it stalls below 94%

Work in this order — the causes are ranked by how often they're actually the problem.

> **Ranking corrected 2026-08-01 after E4.** Item 1 sat at #1 on my prior, and local measurement contradicted it: E4 — weight decay applied to BN and bias, the thing ADR-003 forbids — gave the *highest* accuracy of all nine Sprint 5 runs, +0.40 over F. The "worth roughly 0.2–0.4%" claim originally attached to it has no supporting evidence in this repo. Verify the split is happening, but assume nothing about the size or even the sign of its effect until the seed-variance runs resolve it. Items 2 and 3 are the load-bearing ones — they are correctness bugs, not tuning preferences.

1. **Weight decay reaching BN.** Print the two param-group sizes. For ResNet-18-CIFAR the no-decay group holds **41** tensors and the decayed group 21, summing to the model's 62. *(Corrected in sprint 5 from "roughly 100", which is a ResNet-**50** count — 53 BN layers x 2 ≈ 106. ResNet-18-CIFAR has 20 BN layers x (scale + shift) + the classifier bias = 41, with all 20 convs bias-free. Verified against the built model; `Trainer` now logs both sizes at startup, and `test_resnet18_no_decay_group_holds_every_bn_and_bias` pins them. Chasing the old number would have meant "fixing" a split that was already correct.)*
2. **Scheduler stepping per epoch.** Plot the observed LR from `metrics.jsonl`. If it's a staircase with 120 steps rather than a smooth cosine, this is the bug, and it costs ~1%.
3. **Normalization stats wrong.** Re-run the Sprint 1 verification.
4. **Not enough epochs.** Cosine's final low-LR phase contributes a lot. At 60 epochs you should expect ~93.5%; that's the schedule, not a bug.
5. **Val leakage.** Confirm the disjointness test still passes.
6. **Model selection on the wrong metric.** `best.pth` must track val accuracy, not train.
7. **LR too high for the batch size.** If you changed batch size, scale LR linearly (`lr = 0.1 × batch / 128`).

---

## Implementation prompt

> Read `CLAUDE.md`, `docs/EXPERIMENTS.md`, and `docs/sprints/05-the-94-recipe.md`.
>
> Implement C5.1 (LR range test script), verify C5.2, and write the configs in C5.3 — `configs/resnet18.yaml` plus one ablation file per row of the C5.4 table, each changing exactly one thing from the full recipe.
>
> Then run `scripts/sweep.py --dir configs/ablations/` plus the full recipe. This is hours of GPU time, so start it and report progress as runs land.
>
> The gate is ≥94.0% val accuracy on the full recipe. If it stalls below that, work the checklist at the end of this sprint file in order — do not start randomly changing hyperparameters, and do not touch the test set.
>
> Finish by appending all rows to `docs/EXPERIMENTS.md` with a short interpretation of each result, especially any that fell outside its expected band.
