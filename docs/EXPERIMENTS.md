# EXPERIMENTS.md — Ablation ledger

Every training run that informs a decision gets a row here. A result you can't point to a row for doesn't count.

**Rules**
- One row per run, added when the run *finishes*, not when it's planned.
- `val_acc` is best val accuracy across epochs. Test accuracy appears only for the final model.
- Every row links to its run directory and config hash so it can be rebuilt.
- Rows that fell outside their expected band get a hypothesis in the Notes column. A surprise is data.
- Regenerate a table with `python scripts/make_report.py --all --section <id>` rather than hand-editing.
  It rewrites only between that section's `BEGIN/END GENERATED` markers; prose outside them is safe.
- **Within a sweep, the `Min` and `s/epoch` columns are not a cross-run comparison.** `sweep.py` runs
  jobs back to back, so run order is confounded with GPU temperature — the ninth run starts on a card
  the first eight have been heating, and a 12-epoch run has already been watched drift 36.2 → 40.8
  s/epoch as it warmed. **Only the accuracy columns compare across rows.** Read `GPU C in/out` before
  reading any timing column, and re-measure a timing claim as a standalone run.
- **A wrong run is retracted, never deleted.** Add `"retracted": {"reason": "...", "date": "..."}` to
  its `run_manifest.json`; the row then renders struck through with the reason still visible. A row
  that silently vanishes is worse than a wrong one in a document whose whole job is being checkable.

---

## Hardware

Fill in once, in Sprint 3, so every timing below is interpretable.

| GPU | VRAM | CPU | PyTorch | CUDA | OS |
|---|---|---|---|---|---|
| NVIDIA RTX 4060 Laptop | 8 GB | i7-14700HX | 2.11.0+cu128 | 12.8 | Windows 11 |

Laptop GPU — record whether the machine was plugged in and on the "Best performance" power profile, since thermal throttling on runs over an hour is the main source of timing variance between otherwise identical runs.

**Sprint 3 conditions:** plugged in, Best-performance profile. Under sustained load the GPU held 2580 MHz at 65–68 °C with no throttle flags set; an fp16 4096³ matmul benchmarked at 28.9 TFLOPS, so the card is delivering its clocks. No run in this sprint exceeded 4 minutes, so none of these numbers carry thermal decay.

### Measured epoch times — RUN-LEVEL

Batch 128, AMP on, `channels_last` off (ADR-007). **Run-level**: a real training run, validation and
checkpoint writes and logging included. ADR-007's table is **per-step** and is not comparable with
this one — see `PRD.md` §5. Steady state is the median of a run's back half, not
everything-but-the-first-epoch (see "the ramp is not one epoch" below).

| Model | s/epoch (steady) | Across sessions | First epoch | Measured over |
|---|---|---|---|---|
| SimpleCNN | 4.4 / 6.4 | ±20%, **not reproducible** | 62–73 s | 30 ep (sprint 3), 14 ep ×2 (sprint 4) |
| ResNet-18-CIFAR | **13.8** | ±4% (13.2–14.2) | 80–86 s | 5 ep (sprint 3), 10 + 12 ep (sprint 4) |

The first epoch costs an extra 60–75 s on Windows: `num_workers=4` on both the train and val loaders means eight worker processes spawn and each unpickles its own copy of the CIFAR-10 split. Fixed per run, so it matters at 30 epochs and disappears at 300.

**The ramp is not one epoch.** ResNet-18 settles by epoch 3; SimpleCNN takes until roughly epoch 7 —
the sprint-3 gate run went 65.6, 8.4, 7.0, 5.2, 4.7, 4.3, then flat at 4.3–4.4 for the remaining 24.
Any run shorter than ~10 epochs reports a ramp-inflated number no matter how the median is taken.

**SimpleCNN cannot be used as a timing instrument.** Identical config and commit measured 4.4 s/epoch
in one session and 6.4 in the next. ResNet-18 varied ±4% over the same period. Likely GPU boost
state — ResNet-18 saturates the card, SimpleCNN never draws enough to leave a lower power state, so
its epoch time depends on what ran before it. Run performance A/Bs on ResNet-18.

---

## Sprint 3 — Baseline

| ID | Run | Config hash | Model | Epochs | Val acc | Test acc | Min | Notes |
|---|---|---|---|---|---|---|---|---|
| E0 | `runs/simple_cnn_20260801_004212` | `939482835ef3` | simple_cnn | 30 | **93.02** (best, ep 27) | 91.55 | 3.3 | Gate M4 (≥85%) passed by 8 points. Final-epoch val 92.60 / test 91.76. |

## Sprint 5 — The 94% recipe

**The numbers live in the generated `sprint-5` region at the bottom of this file.** This block is the
ID map only, so that "E4" means something stable in prose without duplicating a figure that a
regeneration would then silently disagree with.

| ID | Config | Hash | Change from full recipe |
|---|---|---|---|
| **F** | `resnet18.yaml` | `3b1347d7678d` | — full recipe — |
| E1 | `abl_adam` | `80f6f78140fa` | AdamW 1e-3 instead of SGD |
| E2 | `abl_step_lr` | `16bc07ef3ade` | Step LR (60, 90) instead of cosine |
| E3 | `abl_no_warmup` | `f2da8632ed18` | `warmup_epochs=0` |
| E4 | `abl_wd_on_bn` | `482da61d0efe` | `no_wd_on_bn_bias=false` |
| E5 | `abl_no_smoothing` | `05b2e398ac31` | `label_smoothing=0.0` |
| E6 | `abl_wd_1e4` | `9d29dcb47e98` | `weight_decay=1e-4` |
| E7 | `abl_30ep` | `d27955206ecd` | `epochs=30` |
| E8 | `abl_no_aug` | `64cf2b47d38e` | crop and flip off |

### M1 — the milestone number

**F's test accuracy is 94.60%.** `scripts/eval.py --run runs/resnet18_cifar_20260801_033919
--checkpoint best`, on the best-val checkpoint (epoch 115), 10 000 test images: top-1 **94.60%**,
top-5 99.44%, loss 0.6462. **M1 requires ≥ 94.0%, so M1 is met with 0.60 points of margin.** This is
a measured test number, not an inference from F's val accuracy — no gate in this project may be
settled by subtracting an assumed val/test offset from a val figure.

The rest of the acceptance criteria, for the record: best val **95.08%** against a 94.5% gate (+0.58);
val − test = **0.48%**, inside the 0.5% band that would otherwise mean an unrepresentative split;
wall clock **28.6 min** against M3's 45; 13.19 s/epoch. Only `scripts/eval.py` ever loaded the test
set, once, after training finished.

Per-class, F's floor is cat **88.6%** and dog **90.1%**; every other class is ≥ 92.3%. 59 cat→dog and
64 dog→cat confusions are the single largest error mode, which is the expected CIFAR-10 result and
the thing Sprint 6's augmentation work has the most room to move.

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
- `channels_last` is off by default (ADR-007) because it measured 2.8× slower here run-level on ResNet-18.
  Re-measure on any other GPU before assuming that still holds — it is one config flag.
- The val loader runs `num_workers=4` to serve ten batches of 512. Four worker spawns almost certainly cost more
  than they save, and they are half of the 60–75 s first-epoch penalty. Worth measuring against `num_workers=0`
  on the eval loaders.

## Findings

**Sprint 3 — the val/test gap is mostly checkpoint selection, not a split bug.** The gate run's best checkpoint scores 93.02% val and 91.55% test, a 1.47% gap wide enough to be worth chasing. Evaluating the *final* checkpoint instead gives 92.60% val and 91.76% test — a 0.84% gap. The difference between the two gaps is the price of picking the best of 30 epochs by val accuracy: roughly 0.6% of apparent accuracy that does not survive contact with data the selection never saw. The residual ~0.85% is the genuine val/test offset, consistent with held-out train-split data being slightly easier than CIFAR-10's official test split. Nothing here indicates a transform or split defect — but it does mean **every val number in this document is optimistic by roughly half a point** relative to what the test set will say, and cross-run comparisons on val are only fair because every run pays the same tax.

**Sprint 3 — NHWC is not free.** See ADR-007. The reflex to enable `channels_last` alongside AMP is ImageNet advice; at 32×32 with CIFAR channel counts it cost 3× the wall clock on this GPU. It is worth a single measured run before adopting any "standard" performance recipe at an unusual input size.

**Sprint 4 — a six-epoch run cannot measure seconds per epoch, and this document briefly claimed otherwise.** An earlier version of this section reported that ADR-007's 2.9× `channels_last` penalty was really 2.2× run-level, and explained the gap as validation and checkpointing diluting the ratio. Both halves were wrong, and both are withdrawn. The measurement was a 6-epoch SimpleCNN A/B whose `channels_last: false` arm was still ramping — sprint 3's 30-epoch run shows SimpleCNN going 8.4, 7.0, 5.2, 4.7, 4.3 before settling at 4.4, so *every* sample in a 6-epoch run is startup transient. The explanation was wrong too: sprint 3's numbers were already run-level, from full training runs with `eval_every: 1` and `save_every: 1`, so there was never a per-step-vs-run-level gap to explain for that model. The tooling repeated the error, because `runtime.seconds_per_epoch` excluded only the first epoch; it now takes the median of the run's back half, which reports 4.4 where the old code reported 6.5.

**Sprint 4 — ADR-007 confirmed run-level, on the right model this time.** ResNet-18, 12 epochs, identical but for the flag: 13.84 s/epoch NHWC-off against 38.78 NHWC-on, a **2.80× penalty** where ADR-007's per-step benchmark predicted 2.90×. For a GPU-bound model the two kinds of number agree to within 4%, which is the useful calibration: per-step numbers are a fine proxy *when the model saturates the card*, and misleading when it does not. Val accuracy came out 0.34% in NHWC-off's favour, which at 12 epochs is noise and is not evidence either way. One incidental observation — the NHWC arm's epoch times drift upward across the run (36.2 → 40.8 s) while the NHWC-off arm stays flat, consistent with the heavier arm being the only one that runs the card hot enough to throttle. Anything measuring a sustained 45-minute recipe should expect that drift; nothing in sprints 3–4 ran long enough to see it before.

**Sprint 4 — ResNet-18's budget is sound, and re-measuring it is what proved the rest wrong.** Re-measured run-level across three sessions (5, 10 and 12 epochs), ResNet-18 holds 13.2–14.2 s/epoch against sprint 3's 13.6. The 120-epoch recipe is ~29 minutes, comfortably inside M3's 45, and the sprint 5 sweep is ~4.4 h. Nothing in the budget moves. The useful discovery was the opposite of the one being looked for: **SimpleCNN's epoch time is not reproducible across sessions** — 4.4 in one, 6.4 in the next, identical config and commit — while ResNet-18 varies ±4% over the same window. The plausible mechanism is GPU boost state, since ResNet-18 saturates the card and SimpleCNN does not draw enough to leave a lower power state. The practical rule is that the cheap model is the untrustworthy instrument: measure performance on the one that loads the hardware.

**Sprint 4 — the gate found a resume bug that no accuracy number would have.** Ctrl-C used to write `last.pth` from a half-finished epoch under that epoch's number, so resuming skipped it and left the LR schedule short of `min_learning_rate` (ADR-009). Nothing in a metrics file looks wrong when this happens — the run reports its full epoch count and a plausible curve. It took an explicit "kill it and resume it" step in the gate to see it, which is an argument for gates that exercise the failure paths rather than only the happy one.

**Sprint 5 — crop and flip are worth 5.80 points, more than every other ingredient combined.** E8 turns off pad-4 random crop and horizontal flip and nothing else, and lands at **89.28%** against F's 95.08%. For scale: the other seven ablations move F by between −1.72 and +0.40, and their *total* spread is 2.12 points. Two lines of transform config, no parameters, no compute cost, beat the optimizer choice (E1, −1.06) by more than five to one and the entire epoch budget from 30 to 120 (E7, −1.72) by more than three to one. The mechanism is not subtle — 45 000 images is far too few for an 11.2M-parameter network, and E8 reaches 100% train accuracy while its val curve flattens 6 points lower — but the *size* of the gap is worth having measured rather than assumed, because it sets the prior for Sprint 6. If crop and flip alone are worth 5.80, the marginal return on Cutout, Mixup and RandAugment stacked on top of them will be a small fraction of that, and Sprint 6's expected deltas should be read in tenths of a point, not points.

**Sprint 5 — cosine barely beat step, and the sprint's ~0.5% estimate was too generous.** E2 swaps cosine for step decay at epochs 60/90 and scores **95.00%** against F's 95.08% — a gap of **0.08 points**, well inside the noise this document can resolve on one seed, and far below the 0.5% the sprint file predicted. The honest reading is that at 120 epochs with warmup and a well-chosen base LR, *how* the LR reaches zero matters much less than *that* it does. Both schedules spend their last twenty-odd epochs at a low enough LR to converge, and both runs peak late (F at 115, E2 at 109). Where cosine still earns its place is having no milestones to tune: E2's advantage is contingent on 60/90 happening to suit a 120-epoch budget, and those numbers would need re-picking for Sprint 6's 200 epochs and Sprint 7's 300, while cosine rescales itself. That is a maintenance argument, not an accuracy one, and the recipe keeps cosine on those grounds rather than on the 0.08.

**Sprint 5 — E4 did not confirm ADR-003; it contradicted it, and ADR-003 stands anyway for now.** E4 applies weight decay to BN scales, BN shifts and biases — the thing ADR-003 exists to prevent — and it produced **95.48%**, the *highest val accuracy of all nine runs*, **+0.40 above F**. Its expected band was 94.2–94.9%, so it overshot by 0.58. This is the ablation that was supposed to be a formality and it went the other way. Three things keep it from overturning the ADR on the spot: it is one seed, +0.40 is close enough to plausible CIFAR seed-to-seed spread that it cannot be separated from noise here, and the effect being tested is genuinely small at this scale — 41 of 62 parameter tensors are in the no-decay group but they hold only ~0.1% of the model's 11.2M parameters, so whichever way the decision goes it is moving very little weight. What is now clear is that **the ~1% cost this repo has been attributing to weight decay reaching BN is not supported by any measurement in this document**, and that number should stop being repeated as though it were. The sprint 5 debug checklist ranks it the single most common cause of a stalled run; on this evidence it belongs much lower. Sprint 7's three-seed runs are what can settle it, and until they do, ADR-003 is retained on its original theoretical argument with this result recorded against it.

**Sprint 5 — selection bias is not a constant tax; it scales with how much the val curve is still moving.** Sprint 3 measured ~0.6% of best-of-N inflation on a 30-epoch SimpleCNN run and this document has since treated "every val number here is optimistic by roughly half a point" as a standing correction. F says otherwise: its **best val is 95.08% and its final-epoch val is 94.98%, a difference of 0.10**, and its val−test gap came in at **0.48%** rather than the ~1.4% Sprint 3 saw. The mechanism is the schedule. Picking the best of N epochs only wins something when the epochs differ; cosine-to-zero over 120 epochs spends its last stretch at a vanishing LR, F's final twenty epochs sit inside a 0.3-point band, and "best of 120" is therefore barely a selection at all. A 30-epoch run is still descending when it stops, its epochs genuinely differ, and best-of-N harvests real noise — which is exactly the regime Sprint 3 measured in. So the correction is not a property of the project, it is a property of the run: **inflation scales with the residual movement in the val curve at the end of training**, approaching zero for a converged long cosine and growing for anything truncated. Practically, F's val figures can be read almost at face value, E7's (30 epochs) cannot, and the standing "subtract half a point" rule should be applied per-run rather than globally. It also means the 94.5% val gate — set 0.5 above M1 specifically to absorb this inflation — was more conservative than it needed to be for a run of this shape.

**Sprint 5 — E3 (no warmup, +0.22) is a hypothesis, not a finding.** Removing the 5-epoch warmup scored **95.30%**, above F and above its own expected band. A plausible mechanism: warmup's established benefit is largely a *large-batch* effect — it exists to stop the first few high-LR steps from wrecking a network whose gradient estimates are being averaged over thousands of samples — and at batch 128 with LR 0.1 there is very little to stabilize, so neutral-to-slightly-negative is an unsurprising outcome rather than a contradiction. Against that, the effect is 0.22 points on a single seed, F and E3 differ by less than E4 does from F, and none of these three separations is resolvable without repeats. **The recipe is not changing on this.** Warmup stays in, both because the evidence does not support removing it and because Sprints 6 and 7 raise the epoch count and may raise the batch size, which is the regime where warmup is supposed to matter. Sprint 7's seed-variance runs (E17–E19) are the right instrument; if seed-to-seed std there is ≲0.1, E3 and E4 both become real and worth acting on, and if it is ~0.25 they were always noise.

**Sprint 5 — three ablations were lost to a console teardown, and the run that recorded it lied for fourteen hours.** The eight-config sweep completed five runs and then died: the launching process exited, and Windows destroyed its children. `abl_step_lr` was ~30 s into startup and took `0x40010004` (`DBG_TERMINATE_PROCESS`); `abl_wd_1e4` and `abl_wd_on_bn` had not begun and exited `0xC000026B` (`STATUS_DLL_INIT_FAILED_LOGOFF`) — the interpreter could not load its DLLs because the session was already shutting down. Not one of those exit codes describes a training fault, and all three configs re-ran clean afterwards with no change. Two things came out of it. First, `sweep.py --detach` now launches sweeps with no controlling console, which is the standard way to start one from here on — Sprint 6 is roughly eight hours of sweeps and this cannot be a recovery procedure. Second, the killed run left `runs/resnet18_cifar_20260801_061849` sitting at `status: "running"` with zero epochs logged, because a process destroyed by `TerminateProcess` executes no `finally` and cannot label its own exit. ADR-006 had explicitly claimed that could not happen; its addendum records why the reasoning failed and moves the fix to the reader, where manifests now carry `process: {pid, host}` and `make_report.py` renders a `running` run whose PID is dead as `killed`. The stub directory was deleted rather than retracted: retract-never-delete protects *claims*, and a zero-epoch run that never reached the ledger makes none.

Write these as you go, in prose, one short paragraph each. The table holds the numbers; this section holds the understanding. Candidates that usually turn out to be worth writing up:

- How much basic augmentation (crop + flip) is actually worth. Expect the largest single delta in the whole document.
- Cosine vs step LR at equal epochs.
- The cost of applying weight decay to BN and bias parameters.
- Whether mixup/cutmix helped at 120 epochs vs 200 — and why the answer differs.
- Label smoothing's effect on accuracy vs its effect on calibration.
- Seed-to-seed variance, and what that implies about how precisely any single-run CIFAR number should be read.

---

## Generated tables

Regions below marked `BEGIN GENERATED` are rewritten wholesale by
`scripts/make_report.py --section <id>`. Everything outside them — every finding, hypothesis and
table above this line — is never touched by the generator (ADR-010). Write prose outside a region;
it will survive any number of regenerations.

<!-- BEGIN GENERATED: sprint-4-channels-last -->
## Sprint 4 - channels_last A/B, run-level (ResNet-18-CIFAR, 12 epochs)

| Run | Hash | channels_last | Best val acc | Best epoch | s/epoch | GPU C in/out | Min | Status |
|---|---|---|---|---|---|---|---|---|
| `runs/r18_runlevel` | `408a9d82636a` | False | 89.68 | 11 | 13.84 | -/- | 4.2 | completed |
| `runs/r18_cl_on` | `d60ad5395da4` | True | 89.34 | 12 | 38.78 | -/- | 8.8 | completed |

![val accuracy](figures/comparison_sprint-4-channels-last.png)

*Generated by `make_report.py --section sprint-4-channels-last`. Edits inside this region are overwritten; write findings outside it.*
<!-- END GENERATED: sprint-4-channels-last -->

<!-- BEGIN GENERATED: sprint-4-superseded -->
## Sprint 4 - superseded: channels_last A/B on SimpleCNN (retracted)

| Run | Hash | channels_last | Best val acc | Best epoch | s/epoch | GPU C in/out | Min | Status | Retracted |
|---|---|---|---|---|---|---|---|---|---|
| ~~`runs/cl_off`~~ | `0342e5ce1f30` | False | ~~90.30~~ | 13 | ~~6.43~~ | -/- | 2.8 | completed | 2026-08-01: SimpleCNN is not a reproducible timing instrument -- its epoch time varies 4.4-6.4 s across sessions on identical config (see Findings). Superseded by the ResNet-18 A/B. |
| ~~`runs/cl_on`~~ | `50365b497b89` | True | ~~90.06~~ | 13 | ~~14.3~~ | -/- | 4.4 | completed | 2026-08-01: SimpleCNN is not a reproducible timing instrument -- its epoch time varies 4.4-6.4 s across sessions on identical config (see Findings). Superseded by the ResNet-18 A/B. |

![val accuracy](figures/comparison_sprint-4-superseded.png)

*Generated by `make_report.py --section sprint-4-superseded`. Edits inside this region are overwritten; write findings outside it.*
<!-- END GENERATED: sprint-4-superseded -->

<!-- BEGIN GENERATED: sprint-5 -->
## Sprint 5 - the 94% recipe and its ablation ladder

| Run | Hash | horizontal_flip_p | random_crop_padding | label_smoothing | learning_rate | no_wd_on_bn_bias | optimizer | scheduler | warmup_epochs | weight_decay | epochs | Best val acc | Best epoch | s/epoch | GPU C in/out | Min | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `runs/resnet18_cifar_20260801_033919` | `3b1347d7678d` | 0.5 | 4 | 0.1 | 0.1 | True | sgd | cosine | 5 | 0.0005 | 120 | 95.08 | 115 | 13.19 | 42/65 | 28.6 | completed |
| `runs/resnet18_cifar_20260801_041953` | `80f6f78140fa` | 0.5 | 4 | 0.1 | 0.001 | True | adamw | cosine | 5 | 0.0005 | 120 | 94.02 | 115 | 13.08 | 51/71 | 30.8 | completed |
| `runs/resnet18_cifar_20260801_201713` | `16bc07ef3ade` | 0.5 | 4 | 0.1 | 0.1 | True | sgd | step | 5 | 0.0005 | 120 | 95.00 | 109 | 13.36 | 40/78 | 27.5 | completed |
| `runs/resnet18_cifar_20260801_054910` | `f2da8632ed18` | 0.5 | 4 | 0.1 | 0.1 | True | sgd | cosine | 0 | 0.0005 | 120 | 95.30 | 118 | 13.05 | 48/63 | 28.7 | completed |
| `runs/resnet18_cifar_20260801_211243` | `482da61d0efe` | 0.5 | 4 | 0.1 | 0.1 | False | sgd | cosine | 5 | 0.0005 | 120 | 95.48 | 112 | 12.62 | 60/65 | 26.6 | completed |
| `runs/resnet18_cifar_20260801_052033` | `05b2e398ac31` | 0.5 | 4 | 0 | 0.1 | True | sgd | cosine | 5 | 0.0005 | 120 | 94.86 | 106 | 12.82 | 50/70 | 27.7 | completed |
| `runs/resnet18_cifar_20260801_204500` | `9d29dcb47e98` | 0.5 | 4 | 0.1 | 0.1 | True | sgd | cosine | 5 | 0.0001 | 120 | 94.42 | 116 | 12.68 | 59/85 | 26.8 | completed |
| `runs/resnet18_cifar_20260801_041046` | `d27955206ecd` | 0.5 | 4 | 0.1 | 0.1 | True | sgd | cosine | 5 | 0.0005 | 30 | 93.36 | 28 | 13.85 | 41/75 | 8.3 | completed |
| `runs/resnet18_cifar_20260801_045134` | `64cf2b47d38e` | 0 | 0 | 0.1 | 0.1 | True | sgd | cosine | 5 | 0.0005 | 120 | 89.28 | 107 | 12.57 | 52/77 | 28.1 | completed |

![val accuracy](figures/comparison_sprint-5.png)

*Generated by `make_report.py --section sprint-5`. Edits inside this region are overwritten; write findings outside it.*
<!-- END GENERATED: sprint-5 -->
