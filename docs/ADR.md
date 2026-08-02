# ADR.md — Architecture Decision Record

Any deviation from `ARCHITECTURE.md`, any added dependency, and any decision that a future reader would otherwise have to reverse-engineer from the code goes here.

## Template

```
## ADR-NNN — <short title>
**Date:** YYYY-MM-DD · **Sprint:** N · **Status:** accepted | superseded by ADR-MMM

**Context.** What situation forced a choice.
**Decision.** What was chosen.
**Alternatives.** What else was on the table and why it lost.
**Consequences.** What this costs, and what it now makes hard.
```

---

## ADR-001 — Fixed val split committed to the repo
**Date:** 2026-07-31 · **Sprint:** 1 · **Status:** accepted

**Context.** The 5,000-image validation split can be derived from a seed at runtime or fixed once and committed. A seed-derived split makes val accuracy incomparable across runs whenever the seed changes, which is exactly what happens during seed-variance experiments in Sprint 7.

**Decision.** Generate a stratified 500-per-class split once, commit it as `configs/val_indices.json`, and load it unconditionally regardless of `cfg.repro.seed`.

**Alternatives.** Seed-derived split (rejected: breaks cross-seed comparability). K-fold CV (rejected: 5× the compute for a benchmark where a single split is the convention).

**Consequences.** Val accuracy is comparable across every run in the project. The split is one arbitrary draw, so val accuracy carries a small fixed bias relative to the true generalization gap — which is fine, because the test set is the arbiter and val is only used for selection.

---

## ADR-002 — CIFAR-style ResNet stem
**Date:** _TBD_ · **Sprint:** 2 · **Status:** accepted

**Context.** Torchvision's ResNet-18 opens with a 7×7 stride-2 conv plus a 3×3 stride-2 maxpool, taking a 32×32 input down to 8×8 before the first residual block.

**Decision.** Replace the stem with a single 3×3 stride-1 conv and drop the maxpool, so stage resolutions are 32 → 16 → 8 → 4.

**Alternatives.** Upsample CIFAR images to 224×224 and use the ImageNet architecture (rejected: ~49× the compute for no benefit without pretrained weights).

**Consequences.** Parameter count is 11,173,962 rather than torchvision's 11,689,512 (the difference is the stem and the 10-class head). Published torchvision numbers are not directly comparable.

---

## ADR-003 — Weight decay excluded from BN and bias parameters
**Date:** _TBD_ · **Sprint:** 3 · **Status:** accepted

**Context.** Applying L2 to BatchNorm scale/shift and to biases shrinks parameters that carry no capacity in the sense weight decay is meant to constrain, and measurably hurts accuracy.

**Decision.** Split into two param groups: all parameters with `ndim <= 1` get `weight_decay=0`.

**Consequences.** Optimizer state inspection now shows two groups, which several tests assert. Config exposes `no_wd_on_bn_bias` so the ablation (E4) can be run.

---

## ADR-004 — Per-step, not per-epoch, LR scheduling
**Date:** _TBD_ · **Sprint:** 3 · **Status:** accepted

**Context.** Cosine annealing stepped once per epoch produces a 120-step staircase rather than a smooth curve, which measurably underperforms and interacts badly with warmup.

**Decision.** All schedulers step once per optimizer step. `build_scheduler` takes `steps_per_epoch` and scales its horizons accordingly.

**Consequences.** `steps_per_epoch` must be known at scheduler construction, so the dataloaders are built before the scheduler. Changing batch size changes the total step count and therefore the schedule — noted in the config docstring.

---

## ADR-005 — `device: auto` resolved at load time
**Date:** 2026-07-31 · **Sprint:** 0 · **Status:** accepted

**Context.** `base.yaml` originally hardcoded `device: cpu` so the repo would run out of the box on a machine without a GPU. That default is exactly wrong on a machine *with* one: nothing about the config says "this is the CPU fallback," so a GPU run silently trains on CPU — a ~50x slowdown — until someone notices, potentially a whole session later.

**Decision.** `device` becomes `auto | cuda | cpu`. `"auto"` resolves at load time to `"cuda"` if `torch.cuda.is_available()` else `"cpu"`; the *resolved* value is what lands in the frozen `Config`, so `run_manifest.json` records what actually ran, never the unresolved intent. An explicit `device: cuda` with no CUDA available still raises — resolution only ever fills in for `"auto"`, it never downgrades an explicit request. `base.yaml` sets `device: auto`. The error message for the explicit-cuda-without-CUDA case now mentions CPU-only wheels, since that's the likely cause on Windows (PyPI's default `torch` wheel has no CUDA support).

**Alternatives.** Keep `device: cpu` as the safe default (rejected: silently wastes GPU time, the failure mode this exists to prevent). Require `device` to always be explicit with no `"auto"` (rejected: pushes a resolution the machine can answer onto every operator, every time).

**Consequences.** Two configs with identical explicit settings but different hardware can now resolve to different `Config.device` values — `config_hash` is unaffected since hashing happens after resolution and hardware-driven runs are expected to differ. Code that inspects `cfg.device` never sees `"auto"`.

---

## ADR-006 — `Run.finalize` takes a `status` argument
**Date:** 2026-08-01 · **Sprint:** 3 · **Status:** accepted

**Context.** `ARCHITECTURE.md` §6 freezes `finalize(summary) -> None`, but the same document's manifest schema has a `status` field with four values (`running | completed | failed | interrupted`), and Sprint 4 requires a crash to write `status: "failed"`. The frozen signature has no way to express which of the four it is.

**Decision.** Add a keyword argument with a default: `finalize(summary, status="completed")`. Every call written against the frozen signature keeps working unchanged.

**Alternatives.** Have callers reach past `finalize` to `write_manifest({"status": ..., ...})` (rejected: duplicates the end-of-run field assembly at every abnormal-exit site, and the two copies will drift). Infer status inside `finalize` from `sys.exc_info()` (rejected: spooky action at a distance for a fact the caller already knows).

**Consequences.** `Trainer.fit` labels its own exit: `completed`, `interrupted` on Ctrl-C, `failed` in an `except` that re-raises. A run directory therefore never sits at `status: "running"` after the process has gone.

### Addendum — 2026-08-01, sprint 5: that last sentence was wrong

**What happened.** Three of sprint 5's eight ablations died when the console that launched the sweep was torn down. `abl_step_lr` was killed ~30 s in with `0x40010004` (`DBG_TERMINATE_PROCESS`); the two that had not started yet exited `0xC000026B` (`STATUS_DLL_INIT_FAILED_LOGOFF`) — Python could not load its DLLs because the session was already shutting down. `runs/resnet18_cifar_20260801_061849` sat at `status: "running"` for fourteen hours afterwards.

**Why the original reasoning failed.** The consequence above holds for every exit *Python can observe*. `finally` runs on a normal return, on an exception, on `KeyboardInterrupt`. It does not run on `TerminateProcess`, on a console/logoff teardown, or on power loss — the process is destroyed without executing another instruction. Nothing the writer does can close this: at the moment the status becomes wrong, the only party who could correct it no longer exists.

**Decision.** Leave `finalize` alone; the writing side was never the problem. Move the fix to the *reading* side. `Run.write_manifest` now records `process: {pid, host}` before epoch 0, and `make_report.reaped_status` renders a `running` run whose PID is no longer live as **`killed`**.

**Conservative on purpose.** A manifest with no `process` block (anything written before this addendum) or one produced on a different host is left at `running`. Reporting a live run as dead is a worse failure than the reverse — a stale `running` is a puzzle, a false `killed` is a wrong claim in the ledger. PID reuse is likewise undetected and errs the same way; closing that needs process creation times and therefore `psutil`, which is not worth a dependency here.

**Consequences.** `status: "running"` in a manifest now means "running, or killed on another machine, or killed before this addendum" rather than "running". The table is where the question gets settled, and `killed` is a fifth rendered status that `finalize` itself can never write. Separately, `sweep.py --detach` removes the common cause: a sweep with no parent console cannot be killed by one closing.

---

## ADR-007 — `channels_last` defaults to off
**Date:** 2026-08-01 · **Sprint:** 3 · **Status:** accepted

**Context.** NHWC memory format is standard advice for AMP training, on the grounds that it feeds tensor cores without transposing. That advice is written for ImageNet-resolution tensors. Measured on this project's actual hardware (RTX 4060 Laptop, torch 2.11.0+cu128), at 32x32 with CIFAR channel counts it is a large *pessimization*:

| model | channels_last | AMP | ms/step | s/epoch |
|---|---|---|---|---|
| SimpleCNN | on | on | 37.3 | 13.1 |
| SimpleCNN | off | on | 12.8 | 4.5 |
| ResNet-18-CIFAR | on | on | 102.4 | 36.0 |
| ResNet-18-CIFAR | off | on | 35.2 | 12.4 |

The GPU itself is healthy — an fp16 4096³ matmul sustains 28.9 TFLOPS at 2580 MHz with no throttling — so this is a kernel-selection effect at small spatial dimensions, not thermal or driver trouble. AMP is a clear win in both formats (1.4x); `cudnn.benchmark` is within noise.

**Decision.** `train.channels_last` defaults to `false` in `base.yaml` and in `TrainConfig`.

**Alternatives.** Keep it on to match the conventional recipe (rejected: 3x the wall clock on every run in the project, which is the difference between the 94% recipe fitting in the M3 45-minute budget and missing it by double). Remove the field (rejected: it is genuinely a win on other GPUs and at other resolutions, and the ablation is worth re-running when the hardware changes).

**Consequences.** Every epoch budget in `PRD.md` §5 and `EXPERIMENTS.md` is measured with NHWC off. Anyone porting this to a desktop GPU or a larger input size should re-measure before assuming the default still holds — the field stayed configurable precisely so that costs one line.

**Sprint 4 addendum.** The table above is **per-step** — milliseconds per optimizer step, no validation, checkpointing or logging. It must not be compared against the run-level tables in `PRD.md` §5 and `EXPERIMENTS.md`; both now say which kind they are.

An earlier version of this addendum claimed the run-level penalty was 2.2× rather than 2.9%, and attributed the gap to format-independent per-epoch work diluting the ratio. **That was wrong and is withdrawn.** It rested on a 6-epoch SimpleCNN A/B in which the `channels_last: false` arm never left its startup ramp, so the denominator was inflated. SimpleCNN was also the wrong model to measure on: its epoch time is not reproducible across sessions (see `EXPERIMENTS.md`). The A/B was re-run on ResNet-18, where both arms are GPU-bound — see the run-level confirmation in `EXPERIMENTS.md` § Sprint 4.

---

## ADR-008 — `keep_last_n_checkpoints` is inert under the two-file scheme
**Date:** 2026-08-01 · **Sprint:** 3 · **Status:** accepted

**Context.** Sprint 3 asks for `last.pth` written every `save_every` epochs, `best.pth` on improvement, and pruning to `keep_last_n_checkpoints`. The first two rotate in place, so after them there is nothing left to prune — the field only becomes meaningful if the trainer also writes per-epoch snapshots (`epoch_0042.pth`), which `ARCHITECTURE.md` §6's run-directory listing does not include.

**Decision.** Write only `last.pth` and `best.pth`. `keep_last_n_checkpoints` stays in the config, unused, and this entry is the record that it is unused — a config field that silently does nothing is the exact failure mode `_validate`'s batch-aug coherence checks exist to prevent, so it should not go unrecorded.

**Alternatives.** Write per-epoch snapshots and prune them (rejected for now: a ResNet-18 checkpoint is ~45 MB and this buys a rollback capability nothing in Sprints 3–8 asks for). Delete the field (rejected: it is in the frozen `ARCHITECTURE.md` §1 contract, so removing it costs more than leaving it).

**Consequences.** Setting `keep_last_n_checkpoints` has no effect today. Revive it the first time a run is long enough that losing the last N epochs actually hurts — Sprint 7's 300-epoch runs are the plausible trigger. At that point it is roughly ten lines in `Trainer._save`.

---

## ADR-009 — Ctrl-C does not write a checkpoint
**Date:** 2026-08-01 · **Sprint:** 4 · **Status:** accepted

**Context.** Sprint 3's `KeyboardInterrupt` handler called `self._save("last.pth", epoch)` on the way out, so an interrupt during epoch 50 wrote the *partially trained* epoch-50 weights under the label "epoch 50 complete". Resuming that checkpoint started at epoch 51: epoch 50 was silently skipped, the run reported 120 epochs having trained 119, and the scheduler — restored mid-epoch-50 but then given only 69 epochs of steps — never reached `min_learning_rate`. Sprint 4's gate ("kill a run with Ctrl-C and confirm `last.pth` resumes cleanly") is exactly the check that surfaced it.

A checkpoint written mid-epoch is internally inconsistent in a way no epoch number can describe: the model has seen part of an epoch, the scheduler and RNG sit at an arbitrary step, and the epoch counter can only round one way or the other.

**Decision.** The interrupt handler saves nothing. The `last.pth` already on disk — written at an epoch boundary, with model, optimizer, scheduler, scaler and RNG all agreeing — is left as the resume point, and the console line says which epoch that is and where a resume will continue from. `fit()` tracks `completed` (last epoch trained and logged) separately from `checkpointed` (last epoch actually written), and `summary["epochs_completed"]` reports the former.

**Alternatives.** Save under `epoch - 1` (rejected: the weights still contain partial epoch-`epoch` progress, so the checkpoint stays inconsistent — it just relabels the inconsistency). Checkpoint mid-epoch with a step offset and resume the loader mid-stream (rejected: needs a resumable sampler and buys nothing that re-running one epoch doesn't).

**Consequences.** Ctrl-C now costs up to `save_every` epochs of work instead of producing a subtly wrong resume — with the default `save_every: 1`, at most the partial epoch that was going to be discarded anyway. A run interrupted inside epoch 0 has no `last.pth` at all, which is correct: nothing completed. `epochs_completed` on an interrupted run is now one lower than sprint 3 would have reported for the same interrupt.

---

## ADR-010 — `EXPERIMENTS.md` is generated between sentinels, and runs are retracted rather than deleted
**Date:** 2026-08-01 · **Sprint:** 4 · **Status:** accepted

**Context.** `EXPERIMENTS.md` is two documents sharing one file: tables produced from `run_manifest.json`, and hand-written findings, hypotheses and hardware notes that are the more valuable half. Sprint 4 shipped `make_report.py` as append-only, on the theory that appending can never destroy prose. That holds right up until a generated number is *wrong* — and then the only available tool is a text editor. It happened twice in one sprint: a 6-epoch `channels_last` A/B whose control arm never left its startup ramp, and the finding written on top of it. Both had to be hand-deleted from the region the generator owns, which is precisely the coupling append-only was supposed to prevent.

Sprint 8's C8.5 already specifies the right shape — "regenerate the table with `make_report.py --all`" — it was simply implemented as append. With nine ablation runs about to land in Sprint 5, that is nine more append-and-hand-edit cycles.

**Decision.** Two changes.

1. **Sentinels.** Each `--section <id>` owns a region delimited by `<!-- BEGIN GENERATED: id -->` / `<!-- END GENERATED: id -->`. Regeneration replaces the region's contents and preserves every byte outside it; an unknown section is appended as a new region. Sections are independent, and the id is stable and separate from the heading, so rewording a title cannot orphan a region. An unmatched marker raises rather than guessing where the region stops — guessing is how a generator eats the paragraph underneath it.
2. **Retraction is data, not deletion.** A run is withdrawn by adding `"retracted": {"reason": "...", "date": "..."}` to its `run_manifest.json`. The row stays in the table, struck through, with the reason rendered beside it.

**Alternatives.** Keep tables in a separate generated file and `include` it (rejected: markdown has no include, and splitting the ledger puts the numbers and the reasoning about them in two places — the failure mode this document exists to avoid). Regenerate the whole file from a template (rejected: makes the prose a generator input, so every finding would have to live in a Python string or a sidecar). Delete retracted runs (rejected: a ledger whose job is being checkable cannot have rows disappear; a reader who remembers a number needs to find out what happened to it, not fail to find it).

**Consequences.** Prose in `EXPERIMENTS.md` is safe by construction, so `--all` is now genuinely re-runnable and Sprint 5's sweep can regenerate its section after every job. Each section carries its own figure (`comparison_{section}.png`) so regenerating one cannot overwrite another's. Retracted runs keep their directory and their row forever, which is the point — the two Sprint 4 retractions above are the worked example. The markers are visible in raw markdown and invisible when rendered.

---

## ADR-011 — Sweep repeats are interleaved across configs, not nested inside them
**Date:** 2026-08-02 · **Sprint:** 6 · **Status:** accepted

**Context.** Sprint 6's four-config table ran each config's two repeats back to back:
`E9, E9, E10, E10, E13, E13, E15, E15`. That ordering makes the within-config spread a
*within-minutes* number — the two runs of a pair share a machine state, a thermal state and a
position in the evening. Pooled across the four pairs it came to 0.083, against 0.209 from step 0's
seed-variance runs, which spanned seven hours and a power-profile change.

The 0.083 cannot be used to judge E9 against E15, because those two ran seven hours apart and the
estimate was constructed to exclude exactly that. The design nests the time variable inside the
config variable, so config and sweep position are confounded and the cheaper noise estimate is the
one that cannot answer the question. This is not a small effect here: whether Mixup or RandAugment
beats plain Cutout is 4.5σ under one estimate and 1.8σ under the other, and the sweep produced no
way to choose between them.

**Decision.** A sweep with repeats runs one full block of every config, then repeats the block:
`E9, E10, E13, E15, E9, E10, E13, E15`. Identical total compute, no code change — it is purely a
question of what `sweep.py --dir` is pointed at and how the configs sort.

**Alternatives.** Randomise run order (rejected: with n=8 a random draw can still nest badly, and
a fixed block is reproducible and inspectable). Hold conditions constant instead (rejected: a
seven-hour sweep on a laptop cannot hold thermal state constant, and pretending otherwise is how the
confound got missed). Estimate σ from dedicated repeat runs only (rejected: that is step 0, and it
costs extra GPU hours to measure something the table can measure for free if ordered correctly).

**Consequences.** Within-config spread becomes an estimate of the noise that actually applies to
between-config comparisons, at zero extra cost. The `Min` and `s/epoch` columns stay
non-comparable within a sweep either way — `EXPERIMENTS.md` already says so, and interleaving does
not fix thermal drift, it only stops that drift from being aliased onto one config. Sprint 6's own
table keeps the nested ordering it was run with; its σ ambiguity is recorded as open rather than
retro-fitted.

---

## ADR-012 — Ablation sweeps run with `cudnn_benchmark: false`
**Date:** 2026-08-03 · **Sprint:** 8 · **Status:** accepted

**Context.** Sprint 6 spent its most valuable measurement on a question it could not close. Step 0
measured run-to-run spread of **0.21** across clean runs of one recipe, with two runs at an
*identical seed and config hash* differing by **0.26**. The four-config ablation table then produced
its own, incompatible estimate of **0.083**, and `EXPERIMENTS.md` records the disagreement as open
because neither could be trusted: 0.083 came from back-to-back repeats and describes minutes, 0.209
rests on two degrees of freedom and one high run. The practical cost was concrete — whether Mixup or
RandAugment beats plain Cutout is 4.5σ under one estimate and 1.8σ under the other, and the sprint
could not say which.

C8.1b then localised the cause. `deterministic: true` switches off two things at once, and separating
them showed that **`cudnn.benchmark = False` alone is sufficient for bit-identical runs** —
permitting nondeterministic algorithms changed nothing for this workload. The variance was never
inherent to GPU training here; it was kernel selection by wall-clock timing, and the reduction order
that follows the kernel.

If the noise can be removed, there is no reason to keep estimating it.

**Decision.** Ablation sweeps set `repro.cudnn_benchmark: false`. Two runs of the same config at the
same seed are then bit-identical, so run-to-run noise at fixed seed is **zero** and any difference
between two configs is the effect plus *seed* variance only — cleanly separated, and seed variance is
the quantity that was actually of interest.

**Cost — measured, not assumed.** Three modes, 10 epochs each, run in two interleaved blocks per
ADR-011, medians over epochs 4–10:

| Mode | Mean median s/epoch | vs default |
|---|---|---|
| `cudnn_benchmark: true` (default) | 13.99 | — |
| **`cudnn_benchmark: false`** | **14.07** | **+0.6%** |
| `deterministic: true` | 15.93 | +13.8% |

**+0.6% is indistinguishable from free** and should be read that way rather than as a precise figure:
the block-to-block spread *within* the default mode was 13.01 to 14.98, wider than the gap being
measured. The honest claim is that turning benchmark off costs nothing detectable, while full
determinism costs about 14%. An earlier estimate of ~25% for determinism is superseded — it compared
runs from different sessions on a drifting thermal state, which is the confound this measurement was
built to avoid.

**Alternatives.** `deterministic: true` for sweeps (rejected: ~14% for no further benefit, since
benchmark alone already gives bit-identical runs on this workload — and `use_deterministic_algorithms`
raises on ops with no deterministic kernel, which is a portability tax for nothing). Keep the default
and average over more seeds (rejected: spends GPU hours estimating noise that can simply be removed;
Sprint 6's σ work cost about 2.5 h and still did not settle the question). Hold machine state constant
across a sweep (rejected: unachievable on a laptop over seven hours, and unnecessary — state cannot
reach the result once kernel selection is fixed).

**Consequences.** Repeats within a sweep must now **vary the seed** to measure anything; two runs of
one config at one seed are the same run and tell you nothing. ADR-011's interleaved blocks are
retained but their motivation narrows: they no longer rescue an accuracy comparison, they keep the
timing columns and thermal exposure fair. Sprint 6's σ ambiguity stands as recorded — it is history,
and this entry does not retro-fit it. Configs that need the default's throughput can keep it; the
recommendation binds sweeps, where comparability is the point, not one-off training runs.
