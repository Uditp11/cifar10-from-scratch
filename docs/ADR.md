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
