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
**Date:** _TBD_ · **Sprint:** 1 · **Status:** accepted

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
