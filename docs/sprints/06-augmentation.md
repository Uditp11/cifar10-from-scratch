# Sprint 6 — Advanced augmentation

**Goal:** Cutout, Mixup, CutMix, and RandAugment implemented from scratch, each ablated, pushing the model to **≥95.0%**.

**Depends on:** Sprint 5. **Blocks:** Sprints 7–8.
**Estimated effort:** 1 session of coding + a long training block.

---

## Why augmentation is the right lever here

At the end of Sprint 5, train accuracy is near 100% and val is ~95%. The model has memorized the training set; the gap is pure overfitting. More capacity won't help, and more epochs won't help. Regularization that manufactures new training signal is the only thing that moves the number.

Because Sprint 1 left the config plumbing and the `MixedBatch` contract in place, this sprint touches only two files plus configs. Nothing in the trainer changes. That's the payoff for the earlier discipline.

## Components

### C6.1 — Cutout

`src/cifar/data/transforms.py` — fill in the Sprint 1 stub.

Zero out `n_holes` square regions of side `length`, centered at uniformly random positions, clipped at image boundaries (so a hole near an edge is partial — this is correct and matches the paper). Applied **after** normalization, so zero equals the dataset mean pixel.

Standard CIFAR-10 setting: 1 hole, 16 px — a quarter of a 32×32 image. That feels aggressive and is nonetheless right.

### C6.2 — Mixup

`src/cifar/data/batch_aug.py`

```
lam = Beta(alpha, alpha).sample()
perm = randperm(batch_size, device=x.device)
mixed_x = lam * x + (1 - lam) * x[perm]
return MixedBatch(mixed_x, y, y[perm], lam)
```

Operates on the collated batch, on GPU, before the forward pass. `alpha=0.2` is the usual CIFAR value; higher alpha means more aggressive mixing.

### C6.3 — CutMix

Same structure, but instead of blending pixel values, paste a rectangular patch from the permuted batch:

```
r = sqrt(1 - lam)
cut_w, cut_h = W * r, H * r
```
random center, clip to bounds, then **recompute `lam` from the actual pasted area** — the clipped box rarely has exactly the intended area, and using the nominal `lam` in the loss is a real (and commonly copied) bug.

`alpha=1.0` for CIFAR.

### C6.4 — Dispatch

`apply_batch_aug(x, y, cfg)`:
- With probability `1 - cfg.aug.batch_aug_prob`, return unmixed (`lam=1.0`, `targets_a is targets_b`).
- Otherwise pick mixup or cutmix — if both alphas > 0, choose uniformly at random per batch; if only one is > 0, use it.
- Always return a `MixedBatch`, so the trainer never branches.

### C6.5 — RandAugment

`src/cifar/data/transforms.py`

Implement from scratch: sample `n` operations uniformly from the op list, each at magnitude `m` on a 0–30 scale. Ops: identity, autocontrast, equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x/y, translate-x/y. Use `PIL.ImageOps` / `ImageEnhance` — implementing the image ops themselves from raw pixels is not the interesting part and burns a session.

Applies **before** `ToTensor()`, on the PIL image.

Time-box this to one session. If it overruns, fall back to `torchvision.transforms.RandAugment` and write an ADR entry explaining the tradeoff. That's an acceptable outcome — the from-scratch principle is about the model and the training loop first.

### C6.6 — Verify visually

Extend `scripts/inspect_data.py` with panels for each augmentation, including a mixup/cutmix batch with its `lam` printed. Genuinely look at the output. CutMix box-clipping bugs are obvious in an image grid and nearly invisible in a loss curve.

---

## Ablations

**Redesigned 2026-08-01, after Sprint 5.** The original table listed eight single-seed configs expecting +0.3 to +0.6 deltas. Sprint 5 then showed four of its nine runs (E4, E3, F, E2) spanning just 0.48 points — at or below the single-seed noise floor. Eight more single-seed runs would have produced eight numbers that cannot be distinguished from one another: six hours of GPU for an uninterpretable table. Fewer configs, more seeds, same budget.

### Step 0 — measure σ first (~1.5 h)

Before any augmentation run, train the **unchanged Sprint 5 F recipe at seeds 42, 1337, 2024**, 120 epochs each. This is Sprint 7's C7.3 seed-variance work pulled forward, because its output is the instrument every row below depends on.

Report **mean ± std**. Then:

- **σ ≲ 0.10** — Sprint 5's E4/E3/E2 spread was real, and single-seed rows here are interpretable. Note that E4 contradicting ADR-003 then becomes a finding that needs acting on.
- **σ ≈ 0.20–0.25** — as suspected. Sprint 5's top four were always noise, and nothing below is trustworthy at one seed. Run the table at 2 seeds and report mean ± std.
- **σ ≳ 0.30** — even 2 seeds won't separate 0.3-point effects. Say so plainly, report the table as indicative only, and let Sprint 7's stacked configuration carry the accuracy claim.

Whatever σ is, it retroactively annotates every Sprint 5 row. That single number is worth more than any ablation in this sprint.

### The table (~6 h at 2 seeds, 200 epochs each)

Four configs, not eight. The dropped rows — mixup alone, cutmix alone, cutout+cutmix, the cutout-length sweep — were all near-duplicates of something retained, and none was separable from its neighbour at the deltas involved.

| ID | Change from Sprint 5 recipe | Expected val acc |
|---|---|---|
| E9 | baseline at 200 ep (no new aug) | 95.0–95.4% |
| E10 | + Cutout(1, 16) | 95.4–95.9% |
| E13 | + Cutout + Mixup α=0.2 | 95.5–96.1% |
| E15 | + RandAugment(2, 9) + Cutout | 95.6–96.2% |

Mixup and cutmix need the longer schedule to pay off — at 120 epochs they can look *worse* than baseline, which is the most common reason people wrongly conclude mixup doesn't work.

**Calibrate your expectations against E8.** Crop + flip alone bought 5.80 points, while Sprint 5's other seven ablations spanned 2.12 points in total. Everything in this table is a tenths-of-a-point game on top of that. The write-up should say so — "we stacked four more augmentations and gained half a point over the two-line baseline" is a more honest and more interesting result than a table implying each one mattered.

Pick the winner as the Sprint 7 starting point. If two are within one σ of each other, prefer the simpler one and say so explicitly — that's a defensible engineering judgment, and it is worth more than a spurious ranking.

---

## Acceptance criteria

```python
def test_cutout_zeroes_expected_area():
    x = torch.ones(3, 32, 32)
    out = Cutout(n_holes=1, length=16)(x)
    zeroed = (out == 0).float().mean().item() * 32 * 32
    assert 64 <= zeroed <= 256    # 16x16=256 max; 8x8=64 min when centered at a corner

def test_cutout_disabled_is_identity():
    x = torch.randn(3, 32, 32)
    assert torch.equal(build_train_transform(cfg(cutout_holes=0)).cutout(x), x)

def test_mixup_lam_in_range_and_shape_preserved():
    b = apply_batch_aug(x, y, cfg(mixup_alpha=0.2, batch_aug_prob=1.0))
    assert 0.0 <= b.lam <= 1.0 and b.inputs.shape == x.shape

def test_mixup_output_is_convex_combination():
    b = apply_batch_aug(x, y, cfg(mixup_alpha=0.2, batch_aug_prob=1.0))
    assert b.inputs.min() >= x.min() - 1e-5
    assert b.inputs.max() <= x.max() + 1e-5

def test_cutmix_lam_matches_actual_pasted_area():
    """The clipped-box bug. lam must be recomputed from the real box."""
    b = apply_batch_aug(x, y, cfg(cutmix_alpha=1.0, batch_aug_prob=1.0))
    changed = (b.inputs != x).float().mean(dim=(1, 2, 3)).mean().item()
    assert abs((1 - b.lam) - changed) < 0.05

def test_batch_aug_prob_zero_is_passthrough():
    b = apply_batch_aug(x, y, cfg(mixup_alpha=0.2, batch_aug_prob=0.0))
    assert b.lam == 1.0 and torch.equal(b.inputs, x) and torch.equal(b.targets_a, b.targets_b)

def test_mixed_criterion_equals_plain_ce_when_unmixed():
    b = MixedBatch(x, y, y, 1.0)
    assert mixed_criterion(ce, logits, b) == pytest.approx(ce(logits, y).item())

def test_randaugment_preserves_size_and_mode():
    img = Image.new("RGB", (32, 32))
    out = RandAugment(2, 9)(img)
    assert out.size == (32, 32) and out.mode == "RGB"

def test_no_augmentation_at_eval_time():
    """Regression guard: the eval transform must contain no stochastic op."""
```

## Gate

Best configuration reaches **≥ 95.5%** val accuracy (≈95.0% test), reported as **mean ± std** across seeds — a bare single-seed number no longer clears this gate. σ from Step 0 is recorded in `EXPERIMENTS.md` and annotates every Sprint 5 row retroactively. The visual inspection grid confirms each augmentation looks right.

---

## Implementation prompt

> Read `CLAUDE.md`, `docs/ARCHITECTURE.md` section 3, and `docs/sprints/06-augmentation.md`.
>
> Fill in the Sprint 1 stubs: `Cutout`, `RandAugment`, and the real `apply_batch_aug` with mixup and cutmix. Nothing in `Trainer` should need to change — if you find yourself editing the trainer, the `MixedBatch` contract is being violated.
>
> Two specific correctness traps: Cutout goes *after* normalization; and CutMix must recompute `lam` from the actually-pasted (clipped) box area rather than the sampled value, or the loss weighting is wrong.
>
> Time-box RandAugment to one session; if it overruns, fall back to torchvision's and write an ADR entry.
>
> Then write the eight ablation configs, run them at 200 epochs, extend `scripts/inspect_data.py` with a visual panel per augmentation, and report which combination wins. Gate is ≥95.0%.
