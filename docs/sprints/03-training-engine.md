# Sprint 3 — Training engine

**Goal:** The first end-to-end training run. `python scripts/train.py --config configs/simplecnn.yaml` trains, validates, checkpoints, resumes, and reaches **≥85% val accuracy**.

**Depends on:** Sprints 0, 1, 2. **Blocks:** everything after.
**Estimated effort:** 2 sessions. This is the biggest sprint.

---

## Milestone

This is the sprint where the project becomes real. Gate M4 from the PRD lands here.

## Components

### C3.1 — Meters

`src/cifar/tracking/meters.py` — `AverageMeter` and `accuracy(logits, targets, topk)` returning percentages. Also a small `Timer` context manager for epoch timing.

`accuracy` with `topk=(1, 5)` must handle the case where a batch has fewer than 5 classes present — it operates on logits, so this is fine, but write the test.

### C3.2 — Criterion

`src/cifar/engine/criterion.py`

- `build_criterion(cfg)` → `nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)`.
- `mixed_criterion(criterion, logits, batch: MixedBatch)` → `lam * L(logits, a) + (1-lam) * L(logits, b)`. With Sprint 1's stub returning `lam=1.0` and `a is b`, this is exactly plain CE today and becomes mixup-aware in Sprint 6 with no trainer change.

### C3.3 — Optimizer & scheduler factories

`src/cifar/engine/optim.py`

**Param groups.** When `cfg.optim.no_wd_on_bn_bias`, split parameters into two groups: everything with `p.ndim <= 1` (BN weights, BN biases, linear/conv biases) gets `weight_decay=0`; everything else gets `cfg.optim.weight_decay`. Applying weight decay to BN parameters actively hurts — this is worth roughly 0.2–0.4% and costs ten lines.

**Scheduler.** Per-step, not per-epoch. Compose with `SequentialLR`:

```
LinearLR(start_factor=cfg.optim.warmup_start_factor, total_iters=warmup_epochs * steps_per_epoch)
  ──► CosineAnnealingLR(T_max=(epochs - warmup_epochs) * steps_per_epoch,
                        eta_min=cfg.optim.min_learning_rate)
```

Also support `step` (MultiStepLR at `step_milestones × steps_per_epoch`), `onecycle`, and `constant`. When `warmup_epochs == 0`, skip `SequentialLR` and return the main scheduler directly — `SequentialLR` with an empty first phase behaves badly.

### C3.4 — Checkpointing

`src/cifar/engine/checkpoint.py`

Save: model state, optimizer state, scheduler state, `GradScaler` state, EMA state (None for now), epoch, best metric, config hash, and **RNG state** (`random`, `numpy`, `torch`, `torch.cuda`). The RNG state is what makes resume produce the same trajectory as an uninterrupted run.

Write `last.pth` every `cfg.train.save_every` epochs and `best.pth` whenever val accuracy improves. Prune to `keep_last_n_checkpoints`. Write atomically: save to `.tmp` then `os.replace`, so a crash mid-save never corrupts the only good checkpoint.

`load_checkpoint` warns loudly if the checkpoint's `cfg_hash` differs from the current config's — resuming into a changed config is a real scenario but should never be silent.

### C3.5 — Trainer

`src/cifar/engine/trainer.py`

```python
class Trainer:
    def __init__(self, model, loaders: tuple[DataLoader, DataLoader], cfg, run): ...
```

The 2-tuple is a structural guarantee that the trainer cannot see the test set.

`train_one_epoch` per-batch flow:

```
x, y = batch  →  .to(device, non_blocking=True)
[channels_last]
batch = apply_batch_aug(x, y, cfg)            # no-op until sprint 6
with autocast(enabled=cfg.train.amp):
    logits = model(batch.inputs)
    loss   = mixed_criterion(criterion, logits, batch)
scaler.scale(loss).backward()
[if grad_clip_norm: scaler.unscale_(opt); clip_grad_norm_]
scaler.step(opt); scaler.update()
opt.zero_grad(set_to_none=True)
scheduler.step()                              # PER STEP
[ema.update(model)]                           # after the step — sprint 7
meters.update(...)
```

`fit()`:
- loop epochs, train, validate every `cfg.train.eval_every`
- track best val acc and best epoch
- log via `Logger` (Sprint 4 — for now, a `rich` progress bar and a plain per-epoch line is fine; keep the call sites so Sprint 4 swaps the implementation)
- checkpoint
- handle `KeyboardInterrupt` by saving `last.pth` and exiting cleanly
- return a summary dict

`resume_from(path)` restores everything including RNG state and continues from `epoch + 1`.

### C3.6 — Evaluator

`src/cifar/engine/evaluator.py` — `evaluate(model, loader, cfg, tta=False)` → `EvalResult` with loss, top-1, top-5, per-class accuracy, and a 10×10 confusion matrix. Under `torch.no_grad()` and `model.eval()`, restoring the previous training mode on exit. TTA hook exists but is Sprint 7; `tta=True` may raise `NotImplementedError` for now.

### C3.7 — Scripts

`scripts/train.py`: parse `--config`, `--override` (repeatable), `--resume`, `--device`; load config; `seed_everything`; build everything; run; print the run dir as the final stdout line.

`scripts/eval.py`: load a run's config and checkpoint, build the **test** loader, evaluate, print a report. This is the only place in the codebase that touches the test set.

---

## Acceptance criteria

`tests/test_engine.py`:

```python
def test_scheduler_steps_per_step_not_per_epoch():
    sched = build_scheduler(opt, cfg(epochs=10, warmup_epochs=0), steps_per_epoch=100)
    lr0 = opt.param_groups[0]["lr"]
    for _ in range(50): opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] < lr0          # decayed within epoch 1

def test_warmup_reaches_base_lr_then_decays():
    cfg_ = cfg(epochs=10, warmup_epochs=2, learning_rate=0.1, warmup_start_factor=0.1)
    sched = build_scheduler(opt, cfg_, steps_per_epoch=10)
    lrs = [step_and_get_lr(opt, sched) for _ in range(100)]
    assert lrs[0] == pytest.approx(0.01, rel=0.1)
    assert max(lrs) == pytest.approx(0.1, rel=0.01)
    assert lrs[-1] < 0.001

def test_no_weight_decay_on_bn_and_bias():
    opt = build_optimizer(build_model(cfg), cfg)
    assert len(opt.param_groups) == 2
    nodecay = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    assert all(p.ndim <= 1 for p in nodecay["params"])

def test_trainer_cannot_access_test_set():
    """Structural: __init__ accepts a 2-tuple."""
    with pytest.raises(TypeError | ValueError):
        Trainer(model, (train_dl, val_dl, test_dl), cfg, run)

def test_one_epoch_on_tiny_subset_runs():
    t = Trainer(model, tiny_loaders(n=64), cfg(epochs=1, amp=False), run)
    m = t.train_one_epoch(0)
    assert 0 < m["train_loss"] < 10 and 0 <= m["train_acc"] <= 100

def test_checkpoint_roundtrip_is_exact():
    save_checkpoint(p, model=m, optimizer=o, scheduler=s, scaler=sc,
                    ema=None, epoch=5, best_metric=0.9, cfg=cfg, rng_state=rng())
    m2 = build_model(cfg)
    meta = load_checkpoint(p, model=m2, optimizer=o2, scheduler=s2, scaler=sc2)
    assert meta["epoch"] == 5
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)

def test_resume_matches_uninterrupted_run():
    """Train 4 epochs straight; train 2 + resume 2. Final val acc must match
    within 0.5% on a tiny subset."""

def test_evaluate_returns_full_result():
    r = evaluate(model, val_loader, cfg)
    assert 0 <= r.top1 <= 100 and r.confusion_matrix.shape == (10, 10)
    assert r.confusion_matrix.sum() == r.num_samples
    assert len(r.per_class_acc) == 10

def test_amp_and_fp32_agree_on_a_step():
    """Loss after one step differs by < 1e-2 between amp and fp32."""
```

## Gate — the big one

```bash
python scripts/train.py --config configs/simplecnn.yaml --override train.epochs=30
```

must produce **val accuracy ≥ 85%**, a complete run directory, and a resumable `last.pth`. Then:

```bash
python scripts/eval.py --run runs/<that_run> --checkpoint best
```

reports test accuracy within ~1% of val accuracy. A large val/test gap at this stage means a split or transform bug — stop and find it.

---

## Debug checklist if 85% isn't reached

Work down this list; the cause is nearly always in the first four.

1. Loss starts near `ln(10) ≈ 2.303`? If it starts much higher, init is wrong.
2. Loss decreasing at all in epoch 1? If flat, LR is wrong by an order of magnitude, or the optimizer isn't stepping.
3. Train accuracy climbing far past val? That's normal overfitting; below 85% val with 99% train means you need the Sprint 5/6 regularization, not a bug fix.
4. Train and val both stuck near 10%? Labels are misaligned with images, or the model output is being permuted.
5. `model.train()` / `model.eval()` set correctly around each loop?
6. `opt.zero_grad()` actually called each step?
7. Normalization applied to val the same as train?

---

## Implementation prompt

> Read `CLAUDE.md`, `docs/ARCHITECTURE.md` sections 1, 3, 4, 5, and `docs/sprints/03-training-engine.md`.
>
> Implement C3.1 through C3.7 in order. Non-obvious requirements, in priority order: the scheduler steps **once per optimizer step**, not per epoch; BN and bias parameters go in a `weight_decay=0` param group; checkpoints include RNG state so resume is faithful; `Trainer.__init__` takes a `(train, val)` 2-tuple and must never be able to reach the test loader.
>
> Keep the logging call sites clean but implement them minimally with `rich` — Sprint 4 replaces the implementation behind the same calls.
>
> When the code is done, run the gate: `python scripts/train.py --config configs/simplecnn.yaml --override train.epochs=30`. Report val accuracy, seconds per epoch, and the run directory. If val accuracy is below 85%, work the debug checklist in this file before changing hyperparameters.
