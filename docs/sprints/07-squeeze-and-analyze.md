# Sprint 7 — Squeeze & analyze

**Goal:** Final accuracy push via EMA and TTA to the **95.5% stretch target**, plus the analysis that turns a number into an understanding — error analysis, calibration, and a model card.

**Depends on:** Sprint 6. **Blocks:** Sprint 8.
**Estimated effort:** 1 session + a long final training block.

---

## Two halves

The first half squeezes the last few tenths. The second half is what separates a training script from a project: knowing *which* images the model gets wrong, and whether its confidence means anything.

## Part A — the squeeze

### C7.1 — EMA

`src/cifar/engine/ema.py`

Maintain a shadow copy of the model updated as `shadow = decay * shadow + (1 - decay) * model` after every optimizer step. Three details that are easy to get wrong:

- **Buffers too.** BN running statistics must be copied (usually directly, not EMA'd — copying is simpler and works well).
- **Warmup.** With `decay=0.999` and ~105 k total steps (351 steps/epoch × 300), a cold-started shadow takes thousands of steps to become useful. Either use `cfg.train.ema_warmup_epochs` before starting updates, or the standard ramp `decay_t = min(decay, (1 + t) / (10 + t))`. Implement the ramp; it's less to tune.
- **Checkpointing.** EMA state goes in the checkpoint, and `eval.py --ema` evaluates the shadow.

Evaluate both the raw and the EMA weights each epoch and log both. EMA usually wins by 0.1–0.3% and is much smoother epoch to epoch.

### C7.2 — TTA

`src/cifar/engine/evaluator.py` — implement `tta=True`: average softmax probabilities over `{identity, horizontal flip}`. Averaging probabilities, not logits.

Optionally add 5-crop TTA from padded images. Report both, then decide: TTA doubles inference cost for ~0.1–0.2%, which is a legitimate tradeoff to state explicitly rather than silently take.

### C7.3 — Final long run

Best Sprint 6 config + EMA, 300 epochs, evaluated with and without TTA. ~60–90 min. This produces the headline number.

Also run it at three seeds (42, 1337, 2024) and report **mean ± std**. A single-seed number on CIFAR-10 carries roughly ±0.2% of noise; reporting the spread is the honest way to state a result, and reviewers notice.

## Part B — the analysis

### C7.4 — Error analysis

`scripts/analyze_errors.py`:

- Confusion matrix, row-normalized. On CIFAR-10 the dominant confusions are almost always cat↔dog and, further behind, automobile↔truck, bird↔airplane, deer↔horse. Confirm and report yours.
- The 50 highest-loss test images in a labeled grid. Expect a mix of genuinely hard images and a few plainly mislabeled ones — CIFAR-10 has known label noise. Finding some is a good sign that the analysis is real.
- Per-class precision, recall, F1 as a markdown table.
- Accuracy vs. prediction confidence, bucketed.

### C7.5 — Calibration

- Reliability diagram: bucket predictions by confidence, plot accuracy vs confidence against the diagonal.
- Expected Calibration Error (ECE), 15 bins.
- Report ECE with and without label smoothing. Label smoothing typically improves calibration substantially even when its accuracy effect is small — that's a genuinely interesting result and most CIFAR write-ups miss it.
- Optional: temperature scaling fitted **on the val set** (never on test), reporting post-scaling ECE.

### C7.6 — Model card

`docs/MODEL_CARD.md`: architecture and parameter count, training data and split, full recipe, results table (val/test, per-class, seed spread), compute cost, known failure modes from C7.4, calibration numbers, and limitations — 32×32 only, 10 classes, no distribution-shift robustness testing, evaluated on a benchmark with known label noise.

---

## Acceptance criteria

```python
def test_ema_tracks_and_lags_model():
    ema = ModelEMA(model, decay=0.9)
    before = clone_params(ema.module)
    perturb(model); ema.update(model)
    assert not params_equal(ema.module, before)      # it moved
    assert not params_equal(ema.module, model)       # but it lags

def test_ema_decay_zero_copies_model_exactly():
    ema = ModelEMA(model, decay=0.0); perturb(model); ema.update(model)
    assert params_equal(ema.module, model)

def test_ema_copies_bn_buffers():
    assert torch.equal(ema.module.bn1.running_mean, model.bn1.running_mean)

def test_ema_survives_checkpoint_roundtrip():
    save_checkpoint(p, ..., ema=ema); ema2 = ModelEMA(build_model(cfg), 0.999)
    load_checkpoint(p, ema=ema2)
    assert params_equal(ema.module, ema2.module)

def test_tta_averages_probabilities_not_logits():
    """A symmetric-under-flip input must give identical results with and
    without TTA; that only holds if probabilities are averaged correctly."""

def test_tta_never_hurts_on_a_trained_model():
    assert evaluate(m, loader, cfg, tta=True).top1 >= evaluate(m, loader, cfg).top1 - 0.5

def test_ece_of_perfectly_calibrated_predictions_is_near_zero():
    assert expected_calibration_error(synthetic_calibrated_probs(), labels) < 0.02

def test_confusion_matrix_rows_sum_to_class_counts():
    r = evaluate(model, loader, cfg)
    assert (r.confusion_matrix.sum(axis=1) == class_counts(loader)).all()
```

## Gate

- Final val accuracy **≥ 95.5%** (stretch; ≥95.0% is an acceptable landing if documented with a hypothesis).
- Three-seed result reported as mean ± std.
- `docs/MODEL_CARD.md` complete.
- Error analysis figures and the reliability diagram in `docs/figures/`.
- Test accuracy reported **once**, at the end, from `eval.py`.

---

## Implementation prompt

> Read `CLAUDE.md`, `docs/ARCHITECTURE.md` section 5, and `docs/sprints/07-squeeze-and-analyze.md`.
>
> Part A: implement `ModelEMA` (with the `(1+t)/(10+t)` decay ramp, BN buffers copied directly, and full checkpoint support) and TTA in the evaluator (averaging *probabilities*). Wire EMA into `Trainer` so both raw and EMA val accuracy are logged each epoch.
>
> Then run the final configuration at 300 epochs across seeds 42, 1337, and 2024, and report mean ± std with and without TTA.
>
> Part B: write `scripts/analyze_errors.py` and the calibration analysis, then `docs/MODEL_CARD.md`. For the error analysis, actually look at the 50 highest-loss images and describe what you see — expect a mix of genuinely ambiguous images and some mislabeled ones. Report ECE with and without label smoothing.
>
> Touch the test set exactly once, at the very end.
