# Sprint 8 — Hardening & release

**Goal:** Anyone can clone this repo and reproduce the headline number in three commands. Every PRD success metric verified green.

**Depends on:** Sprint 7. **Blocks:** nothing.
**Estimated effort:** 1 session + one verification run.

---

## Components

### C8.1 — Reproducibility verification (PRD M2)

`make repro-check`:

1. Run `configs/resnet18.yaml` twice at `epochs=30`, same seed, `repro.deterministic=true`.
2. Compare `metrics.jsonl` from both.
3. Pass if the runs are **bit-identical** — every logged scalar equal and the model weights hashing the same. Fail loudly otherwise.

Also measure and document the throughput cost of `deterministic=true`. Put the number in the README so the tradeoff is explicit rather than folklore.

**Run 2026-08-02 — passed.** Three 10-epoch runs of `configs/repro_check.yaml` were bit-identical, weights hashing to `866f747125539331`, one of them on battery under CPU contention and straddling a replug mid-run at 2.1× the wall clock. Cost of determinism: ~25% (15.1–16.7 s/epoch against fast mode's 12.6). C8.1b then separated the two things `deterministic=true` switches off, and found disabling `cudnn.benchmark` alone sufficient. See `EXPERIMENTS.md`.

If the two runs diverge beyond tolerance, work backward: is a `DataLoader` worker unseeded? Is an augmentation using a global RNG instead of the seeded generator? Is a non-deterministic CUDA kernel in play that `use_deterministic_algorithms` didn't catch?

### C8.2 — Magic-number audit (PRD M5)

Grep `src/` for numeric literals outside of config defaults, shape constants, and mathematical constants. Every survivor either moves into a config field or earns a comment explaining why it's structural. Write the audit result into `docs/ADR.md`.

**Also in scope: config fields that silently do nothing** — the inverse defect, and the worse one. A magic number is a value with no name; a dead config field is a name with no value, and it actively misleads, because setting it looks like it did something. Two known cases:

- **`train.ema_decay` and `train.ema_warmup_epochs`.** Both are read by nothing. `src/cifar/engine/ema.py` is specified in `ARCHITECTURE.md` §5 but unimplemented, and `Trainer._save` passes `ema=None` with a `# sprint 7` comment. Setting `ema_decay: 0.999` today changes nothing and reports nothing. This was surfaced when Sprint 7's Part A was found to have met its target before starting: the dead field is the real defect that sprint was carrying, it is not an accuracy problem, and it does not need a sprint — roughly an hour, no GPU. Either implement EMA or delete the fields; both are honest, leaving them is not.
- **`train.keep_last_n_checkpoints`** is already recorded as inert in ADR-008. That entry is the template for how this class of finding gets written up.

The audit should end with a positive statement — every field in `Config` is either read somewhere in `src/`, or has an ADR entry saying why it isn't.

### C8.3 — Cold-start verification (PRD M7)

On a clean clone in a fresh virtualenv:

```bash
pip install -e .
python scripts/train.py --config configs/resnet18.yaml
python scripts/eval.py --run runs/<latest> --checkpoint best --tta
```

Three commands, no manual data download, no undocumented step. Fix whatever breaks — a missing `data/` mkdir, a Windows `num_workers` crash without the `__main__` guard, an absent `figures/` directory. Do this on a genuinely clean checkout; a working tree that has accumulated state hides exactly these bugs.

### C8.4 — README

The front door. Structure:

1. **One-line description + the headline result badge** — e.g. "96.06% test accuracy on CIFAR-10, ResNet-18 from scratch, 43 min on one RTX 4060 Laptop." Numbers and hardware come from a run manifest, never typed from memory: the GPU is `env.gpu` and the wall clock is `total_seconds`. This line previously read "38 min on one RTX 4070", which was neither the project's GPU nor a measured time.
2. **Results table** — every stage of the accuracy ladder with its config and its delta.
3. **Training curve + confusion matrix figures** — above the fold. This is what people actually look at.
4. **Quickstart** — the three commands from C8.3.
5. **What's implemented from scratch** — the list that makes the point: ResNet blocks, cosine+warmup schedule, Cutout, Mixup, CutMix, RandAugment, EMA, TTA, label smoothing, the whole training engine.
6. **Project structure** — the annotated tree.
7. **Key findings** — the three or four most interesting results, in prose. E.g. basic augmentation is worth ~6 points; excluding BN from weight decay is worth ~0.3; label smoothing barely moves accuracy but cuts ECE by more than half.
8. **Reproducibility** — seeds, the determinism flag and its cost, and one testable claim per mode: bit-identical under `deterministic=true`, a measured run-to-run spread under the default. The old ±0.15% tolerance is retired and must not be quoted.
9. **Full experiment log** — link to `EXPERIMENTS.md`.

Keep it tight. A README that fits on two screens and shows a figure beats a wall of text.

### C8.5 — Report generation

`scripts/make_report.py --all` regenerates every figure and the `EXPERIMENTS.md` table from `runs/`. The README's numbers should never be hand-maintained — they should be regenerable, so they can't drift from reality.

### C8.6 — Final test-set evaluation

One command, one number, reported once:

```bash
python scripts/eval.py --run runs/final --checkpoint best --ema --tta
```

Report top-1, top-5, per-class accuracy, and confusion matrix. This is the number that goes in the README.

### C8.7 — Backlog

`docs/BACKLOG.md` — everything deliberately not built, with a sentence on what each would take. Candidates: DDP, CIFAR-100, WideResNet-28-10 (~96.5% but ~5× the compute), ONNX export, knowledge distillation, `torch.compile`, self-supervised pretraining, a Gradio demo. This shows the non-goals were choices, not oversights.

---

## Final verification checklist

| Metric | Target | Verified by | Status |
|---|---|---|---|
| M1 | Test acc ≥ 94% (stretch 95.5%) | C8.6 | ☐ |
| M2 | Bit-identical under `deterministic=true`; fast-mode spread measured | C8.1 + C8.1b | ☑ 2026-08-02 |
| M3 | ≤ 45 min to the 94% recipe | manifest `total_seconds` | ☐ |
| M4 | SimpleCNN ≥ 85% | Sprint 3 run, still in `EXPERIMENTS.md` | ☐ |
| M5 | No magic numbers in `src/` | C8.2 | ☐ |
| M6 | ≥ 8 ablation rows | `EXPERIMENTS.md` | ☐ |
| M7 | Clone → train in 3 commands | C8.3 | ☐ |

Plus:

- ☐ `pytest -q` green, full suite under 60 s on CPU
- ☐ `ruff check src/ tests/ scripts/` clean
- ☐ Every public function has a type-hinted signature and a docstring
- ☐ No `print()` in `src/`
- ☐ `.gitignore` covers `runs/`, `data/`, `*.pth`
- ☐ Every `TODO` in the codebase is either resolved or listed in `BACKLOG.md`
- ☐ `docs/ADR.md` records every interface deviation from `ARCHITECTURE.md`
- ☐ README figures regenerate from `make_report.py`
- ☐ Model card complete

## Gate

Every box above ticked. Tag `v1.0`.

---

## Implementation prompt

> Read `PRD.md` section 4, and `docs/sprints/08-hardening-release.md`.
>
> Work through C8.1 to C8.7. The two that need real rigor: the reproducibility check (repeat runs, same seed, deterministic mode, bit-identical — and measure the throughput cost of determinism) and the cold-start verification, which must happen on a genuinely clean clone in a fresh virtualenv, not in the working tree.
>
> Then write the README. Lead with the headline result and the figures; keep it to roughly two screens. Numbers come from `make_report.py`, never hand-typed.
>
> Finish by filling in the verification checklist table in this file with actual measured values and a pass/fail for each of M1–M7. Report anything that missed its target honestly rather than adjusting the target.
