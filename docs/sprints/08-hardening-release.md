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
3. Pass if final val accuracy differs by **≤ 0.15%**; fail loudly otherwise.

Also measure and document the throughput cost of `deterministic=true` — typically 10–30% slower. Put the number in the README so the tradeoff is explicit rather than folklore.

If the two runs diverge beyond tolerance, work backward: is a `DataLoader` worker unseeded? Is an augmentation using a global RNG instead of the seeded generator? Is a non-deterministic CUDA kernel in play that `use_deterministic_algorithms` didn't catch?

### C8.2 — Magic-number audit (PRD M5)

Grep `src/` for numeric literals outside of config defaults, shape constants, and mathematical constants. Every survivor either moves into a config field or earns a comment explaining why it's structural. Write the audit result into `docs/ADR.md`.

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

1. **One-line description + the headline result badge** — "95.6% test accuracy on CIFAR-10, ResNet-18 from scratch, 38 min on one RTX 4070."
2. **Results table** — every stage of the accuracy ladder with its config and its delta.
3. **Training curve + confusion matrix figures** — above the fold. This is what people actually look at.
4. **Quickstart** — the three commands from C8.3.
5. **What's implemented from scratch** — the list that makes the point: ResNet blocks, cosine+warmup schedule, Cutout, Mixup, CutMix, RandAugment, EMA, TTA, label smoothing, the whole training engine.
6. **Project structure** — the annotated tree.
7. **Key findings** — the three or four most interesting results, in prose. E.g. basic augmentation is worth ~6 points; excluding BN from weight decay is worth ~0.3; label smoothing barely moves accuracy but cuts ECE by more than half.
8. **Reproducibility** — seeds, determinism flag, its cost, the ±0.15% tolerance.
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
| M2 | Repro within ±0.15% | C8.1 | ☐ |
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

> Read `CLAUDE.md`, `PRD.md` section 4, and `docs/sprints/08-hardening-release.md`.
>
> Work through C8.1 to C8.7. The two that need real rigor: the reproducibility check (two runs, same seed, deterministic mode, ≤0.15% apart — and measure the throughput cost of determinism) and the cold-start verification, which must happen on a genuinely clean clone in a fresh virtualenv, not in the working tree.
>
> Then write the README. Lead with the headline result and the figures; keep it to roughly two screens. Numbers come from `make_report.py`, never hand-typed.
>
> Finish by filling in the verification checklist table in this file with actual measured values and a pass/fail for each of M1–M7. Report anything that missed its target honestly rather than adjusting the target.
