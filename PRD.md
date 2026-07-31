# PRD — CIFAR-10 From Scratch (PyTorch)

**Owner:** Udit Parihar
**Status:** Draft v1.0
**Last updated:** 2026-07-31
**Repo:** `CIFAR-10/`

---

## 1. Summary

Build a research-grade, fully modular CIFAR-10 image classification system in PyTorch — every component written from scratch (no `timm`, no pretrained weights, no Lightning). The system must reach **≥94% test accuracy** on a single consumer GPU in under 45 minutes of training, and it must do so **reproducibly**: same seed, same config, same number, every time.

The project is deliberately structured as a sequence of independently-shippable components so that each one can be handed to an AI coding assistant as a self-contained task with a precise interface contract and machine-checkable acceptance criteria.

## 2. Why this exists

Two goals, in priority order:

1. **Portfolio-grade artifact.** A recruiter or engineer opening this repo should see disciplined ML engineering — config-driven experiments, a run registry, ablation tables that justify every design choice, and results that anyone can reproduce with one command.
2. **Ground-truth understanding.** Every layer, every schedule, every augmentation is implemented or explicitly wired by hand, so the accuracy number is *earned*, not inherited from a library default.

## 3. Non-goals

Explicitly out of scope for v1. Listing these prevents scope creep mid-sprint.

- Multi-GPU / distributed training (`DDP`). Single-device only.
- Pretrained weights or transfer learning of any kind.
- Datasets other than CIFAR-10. (CIFAR-100 is a v2 stretch — the code should not *prevent* it, but no work goes into it.)
- Hyperparameter search frameworks (Optuna, Ray Tune). Ablations are run manually from configs.
- Model serving, ONNX export, or a demo web app.
- Neural architecture search.
- Beating published SOTA (~99%). We target the strong-classical-baseline band.

## 4. Success metrics

| # | Metric | Target | How it's measured |
|---|--------|--------|-------------------|
| M1 | Test accuracy, primary model | ≥ 94.0% (stretch 95.5%) | `eval.py` on the 10,000-image test set, final epoch weights |
| M2 | Reproducibility | Two runs, same seed+config → top-1 identical to within **±0.15%** | `make repro-check` runs the same config twice, diffs the metrics |
| M3 | Wall-clock to target | ≤ 45 min for the 94% recipe | `run_manifest.json` records total train seconds |
| M4 | Baseline sanity | SimpleCNN ≥ 85% in ≤ 30 epochs | Sprint 2 acceptance run |
| M5 | Config coverage | 0 magic numbers in `src/` | grep audit in Sprint 8; every tunable lives in a config dataclass |
| M6 | Evidence | Every recipe decision backed by a logged ablation | `docs/EXPERIMENTS.md` table, ≥ 8 rows |
| M7 | Cold-start time | Fresh clone → training in ≤ 3 commands | `README.md` quickstart, verified in Sprint 8 |

**Definition of "done" for the whole project:** M1–M7 all green, `docs/EXPERIMENTS.md` complete, README shows a training-curve plot and a confusion matrix.

## 5. Constraints & assumptions

- **Hardware:** RTX 4060 Laptop GPU (8 GB) + i7-14700HX. All budgets assume batch size 128 with AMP enabled. VRAM is not a constraint here — ResNet-18 on 32×32 inputs at batch 128 uses under 2 GB — but laptop thermals are, so expect ~10–15% throughput decay on runs over an hour.
- **Python** 3.10+, **PyTorch** 2.x, CUDA available but code must degrade gracefully to CPU (slow) for tests.
- **Dependencies:** `torch`, `torchvision` (for the *dataset download only*, plus `ToTensor`), `numpy`, `pyyaml`, `tensorboard`, `matplotlib`, `rich`. Anything else needs a note in the ADR log.
- **Epoch budget reference** (batch 128, AMP, channels_last, RTX 4060 Laptop): SimpleCNN ≈ 5–6 s/epoch, ResNet-18-CIFAR ≈ 14–18 s/epoch. Measure yours in Sprint 3 and replace these estimates. At ~16 s/epoch: 120 ep ≈ 32 min (inside the M3 budget), 200 ep ≈ 53 min, 300 ep ≈ 80 min. Total GPU time across Sprints 5–7 ≈ 16–18 h.

## 6. Architecture

### 6.1 Design principle

**Deep modules, narrow interfaces.** Each subsystem exposes a small surface and hides its complexity. The training engine does not know what a `Mixup` is; it knows it received a batch. The model does not know what an optimizer is. This is what makes each sprint independently implementable and testable.

```
             ┌──────────────┐
   YAML ───► │    Config    │  frozen dataclasses, validated at load
             └──────┬───────┘
                    │  (single object passed everywhere)
      ┌─────────────┼──────────────┬─────────────────┐
      ▼             ▼              ▼                 ▼
  ┌────────┐  ┌──────────┐  ┌────────────┐   ┌────────────┐
  │  Data  │  │  Models  │  │ Optim/Sched│   │  Tracking  │
  │ module │  │ registry │  │  factory   │   │  (TB+JSON) │
  └────┬───┘  └────┬─────┘  └─────┬──────┘   └─────▲──────┘
       │           │              │                │
       └───────────┴──────┬───────┘                │
                          ▼                        │
                  ┌───────────────┐                │
                  │   Trainer     │────────────────┘
                  │ (epoch loop,  │  emits Metrics events
                  │  AMP, ckpt)   │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │  Evaluator    │  test acc, per-class, confusion, TTA
                  └───────────────┘
```

### 6.2 Repository layout

```
CIFAR-10/
├── README.md
├── CLAUDE.md                  # coding standards — read by AI before every task
├── PRD.md                     # this file
├── Makefile                   # make train / eval / test / repro-check
├── pyproject.toml
├── configs/
│   ├── base.yaml              # defaults every config inherits
│   ├── simplecnn.yaml
│   ├── resnet18.yaml          # the 94% recipe
│   ├── resnet18_plus.yaml     # the 95.5% stretch recipe
│   └── ablations/             # one file per ablation row
├── src/cifar/
│   ├── config.py              # dataclasses + YAML loader + validation
│   ├── seeding.py             # global seed, worker seeding, determinism flags
│   ├── data/
│   │   ├── dataset.py         # CIFAR10 wrapper, train/val split
│   │   ├── transforms.py      # normalization, crop, flip, cutout, randaugment
│   │   ├── batch_aug.py       # mixup / cutmix (operate on batches)
│   │   └── loaders.py         # build_dataloaders(cfg)
│   ├── models/
│   │   ├── registry.py        # @register_model decorator, build_model(cfg)
│   │   ├── simple_cnn.py
│   │   ├── resnet.py          # CIFAR-style ResNet, from scratch
│   │   └── layers.py          # shared blocks
│   ├── engine/
│   │   ├── trainer.py         # Trainer class — the epoch loop
│   │   ├── evaluator.py
│   │   ├── optim.py           # optimizer + scheduler factories, param groups
│   │   ├── criterion.py       # CE, label smoothing, soft-target CE for mixup
│   │   ├── ema.py
│   │   └── checkpoint.py      # save/load/resume, best-tracking
│   ├── tracking/
│   │   ├── run.py             # Run dir, run_manifest.json, git SHA capture
│   │   ├── logger.py          # console (rich) + TensorBoard + metrics.jsonl
│   │   └── meters.py          # AverageMeter, accuracy@k, timers
│   └── utils/
│       └── viz.py             # curves, confusion matrix, sample grids
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── sweep.py               # run a folder of ablation configs sequentially
│   └── make_report.py         # regenerate README figures + EXPERIMENTS table
├── tests/
├── runs/                      # gitignored — one dir per run
└── docs/
    ├── ARCHITECTURE.md        # the binding interface contracts
    ├── EXPERIMENTS.md         # ablation ledger
    ├── ADR.md                 # decision log
    └── sprints/               # 00-08, one spec per sprint
```

### 6.3 Binding interface contracts

These signatures are frozen. A sprint may not change them without an ADR entry. They are the reason sprints can be built in any order once their dependencies land. Full detail in `docs/ARCHITECTURE.md`.

```python
# config.py
def load_config(path: str | Path, overrides: list[str] | None = None) -> Config: ...

# data/loaders.py
def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train, val, test)."""

# models/registry.py
def build_model(cfg: Config) -> nn.Module: ...

# engine/optim.py
def build_optimizer(model: nn.Module, cfg: Config) -> Optimizer: ...
def build_scheduler(opt: Optimizer, cfg: Config, steps_per_epoch: int) -> LRScheduler: ...

# engine/trainer.py
class Trainer:
    def __init__(self, model, loaders, cfg, run: Run) -> None: ...
    def fit(self) -> dict[str, float]:  # returns final summary metrics
    def train_one_epoch(self, epoch: int) -> dict[str, float]: ...

# engine/evaluator.py
def evaluate(model, loader, cfg, *, tta: bool = False) -> EvalResult: ...
```

## 7. Roadmap — 9 sprints

Each sprint is a shippable increment: the repo runs end-to-end at the close of every sprint from Sprint 3 onward. Detailed specs live in `docs/sprints/`.

| Sprint | Name | Ships | Gate |
|---|---|---|---|
| **0** | Foundation | Repo, config system, seeding, Makefile, CI-free test harness | `pytest` green; config round-trips |
| **1** | Data pipeline | Dataset, split, transforms, loaders, stats verification | Normalization stats match published; loader smoke test |
| **2** | Models & registry | `SimpleCNN`, `ResNet-18/34-CIFAR`, registry | Param counts match reference; forward-shape tests |
| **3** | Training engine | Trainer, evaluator, checkpointing, AMP, resume | **First end-to-end run. SimpleCNN ≥ 85%** |
| **4** | Tracking & reporting | Run dirs, manifests, TensorBoard, metrics.jsonl, curves | Two runs comparable side by side; resume is bit-faithful |
| **5** | The 94% recipe | SGD+Nesterov, warmup+cosine, wd tuning, label smoothing | **ResNet-18 ≥ 94.0%** |
| **6** | Advanced augmentation | Cutout, Mixup, CutMix, RandAugment, soft-target loss | **≥ 95.0%**, each aug ablated |
| **7** | Squeeze & analyze | EMA, TTA, longer schedule, error analysis, calibration | **≥ 95.5% stretch**; confusion matrix + model card |
| **8** | Hardening & release | Repro verification, README, report generation, magic-number audit | M2, M5, M7 green; final report |

### Dependency graph

```
S0 ──► S1 ──┐
     └► S2 ──┴──► S3 ──► S4 ──► S5 ──► S6 ──► S7 ──► S8
```

S1 and S2 are parallelizable. Everything after S3 is strictly sequential because each accuracy gate builds on the previous recipe.

## 8. Accuracy ladder

The number the project reports must climb in a documented, attributable way. Each row is one experiment in `EXPERIMENTS.md`.

| Stage | Configuration | Expected test acc | Sprint |
|---|---|---|---|
| A | SimpleCNN, 30 ep, Adam, flip+crop | 85–88% | 3 |
| B | ResNet-18-CIFAR, 30 ep, SGD 0.1, step LR, flip+crop | 91–93% | 5 |
| C | + Nesterov, 5-ep warmup, cosine to 0, 120 ep, wd 5e-4 | 94.5–95.3% | 5 |
| D | + label smoothing 0.1 | +0.1–0.3% | 5 |
| E | + Cutout (1 hole, 16px) | +0.3–0.6% | 6 |
| F | + Mixup α=0.2 **or** CutMix α=1.0, 200 ep | +0.2–0.5% | 6 |
| G | + EMA (decay 0.999) | +0.1–0.3% | 7 |
| H | + TTA (hflip) at eval | +0.1–0.2% | 7 |

Bands are estimates from the literature, not promises. If a stage underperforms its band, that is a finding — log it in `EXPERIMENTS.md` with a hypothesis rather than quietly moving on.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Accuracy plateaus below 94% | Primary metric misses | Sprint 5 has a debug checklist: verify normalization stats, LR range test, confirm no val leakage, check wd is not applied to BN/bias |
| Bit-exact reproducibility unattainable on GPU | M2 misses | M2 is a **±0.15% tolerance**, not bit-exactness. `cudnn.deterministic` mode documented as an opt-in flag with its throughput cost measured |
| Test set used for model selection | Inflated, dishonest number | Sprint 1 carves a 5,000-image val split from train. Test set is touched **only** by `eval.py`, only at the end of a stage. Enforced by a test that asserts `Trainer` never receives the test loader |
| Scope creep into a framework | Never ships | Non-goals in §3 are binding; new ideas go to `docs/BACKLOG.md` |
| Long runs lost to a crash | Time waste | Checkpoint every epoch with resume from Sprint 3, not later |
| AI assistant drifts from interfaces | Integration hell | `CLAUDE.md` + frozen signatures in §6.3; every sprint spec restates its contract |

## 10. Working method (building with AI)

The repo is optimized for AI-assisted implementation. The loop for each sprint:

1. **Load context.** Point the assistant at `CLAUDE.md`, `docs/ARCHITECTURE.md`, and the one sprint file. Nothing else.
2. **Implement one component at a time**, in the order listed in the sprint spec. Each component's spec includes its exact signature and its tests.
3. **Tests before integration.** The sprint spec's acceptance criteria are written as assertions; they become `tests/` files.
4. **Run the gate.** Every sprint has a numeric or executable gate. Do not open the next sprint until it passes.
5. **Log the result.** Append to `EXPERIMENTS.md`; record any interface deviation in `ADR.md`.
6. **Commit per component**, message format `sprint-N: <component>`.

Each sprint file ends with a **ready-to-paste implementation prompt** containing everything the assistant needs.

## 11. Open questions

- Should the val split be a fixed index list committed to the repo (maximum reproducibility) or seed-derived? *Leaning: fixed index file, decided in Sprint 1.*
- RandAugment from scratch vs. `torchvision.transforms.RandAugment`? *Leaning: from scratch, since "from scratch" is the point — but time-box it, and fall back to torchvision with an ADR note if it costs more than a session.*
- Is 200-epoch training acceptable for the final run given the 45-min budget? *Measure in Sprint 5; if ResNet-18 is ~11 s/epoch, 200 epochs ≈ 37 min, which fits.*
