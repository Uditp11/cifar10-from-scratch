# CLAUDE.md — Coding standards for this repo

Read this before writing any code here. These rules exist so that nine sprints of AI-assisted work produce one coherent codebase instead of nine dialects.

## The contract

`docs/ARCHITECTURE.md` holds the frozen function signatures. **Do not change one.** If a signature genuinely must change, stop, write an entry in `docs/ADR.md` explaining why, and only then change it — plus every call site.

## Non-negotiables

1. **No magic numbers in `src/`.** Every tunable value comes from `Config`. If you catch yourself typing `0.1` or `128` or `5e-4` inside `src/`, it belongs in a config dataclass with a default. Exception: mathematical constants and shape literals that follow from CIFAR-10 itself (`10` classes, `32` px).
2. **No pretrained weights, no `timm`, no Lightning, no `torchvision.models`.** `torchvision` is used for `datasets.CIFAR10` download and `ToTensor` only.
3. **The test set is sacred.** `Trainer` never receives it. Only `scripts/eval.py` may load it. No early stopping, no LR decisions, no epoch selection based on test metrics.
4. **Everything seeds.** Any new source of randomness routes through `src/cifar/seeding.py`.
5. **Type hints on every public function.** Return types included.

## Style

- Python 3.10+ syntax: `X | None`, `list[int]`, `match` where it reads well.
- Dataclasses for config and for structured returns (`EvalResult`, `EpochMetrics`). Not dicts, when the shape is known.
- `pathlib.Path`, never string paths.
- f-strings, never `%` or `.format()`.
- Line length 100. Format with `ruff format`, lint with `ruff check`.
- Module-level `__all__` only where the module has a genuinely narrow public API.

## Naming

- `build_*` — factory functions taking `cfg` and returning a constructed object.
- `*_step` — one iteration of something.
- `cfg` is always the `Config` object. Never shadow it.
- Config fields use full words: `learning_rate` not `lr`, `weight_decay` not `wd`. Command-line overrides may abbreviate.

## Docstrings

Every public function gets one. Format:

```python
def build_scheduler(opt: Optimizer, cfg: Config, steps_per_epoch: int) -> LRScheduler:
    """Build the LR scheduler.

    Warmup is applied per-step for `cfg.optim.warmup_epochs`, then the main
    schedule runs per-step to the end of training.

    Args:
        opt: Optimizer whose param groups will be stepped.
        cfg: Full config; reads `cfg.optim`.
        steps_per_epoch: len(train_loader). Needed for per-step schedules.

    Returns:
        A scheduler whose `.step()` is called once per optimizer step.
    """
```

Explain **why**, not what. `# increment counter` is noise; `# BN and bias excluded from wd — see ADR-004` is the kind of comment worth writing.

## Errors

Fail loudly and early. Validate config at load, not at epoch 40.

```python
if cfg.optim.warmup_epochs >= cfg.train.epochs:
    raise ValueError(
        f"warmup_epochs ({cfg.optim.warmup_epochs}) must be < "
        f"epochs ({cfg.train.epochs})"
    )
```

Never swallow an exception silently. Never `except: pass`.

## Tests

- `pytest`, in `tests/`, mirroring the `src/` tree.
- Every test must run on **CPU** in **under 5 seconds**. Use tiny tensors, 2-sample batches, 1-step loops.
- Test the contract, not the implementation. `test_build_model_returns_correct_output_shape`, not `test_resnet_layer3_has_256_channels`.
- Every bug that gets fixed earns a regression test first.
- Mark anything needing a GPU with `@pytest.mark.gpu` so `pytest -m "not gpu"` stays fast.

## Determinism rules

- Seed at process start: Python `random`, `numpy`, `torch`, `torch.cuda`.
- `DataLoader` gets a `worker_init_fn` and a seeded `generator`.
- `cfg.repro.deterministic=True` sets `torch.use_deterministic_algorithms(True)` and `cudnn.deterministic=True`, `cudnn.benchmark=False`. Default is `False` (fast mode) — the repro check flips it on.
- Any run that can't be reproduced within ±0.15% is a bug, not a quirk.

## Logging

- Console output goes through `tracking/logger.py`. No bare `print()` in `src/`.
- Every scalar logged to console is also logged to TensorBoard and appended to `metrics.jsonl`.
- Log at epoch granularity by default; per-step logging is opt-in via `cfg.log.step_interval`.

## Git

- One commit per component, message `sprint-N: <component>` — e.g. `sprint-5: cosine scheduler with warmup`.
- `runs/`, `data/`, `*.pth` are gitignored. Config files and `EXPERIMENTS.md` are not.
- Every run captures the git SHA into its `run_manifest.json`. A run from a dirty tree is marked `dirty: true`.

## When you're unsure

Ask rather than guess, when the choice affects an interface, changes an accuracy claim, or adds a dependency. For anything internal to a module, pick the simpler option and note the choice in the docstring.
