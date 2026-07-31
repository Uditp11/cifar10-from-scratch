# Sprint 0 — Foundation

**Goal:** A repo skeleton where a config can be loaded, validated, hashed, and overridden from the CLI, and where `pytest` runs green in under 10 seconds.

**Depends on:** nothing. **Blocks:** everything.
**Estimated effort:** 1 session.

---

## Why this comes first

Every later sprint reads from `Config`. If the config system is added late, every module gets retrofitted and the "no magic numbers" rule dies in week one. Building it first makes it free.

## Components

### C0.1 — Project scaffold

Create the directory tree from PRD §6.2 with `__init__.py` in every package. Add:

- `pyproject.toml` — project metadata, deps (`torch`, `torchvision`, `numpy`, `pyyaml`, `tensorboard`, `matplotlib`, `rich`), dev deps (`pytest`, `ruff`), ruff config (line-length 100, target py310).
- `.gitignore` — `runs/`, `data/`, `__pycache__/`, `*.pth`, `.pytest_cache/`, `*.egg-info/`
- `Makefile` with targets: `install`, `test`, `lint`, `format`, `train`, `eval`, `clean`, `repro-check`. On Windows, also provide the raw commands in the README since `make` may be absent.
- Empty `docs/ADR.md` with the ADR template at the top.

**Done when:** `pip install -e .` succeeds and `python -c "import cifar; print(cifar.__version__)"` prints a version.

### C0.2 — Config dataclasses

Implement every dataclass from `ARCHITECTURE.md` §1 in `src/cifar/config.py`. All `frozen=True`. All fields have defaults matching the architecture doc exactly.

**Done when:** `Config(DataConfig(), AugConfig(), ...)` constructs with zero arguments beyond the sub-configs.

### C0.3 — YAML loader with inheritance

```python
def load_config(path, overrides=None) -> Config
```

Behavior:
1. Read `configs/base.yaml`.
2. If the target YAML has a top-level `inherits: base.yaml`, deep-merge it over the parent. Support one level of inheritance only — chains are a smell.
3. Apply dotted overrides (`"optim.learning_rate=0.05"`), coercing the string to the target field's annotated type. `true`/`false` → bool; comma-separated → tuple for tuple-typed fields.
4. Construct the nested dataclasses.
5. Validate (see below), then freeze.

Unknown keys must raise, not warn. A typo'd `lerning_rate` that silently does nothing is the single most expensive bug class in ML config systems.

### C0.4 — Validation

Implement every invariant listed in `ARCHITECTURE.md` §1. Each raises `ValueError` with a message naming the offending field and both values. Model-name validation is deferred until the registry exists (Sprint 2) — for now, validate it is a non-empty string, and leave a `# TODO(sprint-2)` marker.

### C0.5 — Config hash + serialization

```python
def config_to_dict(cfg) -> dict
def config_hash(cfg) -> str   # 12 hex chars
```

The hash is over the canonical JSON of the config with `notes` and the whole `log` section removed, keys sorted. Two runs that should be identical must hash identically; changing `learning_rate` must change the hash.

### C0.6 — Seeding module

Implement `ARCHITECTURE.md` §2. `seed_everything` seeds `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, sets `PYTHONHASHSEED`, and applies cudnn flags. When `deterministic=True` also call `torch.use_deterministic_algorithms(True)` and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### C0.7 — Base configs

`configs/base.yaml` holds every default explicitly (not relying on dataclass defaults — an operator should be able to read one file and know the whole setup). `configs/simplecnn.yaml` and `configs/resnet18.yaml` inherit from it and override only what differs.

---

## Acceptance criteria

These become `tests/test_config.py` and `tests/test_seeding.py`.

```python
def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg.optim.learning_rate == 0.1

def test_inheritance_merges_not_replaces():
    cfg = load_config("configs/resnet18.yaml")
    assert cfg.model.name == "resnet18_cifar"      # from child
    assert cfg.data.batch_size == 128              # from base

def test_override_applies_and_coerces_type():
    cfg = load_config("configs/base.yaml", ["optim.learning_rate=0.05",
                                            "train.epochs=200",
                                            "train.amp=false"])
    assert cfg.optim.learning_rate == 0.05 and isinstance(cfg.optim.learning_rate, float)
    assert cfg.train.epochs == 200 and isinstance(cfg.train.epochs, int)
    assert cfg.train.amp is False

def test_unknown_key_raises():
    with pytest.raises(ValueError, match="lerning_rate"):
        load_config("configs/base.yaml", ["optim.lerning_rate=0.05"])

def test_config_is_immutable():
    cfg = load_config("configs/base.yaml")
    with pytest.raises(FrozenInstanceError):
        cfg.optim.learning_rate = 0.5

def test_invalid_warmup_raises():
    with pytest.raises(ValueError, match="warmup_epochs"):
        load_config("configs/base.yaml", ["optim.warmup_epochs=999", "train.epochs=10"])

def test_hash_stable_and_sensitive():
    a = load_config("configs/base.yaml")
    b = load_config("configs/base.yaml", ["notes=different note"])
    c = load_config("configs/base.yaml", ["optim.learning_rate=0.05"])
    assert config_hash(a) == config_hash(b)     # notes excluded
    assert config_hash(a) != config_hash(c)

def test_seeding_is_reproducible():
    seed_everything(42)
    a = torch.randn(100)
    seed_everything(42)
    assert torch.equal(a, torch.randn(100))
```

## Gate

```
pytest -q            # all green, < 10s
ruff check src/      # clean
```

---

## Implementation prompt

> Read `CLAUDE.md` and `docs/ARCHITECTURE.md` sections 1 and 2, then `docs/sprints/00-foundation.md`.
>
> Implement Sprint 0 components C0.1 through C0.7 in that order. After each component, write its tests from the acceptance-criteria section and make them pass before moving on.
>
> Key constraints: all config dataclasses are `frozen=True`; unknown override keys must raise `ValueError` naming the key; the config hash must exclude `notes` and the entire `log` section. Do not implement any data, model, or training code in this sprint — the model-name validation is intentionally deferred to Sprint 2 with a TODO marker.
>
> Stop when `pytest -q` and `ruff check src/` are both clean, and report the test count.
