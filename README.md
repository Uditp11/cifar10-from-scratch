# CIFAR-10

See `PRD.md` for the project brief and `docs/ARCHITECTURE.md` for the frozen interfaces.

## Setup

```bash
pip install -e ".[dev]"
```

## Makefile targets

`make` may not be available on Windows; the raw commands are listed alongside each target.

| Target | Command |
|---|---|
| `install` | `pip install -e ".[dev]"` |
| `test` | `pytest -q` |
| `lint` | `ruff check src/` |
| `format` | `ruff format src/ tests/` |
| `train` | `python scripts/train.py` |
| `eval` | `python scripts/eval.py` |
| `clean` | remove `runs/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__` dirs |
| `repro-check` | `python scripts/train.py repro.deterministic=true` |
