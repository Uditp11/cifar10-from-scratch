.PHONY: install test lint format train eval clean repro-check

install:
	pip install -e ".[dev]"

test:
	pytest -m "not gpu" -q

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

train:
	python scripts/train.py

eval:
	python scripts/eval.py

clean:
	rm -rf runs/ .pytest_cache/ .ruff_cache/ **/__pycache__

# Runs the deterministic config once. The two-run diff that PRD M2 and sprint 8
# C8.1 describe is not automated yet -- that verification has only been done by hand.
repro-check:
	python scripts/train.py --config configs/repro_check.yaml
