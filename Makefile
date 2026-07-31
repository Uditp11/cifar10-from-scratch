.PHONY: install test lint format train eval clean repro-check

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/

train:
	python scripts/train.py

eval:
	python scripts/eval.py

clean:
	rm -rf runs/ .pytest_cache/ .ruff_cache/ **/__pycache__

repro-check: # stub, wired in sprint 8
	python scripts/train.py repro.deterministic=true
