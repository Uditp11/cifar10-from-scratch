"""One-time generator for `configs/val_indices.json`. See ADR-001.

Run once: `python scripts/make_val_split.py`. Do not re-run casually -- the
committed file is the source of truth regardless of any config's seed;
regenerating it changes every run's val split and breaks cross-run comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

from cifar.data.dataset import VAL_INDICES_PATH, generate_val_indices

_DATA_ROOT = Path("./data")
_VAL_SIZE = 5000
_SEED = 0


def main() -> None:
    indices = generate_val_indices(_DATA_ROOT, _VAL_SIZE, seed=_SEED)
    VAL_INDICES_PATH.write_text(json.dumps(indices))
    print(f"wrote {len(indices)} indices to {VAL_INDICES_PATH}")


if __name__ == "__main__":
    main()
