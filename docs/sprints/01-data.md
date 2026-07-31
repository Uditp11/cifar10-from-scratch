# Sprint 1 — Data pipeline

**Goal:** `build_dataloaders(cfg)` returns correct, verified, reproducible train/val/test loaders with a fixed val split and a configurable augmentation pipeline.

**Depends on:** Sprint 0. **Blocks:** Sprint 3. **Parallel with:** Sprint 2.
**Estimated effort:** 1 session.

---

## Why this matters more than it looks

Three of the most common causes of a CIFAR project stalling at 91% instead of 95% live here: wrong normalization statistics, a val split that leaks into train, and augmentation applied in the wrong order. Sprint 1 makes each of those a test failure rather than a mystery.

## Components

### C1.1 — Dataset + fixed val split

`src/cifar/data/dataset.py`

- Download CIFAR-10 via `torchvision.datasets.CIFAR10` to `cfg.data.root`. Download once; subsequent runs must not re-check the network if the files exist.
- **Fixed val split.** Generate `configs/val_indices.json` once, with a script or an inline one-time function: a stratified sample of `cfg.data.val_size` indices from the 50,000 train images, 500 per class when `val_size=5000`. Commit this file. Never re-derive it from a seed at runtime — this is the decision left open in PRD §11, and this is the resolution. Record it in `ADR.md`.
- `build_datasets(cfg)` returns `(train_ds, val_ds, test_ds)` where train and val are disjoint `Subset`s of the CIFAR-10 train split with *different transforms*, and test is the untouched 10,000-image test split.

The different-transforms-on-one-underlying-dataset problem is the subtle part: a naive `Subset` shares the parent's transform. Solve it by constructing two separate `CIFAR10` instances over the same files with different transforms, then subsetting each. Document the choice.

### C1.2 — Normalization statistics

Compute the channel mean/std over the **train split only** (excluding val, to be strictly correct) with a small script, and assert they land within 0.001 of the config defaults `(0.4914, 0.4822, 0.4465) / (0.2470, 0.2435, 0.2616)`. If they differ meaningfully, trust your computation and update the config — but investigate first, because a mismatch usually means the images are being read in the wrong dtype range.

Note the two conventions in the wild: std computed per-image-then-averaged gives ≈`(0.2023, 0.1994, 0.2010)`; std over the whole dataset gives ≈`(0.2470, 0.2435, 0.2616)`. Use the dataset-wide version and say so in a comment, because seeing both numbers in different repos is confusing enough to deserve one sentence of explanation.

### C1.3 — Transform pipeline

`src/cifar/data/transforms.py`

Train pipeline order (order is load-bearing):

```
RandomCrop(32, padding=cfg.aug.random_crop_padding, padding_mode="reflect")
RandomHorizontalFlip(p=cfg.aug.horizontal_flip_p)
[RandAugment(n, m)]        # Sprint 6 — leave the hook, no-op when n == 0
ToTensor()
Normalize(mean, std)
[Cutout(holes, length)]    # Sprint 6 — AFTER normalize; no-op when holes == 0
```

Eval pipeline: `ToTensor()` → `Normalize(mean, std)`. Nothing else, ever.

Every stage is skipped entirely (not applied with p=0) when its config value disables it, so the "no augmentation" baseline is genuinely clean.

In this sprint, implement crop/flip/totensor/normalize. Stub `Cutout` and `RandAugment` as identity classes with a `# implemented in sprint 6` note — the config plumbing should exist from day one so Sprint 6 is a pure fill-in.

### C1.4 — DataLoaders

`src/cifar/data/loaders.py`

- Train loader: `shuffle=True`, `drop_last=True`, seeded `generator`, `worker_init_fn` from `seeding.py`.
- Val/test loaders: `shuffle=False`, `drop_last=False`, `batch_size=cfg.data.eval_batch_size`.
- `pin_memory` and `persistent_workers` from config. Guard `persistent_workers` — it requires `num_workers > 0`.
- On Windows, `num_workers > 0` needs the `if __name__ == "__main__":` guard in scripts. Note it in the README.

### C1.5 — Batch augmentation stubs

`src/cifar/data/batch_aug.py` — define `MixedBatch` and `apply_batch_aug` per the architecture contract. In this sprint the implementation returns `MixedBatch(x, y, y, 1.0)` unconditionally. This lets Sprint 3's trainer write its final shape now and Sprint 6 fill in the logic without touching the trainer.

### C1.6 — Visual sanity check

`scripts/inspect_data.py` — saves a 8×8 grid of augmented training images (denormalized) plus a per-class count table to `docs/figures/data_samples.png`. Look at it. Augmentation bugs are usually obvious to the eye and invisible in the metrics.

---

## Acceptance criteria

`tests/test_data.py`:

```python
def test_split_sizes():
    tr, va, te = build_datasets(cfg)
    assert len(tr) == 45000 and len(va) == 5000 and len(te) == 10000

def test_train_val_disjoint():
    tr, va, _ = build_datasets(cfg)
    assert set(tr.indices).isdisjoint(set(va.indices))

def test_val_split_is_stratified():
    _, va, _ = build_datasets(cfg)
    counts = Counter(va.dataset.targets[i] for i in va.indices)
    assert all(c == 500 for c in counts.values())

def test_val_split_is_fixed_across_seeds():
    a = build_datasets(replace_seed(cfg, 1))[1].indices
    b = build_datasets(replace_seed(cfg, 999))[1].indices
    assert list(a) == list(b)

def test_normalized_batch_statistics():
    tr, _, _ = build_dataloaders(cfg_no_aug)
    x, _ = next(iter(tr))
    assert x.shape == (cfg.data.batch_size, 3, 32, 32)
    assert abs(x.mean().item()) < 0.1
    assert abs(x.std().item() - 1.0) < 0.15

def test_eval_transform_is_deterministic():
    _, va, _ = build_datasets(cfg)
    assert torch.equal(va[0][0], va[0][0])

def test_train_transform_is_stochastic():
    tr, _, _ = build_datasets(cfg)          # crop+flip enabled
    seed_everything(0); a = tr[0][0]
    seed_everything(1); b = tr[0][0]
    assert not torch.equal(a, b)

def test_disabled_aug_is_identity():
    cfg0 = override(cfg, random_crop_padding=0, horizontal_flip_p=0.0)
    tr, _, _ = build_datasets(cfg0)
    assert torch.equal(tr[0][0], tr[0][0])

def test_loader_is_reproducible_across_runs():
    order_a = first_batch_labels(build_dataloaders(cfg)[0])
    order_b = first_batch_labels(build_dataloaders(cfg)[0])
    assert order_a == order_b
```

## Gate

- All tests green.
- `scripts/inspect_data.py` produces a grid where crops and flips are visibly present and images are recognizable (not scrambled, not gray).
- Computed normalization stats match config within 0.001.

---

## Implementation prompt

> Read `CLAUDE.md`, `docs/ARCHITECTURE.md` sections 1 and 3, and `docs/sprints/01-data.md`.
>
> Implement C1.1 through C1.6. Two things to be careful about: (a) train and val must use *different transforms* despite coming from the same underlying CIFAR-10 train split — construct two dataset instances rather than sharing one; (b) the val split comes from a committed `configs/val_indices.json`, stratified 500-per-class, and must be identical regardless of seed.
>
> Leave `Cutout` and `RandAugment` as identity stubs with the config plumbing fully wired, and leave `apply_batch_aug` returning an unmixed `MixedBatch`. Sprint 6 fills those in; nothing downstream should need to change when it does.
>
> Write the tests from the acceptance section as you go. Finish by running `scripts/inspect_data.py` and describing what the sample grid shows.
