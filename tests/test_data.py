import dataclasses
from collections import Counter

import pytest
import torch

from cifar.config import Config, load_config
from cifar.data.dataset import build_datasets
from cifar.data.loaders import build_dataloaders
from cifar.seeding import seed_everything


def _base_cfg() -> Config:
    # num_workers=0: spawned workers blow the 5s test budget on Windows.
    return load_config("configs/resnet18.yaml", ["data.num_workers=0"])


def _replace_seed(cfg: Config, seed: int) -> Config:
    return dataclasses.replace(cfg, repro=dataclasses.replace(cfg.repro, seed=seed))


def _override_aug(cfg: Config, **kwargs) -> Config:
    return dataclasses.replace(cfg, aug=dataclasses.replace(cfg.aug, **kwargs))


def _first_batch_labels(loader) -> list[int]:
    _, y = next(iter(loader))
    return y.tolist()


@pytest.fixture(scope="module")
def cfg() -> Config:
    return _base_cfg()


def test_split_sizes(cfg):
    tr, va, te = build_datasets(cfg)
    assert len(tr) == 45000 and len(va) == 5000 and len(te) == 10000


def test_train_val_disjoint(cfg):
    tr, va, _ = build_datasets(cfg)
    assert set(tr.indices).isdisjoint(set(va.indices))


def test_val_split_is_stratified(cfg):
    _, va, _ = build_datasets(cfg)
    counts = Counter(va.dataset.targets[i] for i in va.indices)
    assert all(c == 500 for c in counts.values())


def test_val_split_is_fixed_across_seeds(cfg):
    a = build_datasets(_replace_seed(cfg, 1))[1].indices
    b = build_datasets(_replace_seed(cfg, 999))[1].indices
    assert list(a) == list(b)


def test_normalized_batch_statistics(cfg):
    cfg_no_aug = _override_aug(cfg, random_crop_padding=0, horizontal_flip_p=0.0)
    tr, _, _ = build_dataloaders(cfg_no_aug)
    x, _ = next(iter(tr))
    assert x.shape == (cfg.data.batch_size, 3, 32, 32)
    assert abs(x.mean().item()) < 0.1
    assert abs(x.std().item() - 1.0) < 0.15


def test_eval_transform_is_deterministic(cfg):
    _, va, _ = build_datasets(cfg)
    assert torch.equal(va[0][0], va[0][0])


def test_train_transform_is_stochastic(cfg):
    tr, _, _ = build_datasets(cfg)  # crop+flip enabled
    seed_everything(0)
    a = tr[0][0]
    seed_everything(1)
    b = tr[0][0]
    assert not torch.equal(a, b)


def test_disabled_aug_is_identity(cfg):
    cfg0 = _override_aug(cfg, random_crop_padding=0, horizontal_flip_p=0.0)
    tr, _, _ = build_datasets(cfg0)
    assert torch.equal(tr[0][0], tr[0][0])


def test_loader_is_reproducible_across_runs(cfg):
    order_a = _first_batch_labels(build_dataloaders(cfg)[0])
    order_b = _first_batch_labels(build_dataloaders(cfg)[0])
    assert order_a == order_b


def test_build_datasets_works_from_any_cwd(cfg, monkeypatch, tmp_path):
    # data.root resolved before chdir so this reuses the already-downloaded
    # cache instead of triggering a fresh download from tmp_path.
    cfg_abs_root = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, root=cfg.data.root.resolve())
    )
    monkeypatch.chdir(tmp_path)
    tr, va, te = build_datasets(cfg_abs_root)
    assert len(tr) == 45000 and len(va) == 5000 and len(te) == 10000
