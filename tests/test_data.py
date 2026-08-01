import dataclasses
from collections import Counter

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from cifar.config import Config, load_config
from cifar.data.batch_aug import apply_batch_aug
from cifar.data.dataset import build_datasets
from cifar.data.loaders import build_dataloaders
from cifar.data.transforms import (
    Cutout,
    RandAugment,
    build_eval_transform,
    build_train_transform,
)
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


# --- sprint 6: cutout ---------------------------------------------------------


def test_cutout_zeroes_expected_area():
    x = torch.ones(3, 32, 32)
    zeroed = (Cutout(n_holes=1, length=16)(x) == 0).float().mean().item() * 32 * 32
    # 16x16=256 when fully inside; 8x8=64 when the centre lands in a corner.
    assert 64 <= zeroed <= 256


def test_cutout_disabled_is_identity():
    x = torch.randn(3, 32, 32)
    assert torch.equal(Cutout(n_holes=0, length=16)(x), x)


def test_cutout_omitted_from_pipeline_when_disabled(cfg):
    stages = build_train_transform(_override_aug(cfg, cutout_holes=0)).transforms
    assert not any(isinstance(s, Cutout) for s in stages)


def test_cutout_does_not_mutate_its_input():
    """The dataset's tensor must survive the transform unchanged."""
    x = torch.ones(3, 32, 32)
    Cutout(n_holes=1, length=16)(x)
    assert x.min().item() == 1.0


def test_cutout_runs_after_normalization(cfg):
    """Data-flow invariant 2: zero must mean the mean pixel, not black."""
    stages = build_train_transform(_override_aug(cfg, cutout_holes=1)).transforms
    types = [type(s) for s in stages]
    assert types.index(Cutout) > types.index(T.Normalize)


# --- sprint 6: batch aug ------------------------------------------------------


def _batch(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n, 3, 32, 32), torch.randint(10, (n,))


def test_mixup_lam_in_range_and_shape_preserved(cfg):
    x, y = _batch()
    b = apply_batch_aug(x, y, _override_aug(cfg, mixup_alpha=0.2, batch_aug_prob=1.0))
    assert 0.0 <= b.lam <= 1.0 and b.inputs.shape == x.shape


def test_mixup_output_is_convex_combination(cfg):
    x, y = _batch()
    b = apply_batch_aug(x, y, _override_aug(cfg, mixup_alpha=0.2, batch_aug_prob=1.0))
    assert b.inputs.min() >= x.min() - 1e-5
    assert b.inputs.max() <= x.max() + 1e-5


def test_cutmix_lam_matches_actual_pasted_area(cfg):
    """The clipped-box bug. lam must be recomputed from the real box."""
    cfg_cutmix = _override_aug(cfg, cutmix_alpha=1.0, batch_aug_prob=1.0)
    for _ in range(20):
        x, y = _batch()
        b = apply_batch_aug(x, y, cfg_cutmix)
        changed = (b.inputs != x).float().mean(dim=(1, 2, 3)).mean().item()
        assert abs((1 - b.lam) - changed) < 0.05


def test_cutmix_pastes_a_rectangle(cfg):
    """A box that clips must stay one contiguous rectangle, not wrap around."""
    x, y = _batch(n=8)
    b = apply_batch_aug(x, y, _override_aug(cfg, cutmix_alpha=1.0, batch_aug_prob=1.0))
    changed = (b.inputs != x).any(dim=1).any(dim=0)  # (H, W): touched anywhere in the batch
    rows, cols = changed.any(dim=1).nonzero(), changed.any(dim=0).nonzero()
    if rows.numel():  # an empty box is legal when lam is drawn near 1
        assert rows.numel() == rows.max() - rows.min() + 1
        assert cols.numel() == cols.max() - cols.min() + 1


def test_batch_aug_prob_zero_is_passthrough(cfg):
    """A live alpha with prob 0 must still never fire (`_validate` rejects this
    combination at load; `dataclasses.replace` here bypasses it on purpose)."""
    x, y = _batch()
    b = apply_batch_aug(x, y, _override_aug(cfg, mixup_alpha=0.2, batch_aug_prob=0.0))
    assert b.lam == 1.0 and torch.equal(b.inputs, x) and torch.equal(b.targets_a, b.targets_b)


def test_batch_aug_fires_at_the_configured_rate(cfg):
    """A prob strictly between 0 and 1 must sometimes skip and sometimes mix."""
    x, y = _batch(n=8)
    cfg_half = _override_aug(cfg, mixup_alpha=0.2, batch_aug_prob=0.5)
    seed_everything(0)
    fired = [apply_batch_aug(x, y, cfg_half).lam != 1.0 for _ in range(40)]
    assert any(fired) and not all(fired)


# --- sprint 6: randaugment ----------------------------------------------------


def test_randaugment_preserves_size_and_mode():
    img = Image.new("RGB", (32, 32))
    out = RandAugment(2, 9)(img)
    assert out.size == (32, 32) and out.mode == "RGB"


def test_randaugment_applies_every_op_at_full_magnitude():
    """Every op in the table must run without raising at the extremes."""
    img = Image.new("RGB", (32, 32), (120, 60, 200))
    for m in (0, 30):
        assert RandAugment(n=40, m=m)(img).size == (32, 32)


def test_randaugment_rejects_out_of_range_magnitude():
    with pytest.raises(ValueError, match="randaugment_m"):
        RandAugment(2, 31)


def test_randaugment_is_stochastic():
    img = Image.new("RGB", (32, 32), (120, 60, 200))
    aug = RandAugment(2, 20)
    seed_everything(0)
    a = aug(img).tobytes()
    seed_everything(1)
    b = aug(img).tobytes()
    assert a != b


def test_no_augmentation_at_eval_time(cfg):
    """Regression guard: the eval transform must contain no stochastic op."""
    cfg_everything = _override_aug(
        cfg, cutout_holes=1, randaugment_n=2, mixup_alpha=0.2, batch_aug_prob=1.0
    )
    stages = build_eval_transform(cfg_everything).transforms
    assert [type(s) for s in stages] == [T.ToTensor, T.Normalize]
