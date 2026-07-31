import pytest
import torch
import torch.nn.functional as F

from cifar.config import Config, load_config
from cifar.models import build_model, list_models
from cifar.models.layers import BasicBlock


def cfg_for(name: str, device: str = "cpu", zero_init_residual: bool | None = None) -> Config:
    overrides = [f"model.name={name}", f"device={device}"]
    if zero_init_residual is not None:
        overrides.append(f"model.zero_init_residual={zero_init_residual}")
    return load_config("configs/base.yaml", overrides)


def all_basic_blocks(m: torch.nn.Module) -> list[BasicBlock]:
    return [mod for mod in m.modules() if isinstance(mod, BasicBlock)]


@pytest.mark.parametrize("name", list_models())
def test_forward_shape(name):
    m = build_model(cfg_for(name))
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


@pytest.mark.parametrize("name", list_models())
def test_batch_size_one_works(name):
    m = build_model(cfg_for(name)).eval()
    assert m(torch.randn(1, 3, 32, 32)).shape == (1, 10)


def test_param_counts_match_reference():
    assert build_model(cfg_for("resnet18_cifar")).num_parameters == 11_173_962
    assert build_model(cfg_for("resnet34_cifar")).num_parameters == 21_282_122
    assert 1_000_000 < build_model(cfg_for("simple_cnn")).num_parameters < 1_500_000


def test_resnet_has_no_maxpool_in_stem():
    m = build_model(cfg_for("resnet18_cifar"))
    assert not any(isinstance(mod, torch.nn.MaxPool2d) for mod in m.modules())


def test_stem_preserves_resolution():
    m = build_model(cfg_for("resnet18_cifar"))
    assert m.stem(torch.randn(1, 3, 32, 32)).shape[-2:] == (32, 32)


def test_zero_init_residual_zeros_last_bn():
    m = build_model(cfg_for("resnet18_cifar", zero_init_residual=True))
    for blk in all_basic_blocks(m):
        assert torch.all(blk.bn2.weight == 0)


def test_zero_init_residual_false_leaves_last_bn_at_one():
    m = build_model(cfg_for("resnet18_cifar", zero_init_residual=False))
    for blk in all_basic_blocks(m):
        assert torch.all(blk.bn2.weight == 1)


def test_gradients_reach_every_parameter():
    m = build_model(cfg_for("resnet18_cifar"))
    m(torch.randn(2, 3, 32, 32)).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


def test_unknown_model_name_raises_with_suggestions():
    with pytest.raises(ValueError, match="resnet18_cifar"):
        load_config("configs/base.yaml", ["model.name=resnet18"])  # ImageNet-style, not registered


def test_model_can_overfit_ten_samples():
    """The single most valuable model test: 200 steps on 10 fixed images
    must drive loss below 0.01. If it can't, the architecture or the
    init is broken and no amount of tuning will save it."""
    m = build_model(cfg_for("resnet18_cifar"))
    x, y = torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.01
