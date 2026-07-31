from dataclasses import FrozenInstanceError

import pytest

from cifar.config import config_hash, load_config


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg.optim.learning_rate == 0.1


def test_inheritance_merges_not_replaces():
    cfg = load_config("configs/resnet18.yaml")
    assert cfg.model.name == "resnet18_cifar"  # from child
    assert cfg.data.batch_size == 128  # from base


def test_override_applies_and_coerces_type():
    cfg = load_config(
        "configs/base.yaml",
        ["optim.learning_rate=0.05", "train.epochs=200", "train.amp=false"],
    )
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
    assert config_hash(a) == config_hash(b)  # notes excluded
    assert config_hash(a) != config_hash(c)
