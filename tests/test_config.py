from dataclasses import FrozenInstanceError
from pathlib import Path

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


def test_batch_aug_prob_positive_requires_an_alpha():
    with pytest.raises(ValueError, match="batch_aug_prob"):
        load_config(
            "configs/base.yaml",
            ["aug.batch_aug_prob=0.5", "aug.mixup_alpha=0.0", "aug.cutmix_alpha=0.0"],
        )


def test_batch_aug_prob_zero_requires_no_alpha():
    with pytest.raises(ValueError, match="batch_aug_prob"):
        load_config(
            "configs/base.yaml",
            ["aug.batch_aug_prob=0.0", "aug.mixup_alpha=0.2", "aug.cutmix_alpha=0.0"],
        )


def test_override_coerces_tuple_of_int():
    cfg = load_config("configs/base.yaml", ["optim.step_milestones=30,60,90"])
    assert cfg.optim.step_milestones == (30, 60, 90)


def test_override_coerces_path():
    cfg = load_config("configs/base.yaml", ["data.root=/tmp/x"])
    assert cfg.data.root == Path("/tmp/x")


def test_override_coerces_none():
    cfg = load_config("configs/base.yaml", ["log.run_name=none"])
    assert cfg.log.run_name is None


def test_simplecnn_config_loads():
    cfg = load_config("configs/simplecnn.yaml")
    assert cfg.model.name == "simple_cnn"


def test_yaml_scientific_notation_becomes_float():
    """Regression: YAML 1.1 parses `5e-4` as a *str* (a plain float needs a dot
    and a signed exponent). Uncoerced, weight_decay reached SGD as a string and
    only raised at the first optimizer step."""
    cfg = load_config("configs/base.yaml")
    assert isinstance(cfg.optim.weight_decay, float)
    assert cfg.optim.weight_decay == pytest.approx(5e-4)


def test_quoted_yaml_numbers_are_coerced():
    cfg = load_config("configs/base.yaml", ["optim.learning_rate=0.05", "train.epochs=7"])
    assert isinstance(cfg.optim.learning_rate, float) and cfg.optim.learning_rate == 0.05
    assert isinstance(cfg.train.epochs, int) and cfg.train.epochs == 7


def test_inherits_resolves_a_chain_not_just_one_hop():
    """Regression: `ablation -> resnet18 -> base` used to fail.

    Resolving one level left the parent's own `inherits:` key in the merged
    dict, which surfaced as `Unknown config key: inherits`.
    """
    cfg = load_config("configs/ablations/abl_wd_1e4.yaml")
    assert cfg.optim.weight_decay == pytest.approx(1e-4)  # the ablation's own delta
    assert cfg.optim.label_smoothing == pytest.approx(0.1)  # from resnet18.yaml
    assert cfg.train.epochs == 120  # from base.yaml


def test_inheritance_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("inherits: b.yaml\n")
    (tmp_path / "b.yaml").write_text("inherits: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_resnet18_recipe_carries_the_full_94_recipe():
    """C5.3: the recipe is asserted here, not restated in the YAML.

    resnet18.yaml holds one line (`label_smoothing`); everything else is
    inherited. That is only safe if something pins what it inherits *to*.
    """
    cfg = load_config("configs/resnet18.yaml")
    assert cfg.model.name == "resnet18_cifar"
    assert (cfg.optim.optimizer, cfg.optim.nesterov) == ("sgd", True)
    assert cfg.optim.learning_rate == pytest.approx(0.1)
    assert cfg.optim.weight_decay == pytest.approx(5e-4)
    assert cfg.optim.no_wd_on_bn_bias is True
    assert (cfg.optim.scheduler, cfg.optim.warmup_epochs) == ("cosine", 5)
    assert cfg.optim.label_smoothing == pytest.approx(0.1)
    assert (cfg.train.epochs, cfg.data.batch_size, cfg.train.amp) == (120, 128, True)
    assert (cfg.aug.random_crop_padding, cfg.aug.horizontal_flip_p) == (4, 0.5)
