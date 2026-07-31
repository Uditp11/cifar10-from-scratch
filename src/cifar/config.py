"""Config dataclasses, YAML loader, validation, and hashing.

See `docs/ARCHITECTURE.md` §1 for the frozen contract this module implements.
"""

from __future__ import annotations

import hashlib
import json
import types
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml

__all__ = [
    "DataConfig",
    "AugConfig",
    "ModelConfig",
    "OptimConfig",
    "TrainConfig",
    "ReproConfig",
    "LogConfig",
    "Config",
    "load_config",
    "config_to_dict",
    "config_hash",
]


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("./data")
    val_size: int = 5000
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class AugConfig:
    random_crop_padding: int = 4  # 0 disables
    horizontal_flip_p: float = 0.5
    cutout_holes: int = 0  # 0 disables
    cutout_length: int = 16
    randaugment_n: int = 0  # 0 disables
    randaugment_m: int = 9
    mixup_alpha: float = 0.0  # 0 disables
    cutmix_alpha: float = 0.0  # 0 disables
    batch_aug_prob: float = 0.0  # P(apply mixup/cutmix); 0 with both alphas 0 = off


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18_cifar"
    num_classes: int = 10
    width_multiplier: float = 1.0
    dropout: float = 0.0
    zero_init_residual: bool = True


@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "sgd"  # sgd | adamw
    learning_rate: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    no_wd_on_bn_bias: bool = True
    scheduler: str = "cosine"  # cosine | step | onecycle | constant
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_learning_rate: float = 0.0
    step_milestones: tuple[int, ...] = (60, 90)
    step_gamma: float = 0.1
    label_smoothing: float = 0.0
    grad_clip_norm: float = 0.0  # 0 disables


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    amp: bool = True
    channels_last: bool = True
    ema_decay: float = 0.0  # 0 disables
    ema_warmup_epochs: int = 0
    eval_every: int = 1
    save_every: int = 1
    keep_last_n_checkpoints: int = 2


@dataclass(frozen=True)
class ReproConfig:
    seed: int = 42
    deterministic: bool = False
    cudnn_benchmark: bool = True


@dataclass(frozen=True)
class LogConfig:
    run_root: Path = Path("./runs")
    run_name: str | None = None  # None -> auto: {model}_{timestamp}
    tensorboard: bool = True
    step_interval: int = 0  # 0 -> epoch-level logging only
    log_confusion_matrix: bool = True


@dataclass(frozen=True)
class Config:
    data: DataConfig
    aug: AugConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    repro: ReproConfig
    log: LogConfig
    device: str = "auto"  # auto | cuda | cpu — "auto" resolves at load time
    notes: str = ""


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "aug": AugConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "train": TrainConfig,
    "repro": ReproConfig,
    "log": LogConfig,
}
_TOP_LEVEL_SCALARS = {"device", "notes"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(raw: str, annotation: Any) -> Any:
    """Coerce a CLI override string to the type a config field expects."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if raw.lower() == "none":
            return None
        if len(non_none) == 1:
            return _coerce(raw, non_none[0])
        raise ValueError(f"cannot coerce override value {raw!r} to {annotation}")

    if origin is tuple:
        elem_type = args[0] if args else str
        items = [item.strip() for item in raw.split(",")] if raw else []
        return tuple(_coerce(item, elem_type) for item in items)

    if annotation is bool:
        low = raw.lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
        raise ValueError(f"cannot coerce override value {raw!r} to bool")

    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is Path:
        return Path(raw)
    return raw


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Malformed override (expected key=value): {override!r}")
    key, raw_value = override.split("=", 1)
    parts = key.split(".")

    if len(parts) == 1:
        (name,) = parts
        if name not in _TOP_LEVEL_SCALARS:
            raise ValueError(f"Unknown config key: {key}")
        hints = typing.get_type_hints(Config)
        data[name] = _coerce(raw_value, hints[name])
        return

    if len(parts) == 2:
        section, field_name = parts
        if section not in _SECTIONS:
            raise ValueError(f"Unknown config key: {key}")
        cls = _SECTIONS[section]
        valid_fields = {f.name for f in fields(cls)}
        if field_name not in valid_fields:
            raise ValueError(f"Unknown config key: {key}")
        hints = typing.get_type_hints(cls)
        coerced = _coerce(raw_value, hints[field_name])
        data.setdefault(section, {})[field_name] = coerced
        return

    raise ValueError(f"Unknown config key: {key}")


def _build_section(cls: type, data: dict[str, Any], section_name: str) -> Any:
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(data) - valid_fields
    if unknown:
        key = next(iter(unknown))
        raise ValueError(f"Unknown config key: {section_name}.{key}")

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        annotation = hints[f.name]
        if typing.get_origin(annotation) is tuple and isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, str) and annotation is not str:
            # YAML 1.1 parses `5e-4` as a *string* -- a plain float needs a dot
            # and a signed exponent (`5.0e-4`). Left uncoerced, `weight_decay`
            # reaches SGD as a str and only explodes at the first optimizer
            # step, an hour into a run. Coerce every non-str field the same way
            # CLI overrides are coerced, so the whole class of trap is closed.
            value = _coerce(value, annotation)
        kwargs[f.name] = value
    return cls(**kwargs)


def _resolve_device(raw: str) -> str:
    """Resolve `auto | cuda | cpu` to the concrete device that lands in Config.

    An explicit `cuda` with no CUDA available raises rather than silently
    downgrading to CPU — a 50x slowdown that should never happen quietly.
    """
    if raw not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device ({raw!r}) must be one of 'auto', 'cuda', 'cpu'")
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "device ('cuda') requires torch.cuda.is_available() to be True. "
            "If a GPU is present, this is commonly a CPU-only torch wheel "
            "(the default on Windows/PyPI) — reinstall from "
            "https://pytorch.org/get-started/locally/ with a cuXXX build."
        )
    return raw


def _dict_to_config(data: dict[str, Any]) -> Config:
    unknown_top = set(data) - set(_SECTIONS) - _TOP_LEVEL_SCALARS
    if unknown_top:
        raise ValueError(f"Unknown config key: {next(iter(unknown_top))}")

    kwargs: dict[str, Any] = {
        name: _build_section(cls, data.get(name, {}), name) for name, cls in _SECTIONS.items()
    }
    kwargs["device"] = _resolve_device(data.get("device", "auto"))
    if "notes" in data:
        kwargs["notes"] = data["notes"]
    return Config(**kwargs)


_KNOWN_OPTIMIZERS = {"sgd", "adamw"}
_KNOWN_SCHEDULERS = {"cosine", "step", "onecycle", "constant"}


def _validate(cfg: Config) -> None:
    """Check every invariant from `ARCHITECTURE.md` §1. Raises ValueError on failure."""
    if not (0 <= cfg.optim.warmup_epochs < cfg.train.epochs):
        raise ValueError(
            f"warmup_epochs ({cfg.optim.warmup_epochs}) must satisfy "
            f"0 <= warmup_epochs < epochs ({cfg.train.epochs})"
        )
    if not (0 < cfg.data.val_size < 50000):
        raise ValueError(f"val_size ({cfg.data.val_size}) must satisfy 0 < val_size < 50000")
    if cfg.aug.batch_aug_prob > 0 and cfg.aug.mixup_alpha == 0 and cfg.aug.cutmix_alpha == 0:
        raise ValueError(
            f"batch_aug_prob ({cfg.aug.batch_aug_prob}) is > 0 but both mixup_alpha and "
            f"cutmix_alpha are 0 — a probability with nothing to apply"
        )
    if cfg.aug.batch_aug_prob == 0 and (cfg.aug.mixup_alpha > 0 or cfg.aug.cutmix_alpha > 0):
        raise ValueError(
            f"batch_aug_prob is 0 but mixup_alpha ({cfg.aug.mixup_alpha}) / cutmix_alpha "
            f"({cfg.aug.cutmix_alpha}) is > 0 — an alpha that can never fire"
        )
    if not cfg.model.name:
        raise ValueError(f"model.name ({cfg.model.name!r}) must be a non-empty string")
    from cifar.models import list_models  # local: cifar.models imports Config at module scope

    if cfg.model.name not in list_models():
        raise ValueError(
            f"model.name ({cfg.model.name!r}) is not registered. Registered models: {list_models()}"
        )
    if cfg.optim.optimizer not in _KNOWN_OPTIMIZERS:
        raise ValueError(f"optimizer ({cfg.optim.optimizer!r}) must be one of {_KNOWN_OPTIMIZERS}")
    if cfg.optim.scheduler not in _KNOWN_SCHEDULERS:
        raise ValueError(f"scheduler ({cfg.optim.scheduler!r}) must be one of {_KNOWN_SCHEDULERS}")


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML, merge onto base.yaml, apply dotted CLI overrides, validate.

    Overrides look like ["optim.learning_rate=0.05", "train.epochs=200"].
    Raises ValueError on unknown keys, type mismatches, or failed invariants.
    """
    path = Path(path)
    target = _read_yaml(path)
    inherits = target.pop("inherits", None)
    if inherits is not None:
        parent = _read_yaml(path.parent / inherits)
        merged = _deep_merge(parent, target)
    else:
        merged = target

    for override in overrides or []:
        _apply_override(merged, override)

    cfg = _dict_to_config(merged)
    _validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Plain, JSON-friendly dict of the config (Path values become strings)."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(asdict(cfg))


def config_hash(cfg: Config) -> str:
    """Stable 12-char hash of the config, excluding `notes` and `log`.

    Two runs with the same hash should produce the same result.
    """
    d = config_to_dict(cfg)
    d.pop("notes", None)
    d.pop("log", None)
    canonical = json.dumps(d, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
