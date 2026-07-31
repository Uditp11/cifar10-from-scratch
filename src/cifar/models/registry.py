"""Model registry. See `docs/ARCHITECTURE.md` §4.

`build_model` is the only place model construction, device placement, and
memory format live -- so every caller (train script, tests, model_summary)
gets identical behavior.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from cifar.config import Config

__all__ = ["register_model", "build_model", "list_models"]

_MODELS: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Class decorator. Registers a model constructor under `name`."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        _MODELS[name] = cls
        return cls

    return decorator


def list_models() -> list[str]:
    """Names of all registered models, sorted."""
    return sorted(_MODELS)


def build_model(cfg: Config) -> nn.Module:
    """Construct the model named by `cfg.model.name`.

    Weight init happens inside each model's own `__init__` (it needs
    architecture-specific knowledge, e.g. which BN sits last in a residual
    branch), so this only has to look the class up, construct it, and place
    it on the target device / memory format.

    Args:
        cfg: Full config; reads `cfg.model`, `cfg.device`, `cfg.train.channels_last`.

    Returns:
        The constructed model, on `cfg.device`.
    """
    name = cfg.model.name
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Registered models: {list_models()}")
    model = _MODELS[name](cfg.model).to(cfg.device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model
