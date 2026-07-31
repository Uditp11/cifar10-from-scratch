"""Model registry and architectures. See `docs/ARCHITECTURE.md` §4.

Importing this package registers every model (the submodule imports below
run each `@register_model` decorator), so `build_model`/`list_models` are
complete as soon as `cifar.models` is imported.
"""

from __future__ import annotations

from cifar.models.registry import build_model, list_models, register_model
from cifar.models.resnet import ResNet18CIFAR, ResNet34CIFAR, ResNetCIFAR  # noqa: F401
from cifar.models.simple_cnn import SimpleCNN  # noqa: F401

__all__ = ["build_model", "list_models", "register_model"]
