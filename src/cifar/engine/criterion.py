"""Loss construction and the mixup-aware wrapper. See `docs/ARCHITECTURE.md` §5."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from cifar.config import Config
from cifar.data.batch_aug import MixedBatch

__all__ = ["build_criterion", "mixed_criterion"]


def build_criterion(cfg: Config) -> nn.Module:
    """Build the training loss.

    Args:
        cfg: Full config; reads `cfg.optim.label_smoothing`.

    Returns:
        Cross-entropy, optionally label-smoothed.
    """
    return nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)


def mixed_criterion(criterion: nn.Module, logits: Tensor, batch: MixedBatch) -> Tensor:
    """Interpolate the loss between a mixed batch's two target sets.

    Until sprint 6 fills in mixup/cutmix this is exactly plain cross-entropy:
    `apply_batch_aug` returns `lam == 1.0` with both target sets identical, so
    the second term is multiplied by zero. Writing it this way now means the
    trainer needs no change when mixing turns on.

    Args:
        criterion: Per-sample-reducing loss, e.g. from `build_criterion`.
        logits: `(B, C)` model outputs.
        batch: The `MixedBatch` the logits were computed from.

    Returns:
        Scalar loss `lam * L(logits, a) + (1 - lam) * L(logits, b)`.
    """
    loss_a = criterion(logits, batch.targets_a)
    if batch.lam == 1.0:
        return loss_a
    return batch.lam * loss_a + (1.0 - batch.lam) * criterion(logits, batch.targets_b)
