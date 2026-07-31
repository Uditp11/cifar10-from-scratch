"""Batch-level augmentation (Mixup/CutMix). See ARCHITECTURE.md section 3.

Sprint 1 stub: `apply_batch_aug` always returns an unmixed batch. Sprint 6
fills in the actual mixup/cutmix logic; the Trainer and `mixed_criterion`
already consume `MixedBatch`, so nothing downstream changes shape when it does.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from cifar.config import Config

__all__ = ["MixedBatch", "apply_batch_aug"]


@dataclass
class MixedBatch:
    inputs: Tensor
    targets_a: Tensor
    targets_b: Tensor
    lam: float


def apply_batch_aug(x: Tensor, y: Tensor, cfg: Config) -> MixedBatch:
    """Apply mixup/cutmix to a batch, chosen per `cfg.aug`.

    Args:
        x: Input batch, `(B, 3, 32, 32)`.
        y: Target batch, `(B,)`.
        cfg: Full config; reads `cfg.aug`.

    Returns:
        A `MixedBatch`. Sprint 1: no aug ever fires, so `lam == 1.0` and
        `targets_a is targets_b`, letting downstream code skip branching.
    """
    return MixedBatch(inputs=x, targets_a=y, targets_b=y, lam=1.0)
