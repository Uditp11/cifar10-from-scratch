"""Batch-level augmentation (Mixup/CutMix). See ARCHITECTURE.md section 3.

Runs on the collated batch, on whatever device it already lives on, between the
loader and the forward pass. Everything returns a `MixedBatch` -- including the
"nothing fired" case, which returns `lam == 1.0` with both target sets being the
same object -- so `Trainer` and `mixed_criterion` never branch on whether mixing
happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from cifar.config import Config

__all__ = ["MixedBatch", "apply_batch_aug"]


@dataclass
class MixedBatch:
    inputs: Tensor
    targets_a: Tensor
    targets_b: Tensor
    lam: float


def _sample_lam(alpha: float) -> float:
    """Draw the mixing coefficient from Beta(alpha, alpha)."""
    return float(torch.distributions.Beta(alpha, alpha).sample())


def _cutmix(x: Tensor, perm: Tensor, lam: float) -> tuple[Tensor, float]:
    """Paste one rectangle from the permuted batch, and report its *actual* area.

    The box is centred uniformly and clipped at the image border, so a box near
    an edge loses part of its area. `lam` is therefore recomputed from the
    clipped box: using the sampled value weights the loss for a rectangle that
    was never pasted, which is a real bug and a commonly copied one -- and it is
    invisible in a loss curve, because the loss stays smooth and merely wrong.

    Args:
        x: Batch, `(B, C, H, W)`.
        perm: Permutation of the batch dimension; the paste source.
        lam: Sampled Beta coefficient, defining the *intended* box area.

    Returns:
        `(pasted_batch, lam_from_actual_area)`.
    """
    height, width = x.shape[-2:]
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    center_y, center_x = int(torch.randint(height, ())), int(torch.randint(width, ()))

    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)

    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    return out, 1.0 - (y2 - y1) * (x2 - x1) / (height * width)


def apply_batch_aug(x: Tensor, y: Tensor, cfg: Config) -> MixedBatch:
    """Apply mixup/cutmix to a batch, chosen per `cfg.aug`.

    Fires with probability `cfg.aug.batch_aug_prob`. When both alphas are
    positive the two methods are chosen uniformly per batch, so a run mixes both
    without either being applied on top of the other.

    Args:
        x: Input batch, `(B, 3, 32, 32)`.
        y: Target batch, `(B,)`.
        cfg: Full config; reads `cfg.aug`.

    Returns:
        A `MixedBatch`. When no aug fires, `lam == 1.0` and `targets_a is
        targets_b`, letting downstream code skip branching.
    """
    aug = cfg.aug
    enabled = [
        (name, alpha)
        for name, alpha in (("mixup", aug.mixup_alpha), ("cutmix", aug.cutmix_alpha))
        if alpha > 0
    ]
    if not enabled or float(torch.rand(())) >= aug.batch_aug_prob:
        return MixedBatch(inputs=x, targets_a=y, targets_b=y, lam=1.0)

    name, alpha = enabled[int(torch.randint(len(enabled), ()))]
    perm = torch.randperm(x.size(0), device=x.device)
    lam = _sample_lam(alpha)
    if name == "mixup":
        mixed = lam * x + (1.0 - lam) * x[perm]
    else:
        mixed, lam = _cutmix(x, perm, lam)
    return MixedBatch(inputs=mixed, targets_a=y, targets_b=y[perm], lam=lam)
