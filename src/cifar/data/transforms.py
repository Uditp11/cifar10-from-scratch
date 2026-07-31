"""Image transform pipelines. See ARCHITECTURE.md sections 3 and 8.

Every optional stage (crop, flip, RandAugment, Cutout) is included only when
its config value enables it, so the fully-disabled train pipeline reduces to
exactly the eval pipeline -- ToTensor + Normalize, nothing else.
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T
from torch import Tensor

from cifar.config import Config

__all__ = ["build_train_transform", "build_eval_transform", "Cutout", "RandAugment"]

_IMAGE_SIZE = 32  # CIFAR-10 image side length


class Cutout:
    """Zeroes `n_holes` square regions of side `length`, applied AFTER normalization.

    Identity stub for sprint 1 -- implemented in sprint 6.
    """

    def __init__(self, n_holes: int, length: int) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: Tensor) -> Tensor:
        return img  # implemented in sprint 6


class RandAugment:
    """`n` random ops at magnitude `m`, applied before ToTensor.

    Identity stub for sprint 1 -- implemented in sprint 6.
    """

    def __init__(self, n: int, m: int) -> None:
        self.n = n
        self.m = m

    def __call__(self, img):
        return img  # implemented in sprint 6


def build_train_transform(cfg: Config) -> Callable:
    """Build the training transform.

    Stage order is load-bearing (ARCHITECTURE.md section 3): crop, flip,
    RandAugment, ToTensor, Normalize, Cutout.

    Args:
        cfg: Full config; reads `cfg.aug` and `cfg.data`.

    Returns:
        A callable PIL-image -> Tensor pipeline.
    """
    stages: list[Callable] = []
    if cfg.aug.random_crop_padding > 0:
        stages.append(
            T.RandomCrop(_IMAGE_SIZE, padding=cfg.aug.random_crop_padding, padding_mode="reflect")
        )
    if cfg.aug.horizontal_flip_p > 0:
        stages.append(T.RandomHorizontalFlip(p=cfg.aug.horizontal_flip_p))
    if cfg.aug.randaugment_n > 0:
        stages.append(RandAugment(cfg.aug.randaugment_n, cfg.aug.randaugment_m))
    stages.append(T.ToTensor())
    stages.append(T.Normalize(cfg.data.mean, cfg.data.std))
    if cfg.aug.cutout_holes > 0:
        stages.append(Cutout(cfg.aug.cutout_holes, cfg.aug.cutout_length))
    return T.Compose(stages)


def build_eval_transform(cfg: Config) -> Callable:
    """Build the eval transform: ToTensor -> Normalize. Nothing else, ever.

    Args:
        cfg: Full config; reads `cfg.data`.

    Returns:
        A callable PIL-image -> Tensor pipeline.
    """
    return T.Compose([T.ToTensor(), T.Normalize(cfg.data.mean, cfg.data.std)])
