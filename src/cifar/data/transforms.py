"""Image transform pipelines. See ARCHITECTURE.md sections 3 and 8.

Every optional stage (crop, flip, RandAugment, Cutout) is included only when
its config value enables it, so the fully-disabled train pipeline reduces to
exactly the eval pipeline -- ToTensor + Normalize, nothing else.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor

from cifar.config import Config

__all__ = ["build_train_transform", "build_eval_transform", "Cutout", "RandAugment"]

_IMAGE_SIZE = 32  # CIFAR-10 image side length

# RandAugment's magnitude axis is fixed at 0-30 by the paper; `randaugment_m`
# picks a point on it. Each op's entry is (magnitude at m=30, is signed) -- the
# ranges *define* the ops rather than tune them, which is why they live here and
# not in AugConfig. `cfg.aug.randaugment_m` is the tunable.
_MAGNITUDE_SCALE = 30.0
_RANDAUGMENT_OPS: dict[str, tuple[float, bool]] = {
    "identity": (0.0, False),
    "autocontrast": (0.0, False),
    "equalize": (0.0, False),
    "rotate": (30.0, True),  # degrees
    "solarize": (256.0, False),  # threshold counts *down* from 256
    "color": (0.9, True),  # enhancement factor 1 +/- level
    "posterize": (4.0, False),  # bits count down from 8
    "contrast": (0.9, True),
    "brightness": (0.9, True),
    "sharpness": (0.9, True),
    "shear_x": (0.3, True),
    "shear_y": (0.3, True),
    "translate_x": (0.3, True),  # fraction of image width
    "translate_y": (0.3, True),
}


class Cutout:
    """Zeroes `n_holes` square regions of side `length`.

    Applied AFTER normalization (ARCHITECTURE.md section 8), so a zeroed pixel
    is the dataset mean pixel rather than black -- the hole removes information
    without injecting an out-of-distribution value.

    Hole centres are uniform over the whole image and the square is clipped at
    the boundary, so a hole near an edge covers less than `length**2`. That is
    the paper's behaviour, not an oversight: clamping the centre inward instead
    would make edge pixels systematically less likely to be occluded.
    """

    def __init__(self, n_holes: int, length: int) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: Tensor) -> Tensor:
        """Zero `n_holes` clipped squares of a normalized CHW tensor.

        Args:
            img: Normalized image tensor, `(C, H, W)`.

        Returns:
            A new tensor; the input is never modified in place.
        """
        if self.n_holes <= 0 or self.length <= 0:
            return img
        height, width = img.shape[-2:]
        half = self.length // 2
        out = img.clone()
        for _ in range(self.n_holes):
            center_y = int(torch.randint(height, ()))
            center_x = int(torch.randint(width, ()))
            out[
                ...,
                max(center_y - half, 0) : min(center_y + half, height),
                max(center_x - half, 0) : min(center_x + half, width),
            ] = 0.0
        return out


def _apply_op(img: Image.Image, name: str, level: float) -> Image.Image:
    """Apply one RandAugment op at an already-scaled, already-signed magnitude."""
    width, height = img.size
    match name:
        case "identity":
            return img
        case "autocontrast":
            return ImageOps.autocontrast(img)
        case "equalize":
            return ImageOps.equalize(img)
        case "rotate":
            return img.rotate(level)
        case "solarize":
            return ImageOps.solarize(img, int(256 - level))
        case "posterize":
            return ImageOps.posterize(img, 8 - int(level))
        case "color":
            return ImageEnhance.Color(img).enhance(1.0 + level)
        case "contrast":
            return ImageEnhance.Contrast(img).enhance(1.0 + level)
        case "brightness":
            return ImageEnhance.Brightness(img).enhance(1.0 + level)
        case "sharpness":
            return ImageEnhance.Sharpness(img).enhance(1.0 + level)
        case "shear_x":
            return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))
        case "shear_y":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))
        case "translate_x":
            return img.transform(img.size, Image.AFFINE, (1, 0, level * width, 0, 1, 0))
        case "translate_y":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level * height))
        case _:
            raise ValueError(f"unknown RandAugment op: {name!r}")


class RandAugment:
    """`n` ops sampled uniformly with replacement, each at magnitude `m`.

    Operates on the PIL image, so it sits before `ToTensor()` in the pipeline.
    Unlike AutoAugment there is no learned policy and no per-op probability:
    every sampled op fires, and the only knobs are `n` and `m`.

    Signed ops (rotate, shear, translate, the enhancement ops) get a random
    direction per application; `m` sets the magnitude, never the sign.
    """

    def __init__(self, n: int, m: int) -> None:
        if not 0 <= m <= _MAGNITUDE_SCALE:
            raise ValueError(f"randaugment_m ({m}) must satisfy 0 <= m <= {_MAGNITUDE_SCALE:g}")
        self.n = n
        self.m = m

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply `n` randomly chosen ops in sequence.

        Args:
            img: RGB PIL image.

        Returns:
            An RGB PIL image of the same size and mode.
        """
        for name in random.choices(list(_RANDAUGMENT_OPS), k=self.n):
            max_magnitude, signed = _RANDAUGMENT_OPS[name]
            level = max_magnitude * self.m / _MAGNITUDE_SCALE
            if signed and random.random() < 0.5:
                level = -level
            img = _apply_op(img, name, level)
        return img


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
