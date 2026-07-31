"""CIFAR ResNet-18/34. See `docs/sprints/02-models.md` C2.4 for the stem
subtlety this exists to get right.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar.config import ModelConfig
from cifar.models.base import ModelBase, init_weights
from cifar.models.layers import BasicBlock, conv3x3
from cifar.models.registry import register_model

__all__ = ["ResNetCIFAR", "ResNet18CIFAR", "ResNet34CIFAR"]

_BASE_CHANNELS = (64, 128, 256, 512)
_STRIDES = (1, 2, 2, 2)


def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    ch = in_ch
    for s in strides:
        layers.append(BasicBlock(ch, out_ch, s))
        ch = out_ch * BasicBlock.expansion
    return nn.Sequential(*layers)


class ResNetCIFAR(ModelBase):
    """ResNet with the CIFAR stem: 3x3 stride-1 conv, no maxpool.

    Applied to a 32x32 image, an ImageNet-style 7x7 stride-2 stem + maxpool
    throws away most of the spatial signal before the first block. This stem
    keeps stage 1 at full 32x32 resolution instead: 32 -> 32 -> 16 -> 8 -> 4
    across the four stages (strides 1, 2, 2, 2).
    """

    def __init__(self, cfg: ModelConfig, blocks_per_stage: tuple[int, int, int, int]) -> None:
        super().__init__()
        channels = [max(1, round(c * cfg.width_multiplier)) for c in _BASE_CHANNELS]

        self.stem = nn.Sequential(
            conv3x3(3, channels[0]), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)
        )

        in_ch = channels[0]
        stages = []
        for out_ch, num_blocks, stride in zip(channels, blocks_per_stage, _STRIDES):
            stages.append(_make_stage(in_ch, out_ch, num_blocks, stride))
            in_ch = out_ch * BasicBlock.expansion
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(channels[-1] * BasicBlock.expansion, cfg.num_classes)
        init_weights(self, zero_init_residual=cfg.zero_init_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


@register_model("resnet18_cifar")
class ResNet18CIFAR(ResNetCIFAR):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, blocks_per_stage=(2, 2, 2, 2))


@register_model("resnet34_cifar")
class ResNet34CIFAR(ResNetCIFAR):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, blocks_per_stage=(3, 4, 6, 3))
