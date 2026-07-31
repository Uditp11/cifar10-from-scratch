"""Fast VGG-ish baseline. See `docs/sprints/02-models.md` C2.3."""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar.config import ModelConfig
from cifar.models.base import ModelBase, init_weights
from cifar.models.layers import conv3x3
from cifar.models.registry import register_model

__all__ = ["SimpleCNN"]

_CHANNELS = (64, 128, 256)


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        conv3x3(in_ch, out_ch),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        conv3x3(out_ch, out_ch),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


@register_model("simple_cnn")
class SimpleCNN(ModelBase):
    """Three `[Conv-BN-ReLU x2 -> MaxPool]` blocks, GAP head. ~1.2M params."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_ch = 3
        blocks = []
        for out_ch in _CHANNELS:
            blocks.append(_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(_CHANNELS[-1], cfg.num_classes)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
