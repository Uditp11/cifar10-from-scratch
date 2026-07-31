"""Shared conv/block building blocks. See `docs/ARCHITECTURE.md` §4 and
`docs/sprints/02-models.md` C2.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["conv3x3", "conv1x1", "BasicBlock"]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """3x3 conv, padding 1, no bias (BN follows)."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """1x1 conv, no bias (BN follows). Used for the projection shortcut."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convs with an identity or 1x1-projection shortcut.

    ReLU after the first BN and after the residual add, not after the second
    BN alone -- the add has to see the raw (unrectified) branch output.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_ch, out_ch * self.expansion, stride),
                nn.BatchNorm2d(out_ch * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)
