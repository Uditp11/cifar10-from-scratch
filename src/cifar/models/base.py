"""Shared model introspection + init. See `docs/sprints/02-models.md` C2.4/C2.5."""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar.models.layers import BasicBlock

__all__ = ["ModelBase", "init_weights"]


def init_weights(module: nn.Module, zero_init_residual: bool = False) -> None:
    """Kaiming-init convs, unit-init BN. Optionally zero the last BN in each
    residual branch (`zero_init_residual`) so every branch starts as an
    identity function -- see the "bag of tricks" paper (He et al., 2018),
    worth ~0.2-0.3% on the final number.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual:
        for m in module.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


class ModelBase(nn.Module):
    """Adds `.num_parameters` and `.summary()` to a plain `nn.Module`.

    `summary()` hooks each top-level child module and runs one dummy
    forward pass, so it works for any architecture whose top-level
    `named_children()` are applied in forward order (true of both
    `SimpleCNN` and `ResNetCIFAR`) without hand-writing a shape trace.
    """

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self, input_size: tuple[int, int, int, int] = (1, 3, 32, 32)) -> str:
        device = next(self.parameters()).device
        x = torch.randn(*input_size, device=device)
        rows: list[tuple[str, tuple[int, ...], int]] = []

        def make_hook(name: str):
            def hook(mod: nn.Module, _inp, out: torch.Tensor) -> None:
                rows.append((name, tuple(out.shape), sum(p.numel() for p in mod.parameters())))

            return hook

        handles = [c.register_forward_hook(make_hook(n)) for n, c in self.named_children()]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(x)
        self.train(was_training)
        for h in handles:
            h.remove()

        lines = [f"{name:<10} {str(shape):<22} {params:,}" for name, shape, params in rows]
        lines.append(f"{'total':<10} {'':<22} {self.num_parameters:,}")
        return "\n".join(lines)
