"""Optimizer and per-step LR scheduler factories. See `docs/ARCHITECTURE.md` §5.

Two decisions are encoded here and nowhere else: BN/bias parameters are
excluded from weight decay (ADR-003) and every schedule advances once per
optimizer *step*, not per epoch (ADR-004).
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    SequentialLR,
)

from cifar.config import Config

__all__ = ["build_optimizer", "build_scheduler"]


def _param_groups(model: nn.Module, cfg: Config) -> list[dict]:
    """Split parameters into decayed and non-decayed groups.

    Everything with `ndim <= 1` -- BN scales and shifts, every bias -- carries
    no capacity that L2 is meant to constrain, and decaying it measurably hurts
    (ADR-003). One group when the behaviour is switched off, so the optimizer
    state stays as simple as the config asked for.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not cfg.optim.no_wd_on_bn_bias:
        return [{"params": params, "weight_decay": cfg.optim.weight_decay}]
    return [
        {"params": [p for p in params if p.ndim > 1], "weight_decay": cfg.optim.weight_decay},
        {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: Config) -> Optimizer:
    """Build the optimizer over weight-decay-split param groups.

    Args:
        model: Model whose trainable parameters are optimized.
        cfg: Full config; reads `cfg.optim`.

    Returns:
        An optimizer with two param groups when `cfg.optim.no_wd_on_bn_bias`,
        one otherwise. `weight_decay` lives on the groups, not the optimizer.
    """
    groups = _param_groups(model, cfg)
    match cfg.optim.optimizer:
        case "sgd":
            return SGD(
                groups,
                lr=cfg.optim.learning_rate,
                momentum=cfg.optim.momentum,
                nesterov=cfg.optim.nesterov and cfg.optim.momentum > 0,
            )
        case "adamw":
            return AdamW(groups, lr=cfg.optim.learning_rate)
        case unknown:  # pragma: no cover - load_config validates this first
            raise ValueError(f"Unknown optimizer {unknown!r}")


def build_scheduler(opt: Optimizer, cfg: Config, steps_per_epoch: int) -> LRScheduler:
    """Build the LR scheduler.

    Every horizon is expressed in optimizer steps (ADR-004): a per-epoch cosine
    is a 120-step staircase, which underperforms and interacts badly with
    warmup. `step_milestones` stay *absolute* epoch numbers -- they are shifted
    by `warmup_epochs` here because `SequentialLR` restarts the second
    schedule's step counter when it hands over.

    `onecycle` owns its own warmup phase, so it is returned directly with
    `pct_start` derived from `warmup_epochs` rather than being wrapped.

    Args:
        opt: Optimizer whose param groups will be stepped.
        cfg: Full config; reads `cfg.optim` and `cfg.train.epochs`.
        steps_per_epoch: `len(train_loader)`. Needed for per-step schedules.

    Returns:
        A scheduler whose `.step()` is called once per optimizer step.
    """
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch ({steps_per_epoch}) must be >= 1")

    warmup_epochs = cfg.optim.warmup_epochs
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if cfg.optim.scheduler == "onecycle":
        return OneCycleLR(
            opt,
            max_lr=cfg.optim.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_epochs / cfg.train.epochs if warmup_epochs else 0.3,
            div_factor=1.0 / cfg.optim.warmup_start_factor,
        )

    main_steps = total_steps - warmup_steps
    match cfg.optim.scheduler:
        case "cosine":
            main: LRScheduler = CosineAnnealingLR(
                opt, T_max=main_steps, eta_min=cfg.optim.min_learning_rate
            )
        case "step":
            main = MultiStepLR(
                opt,
                milestones=[
                    (m - warmup_epochs) * steps_per_epoch
                    for m in cfg.optim.step_milestones
                    if m > warmup_epochs
                ],
                gamma=cfg.optim.step_gamma,
            )
        case "constant":
            main = ConstantLR(opt, factor=1.0, total_iters=0)
        case unknown:  # pragma: no cover - load_config validates this first
            raise ValueError(f"Unknown scheduler {unknown!r}")

    if warmup_steps == 0:
        return main

    warmup = LinearLR(opt, start_factor=cfg.optim.warmup_start_factor, total_iters=warmup_steps)
    return SequentialLR(opt, schedulers=[warmup, main], milestones=[warmup_steps])
