"""Model evaluation. See `docs/ARCHITECTURE.md` §5.

One function serves both the trainer's per-epoch validation and `scripts/eval.py`,
so a val number and a test number are never computed by two different code paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cifar.config import Config
from cifar.data.dataset import CLASS_NAMES
from cifar.engine.criterion import build_criterion
from cifar.tracking.meters import AverageMeter

__all__ = ["EvalResult", "evaluate"]

_NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class EvalResult:
    loss: float
    top1: float
    top5: float
    per_class_acc: dict[str, float]
    confusion_matrix: np.ndarray  # (10, 10), rows = true
    num_samples: int


def evaluate(model: nn.Module, loader: DataLoader, cfg: Config, *, tta: bool = False) -> EvalResult:
    """Evaluate `model` over `loader` and summarize it.

    Accuracies are derived from the confusion matrix rather than accumulated
    separately, so top-1, per-class accuracy and the matrix cannot disagree.

    The model's training mode is restored on exit -- `validate` is called from
    inside `fit`, and silently leaving the model in eval mode would poison the
    rest of training with frozen BN statistics.

    Args:
        model: Model to evaluate.
        loader: Any eval loader. `Trainer` only ever passes val.
        cfg: Full config; reads `cfg.device` and `cfg.train`.
        tta: Average softmax over {identity, horizontal flip}. Sprint 7.

    Returns:
        An `EvalResult` covering the whole loader.
    """
    if tta:
        raise NotImplementedError("TTA lands in sprint 7")

    device = torch.device(cfg.device)
    criterion = build_criterion(cfg)
    loss_meter = AverageMeter()
    confusion = torch.zeros(_NUM_CLASSES, _NUM_CLASSES, dtype=torch.long, device=device)
    top5_correct = 0
    num_samples = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if cfg.train.channels_last:
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast(device.type, enabled=cfg.train.amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                logits = logits.float()
                loss_meter.update(loss.item(), y.numel())
                top5_correct += logits.topk(5, dim=1).indices.eq(y.view(-1, 1)).any(1).sum().item()
                confusion += torch.bincount(
                    y * _NUM_CLASSES + logits.argmax(1), minlength=_NUM_CLASSES**2
                ).view(_NUM_CLASSES, _NUM_CLASSES)
                num_samples += y.numel()
    finally:
        model.train(was_training)

    cm = confusion.cpu().numpy()
    per_class_total = cm.sum(axis=1)
    per_class = np.divide(
        cm.diagonal() * 100.0,
        per_class_total,
        out=np.zeros(_NUM_CLASSES),
        where=per_class_total > 0,
    )
    return EvalResult(
        loss=loss_meter.avg,
        top1=float(cm.diagonal().sum()) / num_samples * 100.0 if num_samples else 0.0,
        top5=top5_correct / num_samples * 100.0 if num_samples else 0.0,
        per_class_acc=dict(zip(CLASS_NAMES, (float(v) for v in per_class))),
        confusion_matrix=cm,
        num_samples=num_samples,
    )
