"""Running averages, top-k accuracy, and epoch timing. See `docs/ARCHITECTURE.md` §6."""

from __future__ import annotations

import time
from types import TracebackType

import torch
from torch import Tensor

__all__ = ["AverageMeter", "accuracy", "Timer"]


class AverageMeter:
    """Running mean of a scalar, weighted by the number of samples behind each update."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Fold `val` into the average as the mean of `n` samples."""
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        """Mean so far; 0.0 before the first update."""
        return self.sum / self.count if self.count else 0.0


def accuracy(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
    """Top-k accuracy in percent, one entry per k.

    Operates on logits, so k may exceed the number of *distinct classes in the
    batch* without trouble -- only the number of output columns bounds k.

    Args:
        logits: `(B, C)` unnormalized scores.
        targets: `(B,)` integer class labels.
        topk: The k values to report.

    Returns:
        One percentage per entry of `topk`, in the same order.
    """
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return [correct[:, : min(k, maxk)].any(dim=1).float().mean().item() * 100.0 for k in topk]


class Timer:
    """Context manager measuring wall-clock seconds into `.seconds`.

    Synchronizes CUDA on exit, since kernels are queued asynchronously and an
    unsynchronized timer measures how fast Python enqueued work, not how long
    the GPU took.
    """

    def __enter__(self) -> Timer:
        self.seconds = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start
