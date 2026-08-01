"""The training loop. See `docs/ARCHITECTURE.md` §5.

`__init__` takes a `(train, val)` 2-tuple, never a 3-tuple. That is the whole
protection against test-set leakage: the trainer cannot select on, early-stop
on, or even read the test set, because it was never handed one.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cifar.config import Config, config_hash
from cifar.data.batch_aug import apply_batch_aug
from cifar.data.dataset import CLASS_NAMES
from cifar.engine.checkpoint import load_checkpoint, save_checkpoint
from cifar.engine.criterion import build_criterion, mixed_criterion
from cifar.engine.evaluator import EvalResult, evaluate
from cifar.engine.optim import build_optimizer, build_scheduler
from cifar.seeding import capture_rng_state, restore_rng_state
from cifar.tracking.logger import Logger
from cifar.tracking.meters import AverageMeter, Timer, accuracy
from cifar.tracking.run import Run
from cifar.utils.viz import plot_confusion_matrix, plot_curves, plot_lr_schedule

__all__ = ["EpochMetrics", "Trainer"]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float
    epoch_seconds: float


class Trainer:
    """Trains one model against one (train, val) pair, into one run directory."""

    def __init__(
        self,
        model: nn.Module,
        loaders: tuple[DataLoader, DataLoader],
        cfg: Config,
        run: Run,
    ) -> None:
        """Wire up model, data, optimizer, scheduler, AMP scaler and logging.

        Args:
            model: Already on `cfg.device` (see `build_model`).
            loaders: `(train, val)`. A 3-tuple raises on unpacking -- deliberately.
            cfg: Full config.
            run: The run directory this training writes into.
        """
        self.train_loader, self.val_loader = loaders
        self.model = model
        self.cfg = cfg
        self.run = run
        self.logger = Logger(run, cfg)
        self.device = torch.device(cfg.device)

        self.criterion = build_criterion(cfg)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, len(self.train_loader))
        # GradScaler is a CUDA mechanism; on CPU, autocast still works (bf16)
        # but there is no fp16 underflow to rescale away.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=cfg.train.amp and self.device.type == "cuda"
        )

        self._log_optimizer()

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = -1
        # Kept so the end-of-run confusion matrix costs no extra forward pass.
        self._last_eval: EvalResult | None = None

    def _log_optimizer(self) -> None:
        """Record the param-group split and momentum flags before epoch 0.

        Weight decay leaking onto BN and bias is the single most common cause of
        a CIFAR ResNet stalling ~1% low, and it is invisible in every metric a
        run produces -- the curves look fine, the number is just worse. The only
        cheap defence is printing what the optimizer actually got.

        For ResNet-18-CIFAR the no-decay group holds **41** tensors: 20 BN
        layers x (scale + shift) + the classifier bias, with all 20 convs
        bias-free. A materially different count means the split is not doing
        what ADR-003 says it does.
        """
        groups = ", ".join(
            f"{len(g['params'])} tensors @ wd {g['weight_decay']:g}"
            for g in self.optimizer.param_groups
        )
        first = self.optimizer.param_groups[0]
        flags = "".join(f" {k} {first[k]!r}" for k in ("momentum", "nesterov") if k in first)
        self.logger.info(f"optimizer {type(self.optimizer).__name__} [{groups}]{flags}")

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one full pass over the training loader.

        The scheduler steps once per optimizer step, not once here (ADR-004).

        Args:
            epoch: 0-based epoch index, for logging only.

        Returns:
            `{"train_loss", "train_acc"}`, averaged over samples.
        """
        self.model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.cfg.train.channels_last:
                x = x.to(memory_format=torch.channels_last)

            batch = apply_batch_aug(x, y, self.cfg)
            with torch.autocast(self.device.type, enabled=self.cfg.train.amp):
                logits = self.model(batch.inputs)
                loss = mixed_criterion(self.criterion, logits, batch)

            self.scaler.scale(loss).backward()
            if self.cfg.optim.grad_clip_norm > 0:
                # Gradients must be unscaled before their norm means anything.
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            n = y.numel()
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(logits.detach().float(), batch.targets_a)[0], n)

        return {"train_loss": loss_meter.avg, "train_acc": acc_meter.avg}

    def validate(self, epoch: int) -> dict[str, float]:
        """Evaluate on the validation loader.

        Args:
            epoch: 0-based epoch index, for logging only.

        Returns:
            `{"val_loss", "val_acc"}`.
        """
        result = evaluate(self.model, self.val_loader, self.cfg)
        self._last_eval = result
        return {"val_loss": result.loss, "val_acc": result.top1}

    def resume_from(self, path: Path) -> None:
        """Restore model, optimizer, scheduler, scaler and RNG state from a checkpoint.

        Training continues from `epoch + 1`. Restoring RNG state is what makes
        the resumed trajectory match an uninterrupted one.

        Args:
            path: A `last.pth` (or any checkpoint) from a previous run.
        """
        meta = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.cfg.device,
        )
        if meta["cfg_hash"] != config_hash(self.cfg):
            warnings.warn(
                f"Checkpoint {path} was written under config hash {meta['cfg_hash']} but the "
                f"current config hashes to {config_hash(self.cfg)}. Resuming across a config "
                "change is legitimate, but this run's metrics will not describe one setup.",
                stacklevel=2,
            )
        restore_rng_state(meta["rng_state"])
        self.start_epoch = meta["epoch"] + 1
        self.best_val_acc = meta["best_metric"]
        self.logger.info(f"Resumed from {path} at epoch {self.start_epoch}")

    def _save(self, name: str, epoch: int) -> None:
        save_checkpoint(
            self.run.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=None,  # sprint 7
            epoch=epoch,
            best_metric=self.best_val_acc,
            cfg=self.cfg,
            rng_state=capture_rng_state(),
        )

    def fit(self) -> dict[str, float]:
        """Train to `cfg.train.epochs`, validating, logging and checkpointing.

        Ctrl-C exits cleanly and leaves the last *epoch-boundary* `last.pth`
        untouched, so an interrupted run resumes from a consistent point rather
        than a half-trained one (ADR-009).

        Returns:
            `{"best_val_acc", "best_epoch", "final_val_acc", "epochs_completed"}`.
        """
        self.run.write_manifest()
        # Before epoch 0: a schedule that is wrong is wrong now, and one second
        # of plotting beats discovering it an hour in.
        self.logger.log_figure(
            "lr_schedule", plot_lr_schedule(self.cfg, len(self.train_loader)), step=0
        )
        val: dict[str, float] = {"val_loss": math.nan, "val_acc": math.nan}
        # Two different "last epoch" notions, and conflating them is the bug
        # ADR-009 fixes: `completed` is what was trained, `checkpointed` is what
        # a resume would actually pick up. Neither is the epoch that happened to
        # be running when the process stopped.
        completed = checkpointed = self.start_epoch - 1
        status = "completed"

        try:
            for epoch in range(self.start_epoch, self.cfg.train.epochs):
                with Timer() as timer:
                    train = self.train_one_epoch(epoch)
                    # Always validate on the last epoch, so `final_val_acc` is
                    # a real number whatever `eval_every` is set to.
                    is_eval_epoch = (epoch + 1) % self.cfg.train.eval_every == 0 or (
                        epoch == self.cfg.train.epochs - 1
                    )
                    val = (
                        self.validate(epoch)
                        if is_eval_epoch
                        else {"val_loss": math.nan, "val_acc": math.nan}
                    )

                self.logger.log_epoch(
                    EpochMetrics(
                        epoch=epoch,
                        **train,
                        **val,
                        learning_rate=self.optimizer.param_groups[0]["lr"],
                        epoch_seconds=timer.seconds,
                    )
                )
                completed = epoch

                improved = is_eval_epoch and val["val_acc"] > self.best_val_acc
                if improved:
                    self.best_val_acc = val["val_acc"]
                    self.best_epoch = epoch
                    self._save("best.pth", epoch)
                if (epoch + 1) % self.cfg.train.save_every == 0:
                    self._save("last.pth", epoch)
                    checkpointed = epoch

        except KeyboardInterrupt:
            # Deliberately no save here: the partial epoch's weights, scheduler
            # and RNG do not agree with each other, and writing them over a good
            # last.pth is how a resumed run silently skips an epoch (ADR-009).
            status = "interrupted"
            self.logger.info(
                f"\nInterrupted during epoch {epoch + 1}; last.pth holds epoch "
                f"{checkpointed + 1}, resume continues from epoch {checkpointed + 2}"
            )
        except Exception:
            # A crashed run must not leave a manifest that says "running"
            # forever; finalize below still runs, then the error propagates.
            status = "failed"
            raise
        finally:
            summary = {
                "best_val_acc": self.best_val_acc,
                "best_epoch": self.best_epoch,
                "final_val_acc": val["val_acc"],
                "epochs_completed": completed + 1,
            }
            # Finalize first: figure generation must never be what stops a
            # manifest from being stamped, and it can raise on its own.
            self.run.finalize(summary, status=status)
            self._write_figures(completed)
            self.logger.close()

        return summary

    def _write_figures(self, epoch: int) -> None:
        """Draw the end-of-run figures, if any epoch actually got logged.

        A run that dies inside its first epoch has nothing to plot, and raising
        from here would replace the real traceback with a plotting error.
        """
        metrics_path = self.run.dir / "metrics.jsonl"
        if not metrics_path.exists() or not metrics_path.read_text().strip():
            return
        self.logger.log_figure("curves", plot_curves(self.run.dir / "metrics.jsonl"), step=epoch)
        if self.cfg.log.log_confusion_matrix and self._last_eval is not None:
            self.logger.log_figure(
                "confusion_matrix",
                plot_confusion_matrix(self._last_eval.confusion_matrix, list(CLASS_NAMES)),
                step=epoch,
            )
