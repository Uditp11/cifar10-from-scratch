import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cifar.config import Config, load_config
from cifar.data.batch_aug import apply_batch_aug
from cifar.engine.checkpoint import load_checkpoint, save_checkpoint
from cifar.engine.criterion import build_criterion, mixed_criterion
from cifar.engine.evaluator import evaluate
from cifar.engine.optim import build_optimizer, build_scheduler
from cifar.engine.trainer import Trainer
from cifar.models import build_model
from cifar.seeding import capture_rng_state, seed_everything
from cifar.tracking.meters import AverageMeter, accuracy
from cifar.tracking.run import Run

NUM_CLASSES = 10


def cfg_for(tmp_path=None, **overrides) -> Config:
    """Tiny CPU config. Anything not overridden comes from configs/base.yaml."""
    flat = ["device=cpu", "model.name=simple_cnn", "train.amp=False", "train.channels_last=False"]
    flat += [f"{k.replace('__', '.')}={v}" for k, v in overrides.items()]
    if tmp_path is not None:
        flat.append(f"log.run_root={tmp_path}")
    return load_config("configs/base.yaml", flat)


def tiny_loaders(n: int = 64, batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    """Random 32x32 data with every class present, so top-5 and the 10x10
    confusion matrix are exercised without touching the real dataset."""
    x = torch.randn(n, 3, 32, 32)
    y = torch.arange(n) % NUM_CLASSES
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size), DataLoader(ds, batch_size=batch_size)


def step_and_get_lr(opt, sched) -> float:
    lr = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
    return lr


# --- meters ---------------------------------------------------------------


def test_average_meter_weights_by_count():
    m = AverageMeter()
    m.update(1.0, n=3)
    m.update(5.0, n=1)
    assert m.avg == pytest.approx(2.0)


def test_accuracy_top1_and_top5():
    logits = torch.tensor([[9.0, 1, 2, 3, 4, 5, 6, 7, 8, 0]])
    assert accuracy(logits, torch.tensor([0]), topk=(1, 5)) == [100.0, 100.0]
    assert accuracy(logits, torch.tensor([9]), topk=(1, 5)) == [0.0, 0.0]


def test_accuracy_handles_batch_with_fewer_than_five_classes():
    """A 2-sample batch has at most 2 distinct labels; top-5 is over logits,
    not over classes present, so it must still work."""
    logits = torch.randn(2, NUM_CLASSES)
    top1, top5 = accuracy(logits, torch.tensor([3, 3]), topk=(1, 5))
    assert 0.0 <= top1 <= top5 <= 100.0


# --- criterion ------------------------------------------------------------


def test_mixed_criterion_equals_plain_ce_when_unmixed():
    cfg = cfg_for()
    criterion = build_criterion(cfg)
    logits, y = torch.randn(8, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (8,))
    batch = apply_batch_aug(logits.new_zeros(8, 3, 32, 32), y, cfg)
    assert mixed_criterion(criterion, logits, batch) == pytest.approx(criterion(logits, y).item())


# --- optimizer & scheduler ------------------------------------------------


def test_no_weight_decay_on_bn_and_bias():
    cfg = cfg_for()
    opt = build_optimizer(build_model(cfg), cfg)
    assert len(opt.param_groups) == 2
    nodecay = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    assert all(p.ndim <= 1 for p in nodecay["params"])
    decay = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    assert all(p.ndim > 1 for p in decay["params"])


def test_single_param_group_when_no_wd_split_disabled():
    cfg = cfg_for(optim__no_wd_on_bn_bias=False)
    assert len(build_optimizer(build_model(cfg), cfg).param_groups) == 1


def test_nesterov_reaches_optimizer():
    """C5.2: `nesterov: true` in the config has to survive into every group.

    It is a keyword that SGD silently accepts and momentum silently ignores, so
    nothing downstream would ever complain if it were dropped.
    """
    cfg = cfg_for(optim__optimizer="sgd", optim__nesterov=True)
    opt = build_optimizer(build_model(cfg), cfg)
    assert all(g["nesterov"] for g in opt.param_groups)


def test_nesterov_is_dropped_without_momentum():
    """Torch rejects Nesterov at momentum 0; the config pair must not raise."""
    cfg = cfg_for(optim__optimizer="sgd", optim__nesterov=True, optim__momentum=0.0)
    opt = build_optimizer(build_model(cfg), cfg)
    assert not any(g["nesterov"] for g in opt.param_groups)


def test_resnet18_no_decay_group_holds_every_bn_and_bias():
    """C5.2: pin the no-decay group size for the model the 94% recipe trains.

    41 = 20 BatchNorm layers x (scale, shift) + the classifier bias; all 20
    convs are bias-free. Weight decay leaking onto those is the top item on
    sprint 5's debug checklist and costs ~1% with no other symptom, so the
    count is asserted rather than eyeballed.
    """
    cfg = cfg_for(model__name="resnet18_cifar")
    opt = build_optimizer(build_model(cfg), cfg)
    nodecay = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    decay = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    assert len(nodecay["params"]) == 41
    assert len(decay["params"]) == 21


def test_scheduler_steps_per_step_not_per_epoch():
    cfg = cfg_for(train__epochs=10, optim__warmup_epochs=0)
    opt = build_optimizer(build_model(cfg), cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=100)
    lr0 = opt.param_groups[0]["lr"]
    for _ in range(50):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < lr0  # decayed within epoch 1


def test_warmup_reaches_base_lr_then_decays():
    cfg = cfg_for(
        train__epochs=10,
        optim__warmup_epochs=2,
        optim__learning_rate=0.1,
        optim__warmup_start_factor=0.1,
    )
    opt = build_optimizer(build_model(cfg), cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=10)
    lrs = [step_and_get_lr(opt, sched) for _ in range(100)]
    assert lrs[0] == pytest.approx(0.01, rel=0.1)
    assert max(lrs) == pytest.approx(0.1, rel=0.01)
    assert lrs[-1] < 0.001


def test_step_milestones_are_absolute_epochs_despite_warmup():
    cfg = cfg_for(
        train__epochs=10,
        optim__warmup_epochs=2,
        optim__scheduler="step",
        optim__learning_rate=0.1,
        optim__step_milestones="4,6",
    )
    opt = build_optimizer(build_model(cfg), cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=10)
    lrs = [step_and_get_lr(opt, sched) for _ in range(100)]
    assert lrs[39] == pytest.approx(0.1)  # end of epoch 4
    assert lrs[40] == pytest.approx(0.01)  # first drop, at absolute epoch 4
    assert lrs[60] == pytest.approx(0.001)  # second, at absolute epoch 6


@pytest.mark.parametrize("name", ["cosine", "step", "constant", "onecycle"])
def test_every_scheduler_builds_and_steps(name):
    cfg = cfg_for(train__epochs=4, optim__warmup_epochs=1, optim__scheduler=name)
    opt = build_optimizer(build_model(cfg), cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=5)
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] >= 0


# --- checkpointing --------------------------------------------------------


def test_checkpoint_roundtrip_is_exact(tmp_path):
    cfg = cfg_for(tmp_path)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "ckpt.pth"

    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        ema=None,
        epoch=5,
        best_metric=0.9,
        cfg=cfg,
        rng_state=capture_rng_state(),
    )

    model2 = build_model(cfg)
    opt2 = build_optimizer(model2, cfg)
    sched2 = build_scheduler(opt2, cfg, steps_per_epoch=10)
    meta = load_checkpoint(path, model=model2, optimizer=opt2, scheduler=sched2)

    assert meta["epoch"] == 5 and meta["best_metric"] == 0.9
    assert meta["cfg_hash"] and "rng_state" in meta
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_write_is_atomic(tmp_path):
    """No .tmp file survives a successful save."""
    cfg = cfg_for(tmp_path)
    model = build_model(cfg)
    save_checkpoint(
        tmp_path / "ckpt.pth",
        model=model,
        optimizer=build_optimizer(model, cfg),
        scheduler=None,
        scaler=None,
        ema=None,
        epoch=0,
        best_metric=0.0,
        cfg=cfg,
        rng_state=capture_rng_state(),
    )
    assert list(tmp_path.glob("*.tmp")) == []


# --- trainer --------------------------------------------------------------


def test_trainer_cannot_access_test_set(tmp_path):
    """Structural: __init__ accepts a 2-tuple and nothing else."""
    cfg = cfg_for(tmp_path, train__epochs=1, optim__warmup_epochs=0)
    train_dl, val_dl = tiny_loaders()
    with pytest.raises((TypeError, ValueError)):
        Trainer(build_model(cfg), (train_dl, val_dl, val_dl), cfg, Run(cfg))


def test_one_epoch_on_tiny_subset_runs(tmp_path):
    cfg = cfg_for(tmp_path, train__epochs=1, optim__warmup_epochs=0)
    t = Trainer(build_model(cfg), tiny_loaders(n=64), cfg, Run(cfg))
    m = t.train_one_epoch(0)
    assert 0 < m["train_loss"] < 10
    assert 0 <= m["train_acc"] <= 100


def test_fit_writes_checkpoints_and_manifest(tmp_path):
    cfg = cfg_for(tmp_path, train__epochs=2, optim__warmup_epochs=1)
    run = Run(cfg)
    summary = Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run).fit()

    assert summary["epochs_completed"] == 2
    assert (run.checkpoint_dir / "last.pth").exists()
    assert (run.checkpoint_dir / "best.pth").exists()
    assert len((run.dir / "metrics.jsonl").read_text().splitlines()) == 2

    import json

    manifest = json.loads((run.dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["best_val_acc"] == summary["best_val_acc"]


def test_resume_matches_uninterrupted_run(tmp_path):
    """Train 4 epochs straight vs 2 + resume 2; final val acc must match."""

    def train(epochs: int, resume_from=None) -> tuple[float, Run]:
        seed_everything(0)
        cfg = cfg_for(tmp_path, train__epochs=epochs, optim__warmup_epochs=1)
        model = build_model(cfg)
        run = Run(cfg)
        trainer = Trainer(model, tiny_loaders(n=64), cfg, run)
        if resume_from is not None:
            trainer.resume_from(resume_from)
        return trainer.fit()["final_val_acc"], run

    straight, _ = train(4)
    _, partial = train(2)
    # The 2-epoch config hashes differently from the 4-epoch one, so the
    # mismatch warning is expected here -- it is asserted on separately below.
    with pytest.warns(UserWarning, match="config hash"):
        resumed, _ = train(4, resume_from=partial.checkpoint_dir / "last.pth")

    assert resumed == pytest.approx(straight, abs=0.5)


def test_resume_warns_on_config_hash_mismatch(tmp_path):
    cfg = cfg_for(tmp_path, train__epochs=2, optim__warmup_epochs=1)
    run = Run(cfg)
    Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run).fit()

    other = cfg_for(tmp_path, train__epochs=2, optim__warmup_epochs=1, optim__learning_rate=0.007)
    trainer = Trainer(build_model(other), tiny_loaders(n=32), other, Run(other))
    with pytest.warns(UserWarning, match="config hash"):
        trainer.resume_from(run.checkpoint_dir / "last.pth")


# --- evaluator ------------------------------------------------------------


def test_evaluate_returns_full_result(tmp_path):
    cfg = cfg_for(tmp_path)
    _, val_loader = tiny_loaders(n=40)
    r = evaluate(build_model(cfg), val_loader, cfg)
    assert 0 <= r.top1 <= 100 and r.top1 <= r.top5
    assert r.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert r.confusion_matrix.sum() == r.num_samples == 40
    assert len(r.per_class_acc) == NUM_CLASSES


def test_evaluate_restores_training_mode(tmp_path):
    cfg = cfg_for(tmp_path)
    model = build_model(cfg).train()
    _, val_loader = tiny_loaders(n=16)
    evaluate(model, val_loader, cfg)
    assert model.training


def test_evaluate_tta_not_implemented_yet(tmp_path):
    cfg = cfg_for(tmp_path)
    _, val_loader = tiny_loaders(n=16)
    with pytest.raises(NotImplementedError):
        evaluate(build_model(cfg), val_loader, cfg, tta=True)


# --- AMP ------------------------------------------------------------------


@pytest.mark.gpu
def test_amp_and_fp32_agree_on_a_step(tmp_path):
    """One optimizer step under AMP and under fp32 must land in the same place."""
    losses = []
    for amp in (True, False):
        seed_everything(0)
        cfg = cfg_for(tmp_path, train__amp=amp, device="cuda")
        model = build_model(cfg)
        trainer = Trainer(model, tiny_loaders(n=32), cfg, Run(cfg))
        losses.append(trainer.train_one_epoch(0)["train_loss"])
    assert abs(losses[0] - losses[1]) < 1e-2


def test_nan_val_metrics_when_eval_is_skipped(tmp_path):
    """eval_every > 1 must not fabricate a validation number."""
    cfg = cfg_for(tmp_path, train__epochs=3, optim__warmup_epochs=1, train__eval_every=3)
    run = Run(cfg)
    Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run).fit()
    import json

    rows = [json.loads(line) for line in (run.dir / "metrics.jsonl").read_text().splitlines()]
    assert math.isnan(rows[0]["val_acc"])
    assert not math.isnan(rows[2]["val_acc"])


def test_epoch_log_is_persisted_before_the_console_write(tmp_path):
    """Regression: a UnicodeEncodeError while printing the epoch line killed a
    GPU run and took the completed epoch's metrics with it."""
    cfg = cfg_for(tmp_path, train__epochs=1, optim__warmup_epochs=0)
    run = Run(cfg)
    trainer = Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run)
    trainer.logger._console.print = lambda *a, **k: (_ for _ in ()).throw(
        UnicodeEncodeError("charmap", "x", 0, 1, "boom")
    )

    with pytest.raises(UnicodeEncodeError):
        trainer.fit()

    assert len((run.dir / "metrics.jsonl").read_text().splitlines()) == 1
    import json

    assert json.loads((run.dir / "run_manifest.json").read_text())["status"] == "failed"


def test_console_epoch_line_is_ascii(tmp_path):
    """Windows' legacy console encodes cp1252; a non-ASCII separator raises."""
    cfg = cfg_for(tmp_path, train__epochs=1, optim__warmup_epochs=0)
    run = Run(cfg)
    trainer = Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run)
    printed = []
    trainer.logger._console.print = printed.append
    trainer.fit()
    assert printed and all(line.isascii() for line in printed)
