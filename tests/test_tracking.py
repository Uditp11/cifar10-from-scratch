import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cifar.config import config_hash, load_config
from cifar.engine.trainer import EpochMetrics, Trainer
from cifar.models import build_model
from cifar.tracking.logger import Logger
from cifar.tracking.run import Run
from cifar.utils.viz import (
    plot_confusion_matrix,
    plot_curves,
    plot_lr_schedule,
    plot_sample_predictions,
)
from tests.test_engine import cfg_for, tiny_loaders

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_report  # noqa: E402


def tb_scalars(run: Run) -> set[str]:
    """Tags actually present in the run's TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run.tb_dir))
    acc.Reload()
    return set(acc.Tags()["scalars"])


# --- run directory --------------------------------------------------------


def test_run_dir_created_with_expected_structure(tmp_path):
    run = Run(cfg_for(tmp_path))
    assert (run.dir / "config.yaml").exists()
    assert run.checkpoint_dir.is_dir()
    assert run.figures_dir.is_dir()
    assert run.tb_dir.is_dir()

    run.write_manifest()
    assert (run.dir / "run_manifest.json").exists()


def test_manifest_captures_git_and_env(tmp_path):
    cfg = cfg_for(tmp_path)
    run = Run(cfg)
    run.write_manifest()
    m = json.loads((run.dir / "run_manifest.json").read_text())

    assert "sha" in m["git"] and "dirty" in m["git"]
    assert "torch" in m["env"] and "python" in m["env"] and "platform" in m["env"]
    assert m["config_hash"] == config_hash(cfg)
    assert m["status"] == "running"


def test_manifest_records_resolved_device_and_memory_format(tmp_path):
    """ADR-005 and ADR-007: neither is a detail, both are experimental variables."""
    cfg = cfg_for(tmp_path, train__channels_last=False)
    run = Run(cfg)
    run.write_manifest()
    runtime = json.loads((run.dir / "run_manifest.json").read_text())["runtime"]

    assert runtime["device"] == "cpu"  # resolved, never "auto"
    assert runtime["channels_last"] is False
    assert runtime["seconds_per_epoch"] is None  # nothing measured yet
    # Present whether or not nvidia-smi exists -- None is a valid reading, a
    # missing key would make "was this run hot?" unanswerable after the fact.
    assert "gpu_temp_start_c" in runtime and "gpu_temp_end_c" in runtime


def test_manifest_seconds_per_epoch_measures_the_steady_state(tmp_path):
    """Regression: excluding only epoch 1 reported SimpleCNN at 6.5 s/epoch
    against a true 4.4 -- the run ramps for several epochs, not one."""
    run = Run(cfg_for(tmp_path))
    ramp = [90.0, 8.4, 7.0, 5.2, 4.7, 4.3, 4.4, 4.5, 4.4, 4.4, 4.3, 4.4]
    rows = [{"epoch": i, "epoch_seconds": s} for i, s in enumerate(ramp)]
    (run.dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    run.finalize({"best_val_acc": 1.0})
    m = json.loads((run.dir / "run_manifest.json").read_text())
    assert m["runtime"]["seconds_per_epoch"] == pytest.approx(4.4)
    assert m["status"] == "completed"


def test_saved_config_reloads_to_equal_config(tmp_path):
    cfg = cfg_for(tmp_path, optim__learning_rate=0.037)
    run = Run(cfg)
    assert config_hash(load_config(run.dir / "config.yaml")) == config_hash(cfg)


def test_run_name_collision_gets_a_suffix(tmp_path):
    cfg = cfg_for(tmp_path, log__run_name="fixed")
    assert Run(cfg).name == "fixed"
    assert Run(cfg).name == "fixed_2"


# --- logger ---------------------------------------------------------------


def test_logger_writes_all_destinations(tmp_path):
    cfg = cfg_for(tmp_path)
    run = Run(cfg)
    logger = Logger(run, cfg)
    printed = []
    logger._console.print = printed.append

    logger.log_scalars({"acc": 91.2}, step=1, prefix="val")
    logger.close()

    assert any("91.2" in line for line in printed)
    assert "91.2" in (run.dir / "train.log").read_text()
    assert json.loads((run.dir / "metrics.jsonl").read_text().splitlines()[-1])["val_acc"] == 91.2
    assert "val/acc" in tb_scalars(run)


def test_logger_epoch_line_reports_the_best_so_far(tmp_path):
    cfg = cfg_for(tmp_path, train__epochs=3, optim__warmup_epochs=1)
    logger = Logger(Run(cfg), cfg)
    printed = []
    logger._console.print = printed.append

    for epoch, acc in enumerate([80.0, 91.5, 88.0]):
        logger.log_epoch(EpochMetrics(epoch, 1.0, 50.0, 1.0, acc, 0.1, 1.0))
    logger.close()

    assert "best 91.5% @2" in printed[-1]
    assert all(line.isascii() for line in printed)


def test_logger_skips_nan_in_tensorboard(tmp_path):
    """A skipped eval must leave a gap, not a point at zero."""
    cfg = cfg_for(tmp_path)
    run = Run(cfg)
    logger = Logger(run, cfg)
    logger._console.print = lambda *a, **k: None
    logger.log_epoch(EpochMetrics(0, 1.0, 50.0, float("nan"), float("nan"), 0.1, 1.0))
    logger.close()

    tags = tb_scalars(run)
    assert "train/loss" in tags and "val/loss" not in tags


def test_logger_can_run_without_tensorboard(tmp_path):
    cfg = cfg_for(tmp_path, log__tensorboard=False)
    run = Run(cfg)
    logger = Logger(run, cfg)
    logger._console.print = lambda *a, **k: None
    logger.log_scalars({"acc": 1.0}, step=0, prefix="val")
    logger.close()
    assert (run.dir / "metrics.jsonl").exists()


# --- figures --------------------------------------------------------------


def test_lr_schedule_figure_matches_actual_lrs(tmp_path):
    """The plotted schedule must describe what the trainer really does."""
    cfg = cfg_for(tmp_path, train__epochs=3, optim__warmup_epochs=1)
    loaders = tiny_loaders(n=32, batch_size=16)
    steps_per_epoch = len(loaders[0])

    plotted = plot_lr_schedule(cfg, steps_per_epoch).axes[0].lines[0].get_ydata()
    assert len(plotted) == cfg.train.epochs * steps_per_epoch

    run = Run(cfg)
    Trainer(build_model(cfg), loaders, cfg, run).fit()
    logged = [
        json.loads(line)["learning_rate"]
        for line in (run.dir / "metrics.jsonl").read_text().splitlines()
    ]
    # Epoch e is logged after (e + 1) * steps_per_epoch steps.
    assert logged[0] == pytest.approx(plotted[steps_per_epoch], rel=1e-6)
    assert logged[1] == pytest.approx(plotted[2 * steps_per_epoch], rel=1e-6)


def test_plot_curves_marks_the_best_epoch(tmp_path):
    rows = [
        {
            "epoch": e,
            "train_loss": 1.0 - e / 10,
            "train_acc": 50.0 + e,
            "val_loss": 1.0,
            "val_acc": 60.0 + e,
            "learning_rate": 0.1,
            "epoch_seconds": 1.0,
        }
        for e in range(4)
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    fig = plot_curves(path)
    assert len(fig.axes) == 2
    assert any("best 63.00% @4" in t.get_text() for t in fig.axes[1].texts)


def test_plot_curves_rejects_an_empty_metrics_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no epochs"):
        plot_curves(path)


def test_plot_confusion_matrix_normalizes_rows(tmp_path):
    cm = np.array([[8, 2], [0, 10]])
    fig = plot_confusion_matrix(cm, ["cat", "dog"], normalize=True)
    assert {t.get_text() for t in fig.axes[0].texts} == {"80.0", "20.0", "0.0", "100.0"}


def test_plot_sample_predictions_labels_every_tile(tmp_path):
    cfg = cfg_for(tmp_path)
    _, loader = tiny_loaders(n=16, batch_size=16)
    fig = plot_sample_predictions(build_model(cfg), loader, n=4)
    assert sum(1 for ax in fig.axes if ax.get_title()) == 4


def test_interrupt_marks_the_manifest_and_resumes_from_a_clean_epoch(tmp_path):
    """Regression (ADR-009): Ctrl-C used to overwrite last.pth with the *partial*
    epoch's weights under the interrupted epoch's number, so the resumed run
    skipped that epoch entirely and ran the schedule short."""
    cfg = cfg_for(tmp_path, train__epochs=5, optim__warmup_epochs=1)
    run = Run(cfg)
    trainer = Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run)

    real_epoch = trainer.train_one_epoch

    def interrupt_during_epoch_2(epoch: int):
        if epoch == 2:
            raise KeyboardInterrupt
        return real_epoch(epoch)

    trainer.train_one_epoch = interrupt_during_epoch_2
    summary = trainer.fit()

    manifest = json.loads((run.dir / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert summary["epochs_completed"] == 2  # epochs 0 and 1, not the aborted 2

    resumed = Trainer(build_model(cfg), tiny_loaders(n=32), cfg, Run(cfg))
    resumed.resume_from(run.checkpoint_dir / "last.pth")
    assert resumed.start_epoch == 2  # redo the interrupted epoch, never skip it
    assert resumed.fit()["epochs_completed"] == 5


def test_fit_writes_the_expected_figures(tmp_path):
    cfg = cfg_for(tmp_path, train__epochs=1, optim__warmup_epochs=0)
    run = Run(cfg)
    Trainer(build_model(cfg), tiny_loaders(n=32), cfg, run).fit()
    for name in ("lr_schedule", "curves", "confusion_matrix"):
        assert (run.figures_dir / f"{name}.png").exists()


# --- comparison report ----------------------------------------------------


def _fake_run(root: Path, name: str, **params) -> Path:
    """A minimal finished run directory: manifest plus two epochs."""
    d = root / name
    d.mkdir(parents=True)
    manifest = {
        "run_name": name,
        "config_hash": name,
        "config": {
            "optim": {"learning_rate": params.get("lr", 0.1), "weight_decay": 5e-4},
            "train": {"channels_last": params.get("channels_last", False), "epochs": 2},
            "log": {"run_name": name},
        },
        "runtime": {
            "device": "cuda",
            "channels_last": params.get("channels_last", False),
            "seconds_per_epoch": params.get("spe", 4.5),
            "gpu_temp_start_c": params.get("temps", (None, None))[0],
            "gpu_temp_end_c": params.get("temps", (None, None))[1],
        },
        "status": "completed",
        "total_seconds": 120.0,
        "summary": {"best_val_acc": params.get("acc", 90.0), "best_epoch": 1},
    }
    (d / "run_manifest.json").write_text(json.dumps(manifest))
    (d / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"epoch": e, "val_acc": params.get("acc", 90.0) - (1 - e)}) for e in range(2)
        )
    )
    return d


def test_report_shows_only_the_differing_hyperparameter(tmp_path):
    runs = [
        make_report.load_run(_fake_run(tmp_path, "a", channels_last=False, spe=4.5)),
        make_report.load_run(_fake_run(tmp_path, "b", channels_last=True, spe=13.1)),
    ]
    assert make_report.differing_params(runs) == ["train.channels_last"]

    table = make_report.build_table(runs)
    assert "channels_last" in table
    assert "weight_decay" not in table  # identical across runs
    assert "4.5" in table and "13.1" in table


def test_report_writes_an_overlaid_curve(tmp_path):
    runs = [make_report.load_run(_fake_run(tmp_path, n)) for n in ("a", "b")]
    out = tmp_path / "docs" / "figures" / "comparison.png"
    make_report.plot_comparison(runs, out)
    assert out.exists()


def test_load_run_refuses_a_directory_without_a_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        make_report.load_run(tmp_path / "empty")


def test_retracted_run_stays_in_the_table_struck_through_with_its_reason(tmp_path):
    """A vanished run is worse than a retracted one in a ledger (ADR-010)."""
    good = make_report.load_run(_fake_run(tmp_path, "good", acc=91.0))
    bad = _fake_run(tmp_path, "bad", acc=99.9)
    manifest = json.loads((bad / "run_manifest.json").read_text())
    manifest["retracted"] = {"reason": "val loader saw train data", "date": "2026-08-01"}
    (bad / "run_manifest.json").write_text(json.dumps(manifest))

    table = make_report.build_table([good, make_report.load_run(bad)])
    assert "~~`" in table and "99.90~~" in table
    assert "val loader saw train data" in table
    assert "~~91.00~~" not in table  # the un-retracted row is untouched


def test_no_retracted_column_when_nothing_is_retracted(tmp_path):
    runs = [make_report.load_run(_fake_run(tmp_path, n)) for n in ("a", "b")]
    assert "Retracted" not in make_report.build_table(runs)


def test_table_reports_gpu_temperature_at_both_ends(tmp_path):
    runs = [make_report.load_run(_fake_run(tmp_path, "a", temps=(48, 71)))]
    assert "48/71" in make_report.build_table(runs)


# --- generated regions ----------------------------------------------------


def test_regenerating_a_section_leaves_hand_written_prose_untouched(tmp_path):
    """The whole point of the sentinels (ADR-010)."""
    doc = tmp_path / "EXPERIMENTS.md"
    doc.write_text(
        "# Ledger\n\nHand-written finding worth more than the table.\n", encoding="utf-8"
    )

    make_report.write_section(doc, "sprint-5", "first table")
    doc.write_text(
        doc.read_text(encoding="utf-8") + "\nA finding written after the first report.\n",
        encoding="utf-8",
    )
    make_report.write_section(doc, "sprint-5", "second table")

    text = doc.read_text(encoding="utf-8")
    assert "Hand-written finding worth more than the table." in text
    assert "A finding written after the first report." in text
    assert "second table" in text
    assert "first table" not in text
    assert text.count(make_report.begin_marker("sprint-5")) == 1


def test_sections_are_independent(tmp_path):
    doc = tmp_path / "EXPERIMENTS.md"
    make_report.write_section(doc, "sprint-5", "five")
    make_report.write_section(doc, "sprint-6", "six")
    make_report.write_section(doc, "sprint-5", "five again")

    text = doc.read_text(encoding="utf-8")
    assert "five again" in text and "six" in text and "five\n" not in text


def test_malformed_region_refuses_to_regenerate(tmp_path):
    """An unmatched marker means the generator cannot tell where the region
    stops, and guessing eats the paragraph underneath it."""
    doc = tmp_path / "EXPERIMENTS.md"
    doc.write_text(
        make_report.begin_marker("sprint-5") + "\ntable\n\nprose someone added\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed generated region"):
        make_report.write_section(doc, "sprint-5", "new table")


# --- reaping stranded "running" manifests (ADR-006 addendum) ---------------


def test_manifest_records_pid_and_host(tmp_path):
    """The reaper needs a PID to ask about, written before epoch 0."""
    import os
    import platform

    cfg = cfg_for(tmp_path)
    run = Run(cfg)
    run.write_manifest()
    manifest = json.loads((run.dir / "run_manifest.json").read_text())
    assert manifest["process"] == {"pid": os.getpid(), "host": platform.node()}


def test_running_run_with_dead_pid_reaps_to_killed():
    """A console teardown runs no `finally`, so the writer cannot fix this."""
    import platform

    # PID 0x7FFFFFFF is not allocatable on Windows or Linux, so it is reliably
    # dead without having to kill a real process from a test.
    manifest = {"status": "running", "process": {"pid": 0x7FFFFFFF, "host": platform.node()}}
    assert make_report.reaped_status(manifest) == "killed"


def test_running_run_with_live_pid_stays_running():
    import os
    import platform

    manifest = {"status": "running", "process": {"pid": os.getpid(), "host": platform.node()}}
    assert make_report.reaped_status(manifest) == "running"


def test_reaper_leaves_other_hosts_and_old_manifests_alone():
    """Conservative by design: calling a live run dead is the worse error."""
    assert make_report.reaped_status({"status": "running"}) == "running"
    assert (
        make_report.reaped_status(
            {"status": "running", "process": {"pid": 0x7FFFFFFF, "host": "some-other-box"}}
        )
        == "running"
    )


def test_reaper_does_not_touch_finished_statuses():
    for status in ("completed", "failed", "interrupted"):
        assert make_report.reaped_status({"status": status}) == status


def test_killed_run_renders_as_killed_in_the_table(tmp_path):
    import platform

    d = _fake_run(tmp_path, "stranded")
    manifest = json.loads((d / "run_manifest.json").read_text())
    manifest["status"] = "running"
    manifest["process"] = {"pid": 0x7FFFFFFF, "host": platform.node()}
    (d / "run_manifest.json").write_text(json.dumps(manifest))
    assert "killed" in make_report.build_table([make_report.load_run(d)])
