# Sprint 4 — Tracking & reporting

**Goal:** Every run leaves behind a self-describing artifact: a run directory with a manifest, machine-readable metrics, TensorBoard scalars, and generated figures. Two runs become directly comparable.

**Depends on:** Sprint 3. **Blocks:** Sprints 5–8 (which are all ablation-driven).
**Estimated effort:** 1 session.

---

## Why before the accuracy work

Sprints 5–7 are a sequence of experiments. Running experiments without a ledger means re-running them, mis-remembering them, or reporting a number you can't reconstruct. One session spent here saves several later, and it's what turns the repo from a script into a project.

## Components

### C4.1 — Run directory

`src/cifar/tracking/run.py`

```
runs/{run_name}/
├── run_manifest.json      # written at start (status: running), updated at end
├── config.yaml            # fully resolved config, re-loadable
├── metrics.jsonl          # one JSON object per epoch
├── train.log              # console mirror
├── checkpoints/
│   ├── last.pth
│   └── best.pth
├── tb/                    # TensorBoard event files
└── figures/
    ├── curves.png
    └── confusion_matrix.png
```

`run_name` defaults to `{model.name}_{YYYYmmdd_HHMMSS}`. On collision, append `_2`.

Manifest schema is fixed in `ARCHITECTURE.md` §6. Capture:
- git SHA, branch, and dirty flag (via `subprocess`; degrade gracefully if git is absent)
- python/torch/cuda versions, GPU name, platform
- full resolved config and its hash
- start/finish timestamps, total seconds, status (`running` | `completed` | `failed` | `interrupted`)

`finalize(summary)` sets the end fields and status. Wrap `fit()` in try/except/finally so a crash writes `status: "failed"` with the traceback rather than leaving a permanently `running` manifest.

### C4.2 — Logger

`src/cifar/tracking/logger.py`

One call, three destinations:

```python
logger.log_scalars({"loss": 0.31, "acc": 91.2}, step=epoch, prefix="val")
```

writes a formatted console line (via `rich`), a TensorBoard scalar under `val/loss`, and appends to `metrics.jsonl`. Nothing in `src/` should ever write to only one of the three.

`metrics.jsonl` line format:

```json
{"epoch": 42, "train_loss": 0.31, "train_acc": 89.4, "val_loss": 0.28,
 "val_acc": 91.2, "learning_rate": 0.048, "epoch_seconds": 11.3,
 "timestamp": "2026-07-31T14:31:02Z"}
```

JSONL because it survives a crash mid-run — you can always read the epochs that completed.

Console epoch line, aimed at being scannable at a glance:

```
epoch  42/120 │ train 0.310 / 89.4% │ val 0.284 / 91.2% │ lr 4.8e-02 │ 11.3s │ best 91.2% @42
```

### C4.3 — Figures

`src/cifar/utils/viz.py`

- `plot_curves(metrics_path) -> Figure` — two panels, loss and accuracy, train vs val, with the best epoch marked.
- `plot_confusion_matrix(cm, class_names, normalize=True) -> Figure` — annotated heatmap.
- `plot_lr_schedule(cfg, steps_per_epoch) -> Figure` — plot the schedule *before* training. Cheap, and catches a broken scheduler in one second instead of one hour.
- `plot_sample_predictions(model, loader, n=16) -> Figure` — correct in green, wrong in red with the predicted label. Sprint 7 uses this for error analysis.

Log figures to TensorBoard and save to `figures/`.

### C4.4 — Comparison tooling

`scripts/make_report.py --runs runs/a runs/b runs/c --out docs/`

- A markdown table: run name, config hash, key hyperparameters that differ, best val acc, best epoch, total minutes.
- An overlaid val-accuracy curve across runs.
- Appends new rows to `docs/EXPERIMENTS.md` rather than rewriting it.

Only show hyperparameters that *differ* between the compared runs — a table with 40 identical columns is unreadable.

### C4.5 — Sweep runner

`scripts/sweep.py --dir configs/ablations/` runs every YAML in a directory sequentially, one run each, continues past a failure (logging it), and prints a summary table at the end. `--dry-run` prints what would run.

This is what makes Sprints 5–7 tolerable: define ablations as config files, start it, come back later.

### C4.6 — Wire the trainer

Replace Sprint 3's minimal logging with the real `Logger`, and have `Trainer` call `run.write_manifest()` at start and `run.finalize()` at end. Log the LR schedule figure before epoch 0.

---

## Acceptance criteria

```python
def test_run_dir_created_with_expected_structure():
    run = Run(cfg)
    assert (run.dir / "config.yaml").exists()
    assert (run.dir / "run_manifest.json").exists()
    assert run.checkpoint_dir.is_dir()

def test_manifest_captures_git_and_env():
    m = json.loads((run.dir / "run_manifest.json").read_text())
    assert "sha" in m["git"] and "torch" in m["env"]
    assert m["config_hash"] == config_hash(cfg)

def test_saved_config_reloads_to_equal_config():
    cfg2 = load_config(run.dir / "config.yaml")
    assert config_hash(cfg2) == config_hash(cfg)

def test_metrics_jsonl_is_one_valid_object_per_epoch():
    lines = (run.dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == n_epochs
    assert all("val_acc" in json.loads(l) for l in lines)

def test_logger_writes_all_three_destinations():
    logger.log_scalars({"acc": 91.2}, step=1, prefix="val")
    assert "91.2" in captured_console
    assert "val_acc" in last_jsonl_line()
    assert tb_has_scalar("val/acc")

def test_failed_run_marks_status_failed():
    with pytest.raises(RuntimeError):
        run_that_raises()
    assert manifest()["status"] == "failed"

def test_lr_schedule_figure_matches_actual_lrs():
    """Plot from config vs LRs actually observed in a short run — same shape."""
```

## Gate

Run two short trainings with different learning rates, then:

```bash
python scripts/make_report.py --runs runs/a runs/b --out docs/
tensorboard --logdir runs/
```

The report must show only the differing hyperparameter, and TensorBoard must show both runs overlaid. Kill a run mid-training with Ctrl-C and confirm the manifest says `interrupted` and `last.pth` resumes cleanly.

---

## Implementation prompt

> Read `docs/ARCHITECTURE.md` sections 5 and 6, and `docs/sprints/04-tracking.md`.
>
> Implement C4.1 through C4.6. The design rule: `log_scalars` is one call that fans out to console, TensorBoard, and `metrics.jsonl` — no code in `src/` writes to just one of them. Manifests must be written at run start with `status: "running"` and finalized in a `finally` block so crashed and interrupted runs are correctly labeled.
>
> Then go back to `Trainer` and replace the Sprint 3 placeholder logging with the real `Logger`, and add the pre-training LR-schedule plot.
>
> Finish by running two short trainings that differ only in learning rate, generating the comparison report, and describing what it shows.
