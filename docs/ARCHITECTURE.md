# ARCHITECTURE.md — Binding interface contracts

Everything in this file is frozen. Changing a signature here requires an ADR entry in `docs/ADR.md` and an update to every call site. Sprints depend on these contracts to be implementable independently.

---

## 1. Config

The config object is the single source of truth. It is constructed once, at process start, and passed down. Nothing mutates it after construction (`frozen=True`).

```python
# src/cifar/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("./data")
    val_size: int = 5000
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

@dataclass(frozen=True)
class AugConfig:
    random_crop_padding: int = 4       # 0 disables
    horizontal_flip_p: float = 0.5
    cutout_holes: int = 0              # 0 disables
    cutout_length: int = 16
    randaugment_n: int = 0             # 0 disables
    randaugment_m: int = 9
    mixup_alpha: float = 0.0           # 0 disables
    cutmix_alpha: float = 0.0          # 0 disables
    batch_aug_prob: float = 0.0        # P(apply mixup/cutmix); 0 with both alphas 0 = off

@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18_cifar"
    num_classes: int = 10
    width_multiplier: float = 1.0
    dropout: float = 0.0
    zero_init_residual: bool = True

@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "sgd"             # sgd | adamw
    learning_rate: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    no_wd_on_bn_bias: bool = True
    scheduler: str = "cosine"          # cosine | step | onecycle | constant
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_learning_rate: float = 0.0
    step_milestones: tuple[int, ...] = (60, 90)
    step_gamma: float = 0.1
    label_smoothing: float = 0.0
    grad_clip_norm: float = 0.0        # 0 disables

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    amp: bool = True
    channels_last: bool = False        # 3x slower at 32x32 — measured, ADR-007
    ema_decay: float = 0.0             # 0 disables
    ema_warmup_epochs: int = 0
    eval_every: int = 1
    save_every: int = 1
    keep_last_n_checkpoints: int = 2

@dataclass(frozen=True)
class ReproConfig:
    seed: int = 42
    deterministic: bool = False
    cudnn_benchmark: bool = True

@dataclass(frozen=True)
class LogConfig:
    run_root: Path = Path("./runs")
    run_name: str | None = None        # None -> auto: {model}_{timestamp}
    tensorboard: bool = True
    step_interval: int = 0             # 0 -> epoch-level logging only
    log_confusion_matrix: bool = True

@dataclass(frozen=True)
class Config:
    data: DataConfig
    aug: AugConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    repro: ReproConfig
    log: LogConfig
    device: str = "auto"   # auto | cuda | cpu — "auto" resolves at load time
    notes: str = ""
```

### Loader

```python
def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML, merge onto base.yaml, apply dotted CLI overrides, validate.

    Overrides look like ["optim.learning_rate=0.05", "train.epochs=200"].
    Raises ValueError on unknown keys, type mismatches, or failed invariants.
    """

def config_to_dict(cfg: Config) -> dict: ...
def config_hash(cfg: Config) -> str:
    """Stable 12-char hash of the config, excluding `notes` and `log`.
    Two runs with the same hash should produce the same result."""
```

**Invariants validated at load:**
- `0 <= warmup_epochs < epochs`
- `0 < val_size < 50000`
- **Batch-aug coherence**, both directions — a config that silently does nothing is the bug being prevented:
  - `batch_aug_prob > 0` requires `mixup_alpha > 0 or cutmix_alpha > 0` (a probability with nothing to apply)
  - `batch_aug_prob == 0` requires `mixup_alpha == 0 and cutmix_alpha == 0` (an alpha that can never fire)
  - both alphas may be > 0 simultaneously; `apply_batch_aug` chooses between them per batch
- `model.name` is registered
- `optimizer` and `scheduler` are known strings
- `device` is one of `auto | cuda | cpu`. `"auto"` resolves to `"cuda"` when `torch.cuda.is_available()` else `"cpu"`, and the resolved value is what lands in the frozen `Config` (so `run_manifest.json` records what actually ran). Explicit `device: cuda` without CUDA raises — an explicit request must never be silently downgraded.

---

## 2. Seeding

```python
# src/cifar/seeding.py
def seed_everything(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    """Seed random, numpy, torch, torch.cuda; set cudnn flags."""

def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding. Derives from the base seed + worker_id."""

def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling."""
```

---

## 3. Data

```python
# src/cifar/data/dataset.py
def build_datasets(cfg: Config) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test). Train gets train transforms; val and test get eval
    transforms. The val split comes from the fixed index file at
    `configs/val_indices.json` — never a random split."""

# src/cifar/data/transforms.py
def build_train_transform(cfg: Config) -> Callable: ...
def build_eval_transform(cfg: Config) -> Callable: ...

class Cutout:
    def __init__(self, n_holes: int, length: int) -> None: ...
    def __call__(self, img: Tensor) -> Tensor:
        """Applied AFTER normalization; zeroes square regions (== mean pixel)."""

# src/cifar/data/batch_aug.py
@dataclass
class MixedBatch:
    inputs: Tensor
    targets_a: Tensor
    targets_b: Tensor
    lam: float

def apply_batch_aug(x: Tensor, y: Tensor, cfg: Config) -> MixedBatch:
    """Returns a MixedBatch. When no aug fires, lam == 1.0 and
    targets_a is targets_b, so downstream code needs no branch."""

# src/cifar/data/loaders.py
def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]: ...
```

---

## 4. Models

```python
# src/cifar/models/registry.py
def register_model(name: str) -> Callable[[type], type]:
    """Class decorator. Registers a model constructor under `name`."""

def build_model(cfg: Config) -> nn.Module:
    """Constructs the model named by cfg.model.name, moves it to cfg.device,
    and applies channels_last if configured. Weight init is NOT done here —
    see below."""

def list_models() -> list[str]: ...
```

**Every registered model must:**
- accept `(cfg: ModelConfig)` as its only constructor argument
- map `(B, 3, 32, 32) -> (B, num_classes)`
- expose `.num_parameters` as a property (via `ModelBase`)
- work at `B=1` (no batch-size assumptions)
- **initialize its own weights in `__init__`**, including `zero_init_residual`. Init needs architecture-specific knowledge — which BN sits last in a residual branch — and hoisting that into `build_model` would couple the registry to one architecture. The registry places and configures; the model constructs itself.

Registration happens as an import side effect of `cifar/models/__init__.py`. Always import `from cifar.models import ...`, never `from cifar.models.registry import ...` — the latter skips the registrations and yields an empty registry.

Reference parameter counts (used in tests): `SimpleCNN` ≈ 1.2M, `ResNet-18-CIFAR` = 11,173,962, `ResNet-34-CIFAR` = 21,282,122.

---

## 5. Engine

```python
# src/cifar/engine/optim.py
def build_optimizer(model: nn.Module, cfg: Config) -> Optimizer:
    """Builds param groups. When cfg.optim.no_wd_on_bn_bias, all 1-D
    parameters (BN weights, all biases) go into a weight_decay=0 group."""

def build_scheduler(opt: Optimizer, cfg: Config, steps_per_epoch: int) -> LRScheduler:
    """Per-STEP scheduler. Trainer calls .step() after every optimizer step,
    not once per epoch. Warmup is folded in via SequentialLR."""

# src/cifar/engine/criterion.py
def build_criterion(cfg: Config) -> nn.Module: ...

def mixed_criterion(criterion, logits: Tensor, batch: MixedBatch) -> Tensor:
    """lam * loss(logits, a) + (1 - lam) * loss(logits, b)."""

# src/cifar/engine/ema.py
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None: ...
    def update(self, model: nn.Module) -> None: ...
    @property
    def module(self) -> nn.Module: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...

# src/cifar/engine/checkpoint.py
def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler,
                    ema, epoch: int, best_metric: float, cfg: Config,
                    rng_state: dict) -> None: ...

def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, ema=None, map_location="cpu") -> dict:
    """Returns {"epoch", "best_metric", "cfg_hash", "rng_state"}.
    Restoring rng_state is what makes resume bit-faithful."""

# src/cifar/engine/trainer.py
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
    def __init__(self, model: nn.Module,
                 loaders: tuple[DataLoader, DataLoader],   # (train, val) ONLY
                 cfg: Config, run: Run) -> None: ...
    def fit(self) -> dict[str, float]: ...
    def train_one_epoch(self, epoch: int) -> dict[str, float]: ...
    def validate(self, epoch: int) -> dict[str, float]: ...
    def resume_from(self, path: Path) -> None: ...
```

`Trainer.__init__` takes a 2-tuple. This is deliberate: it is structurally impossible for the trainer to see the test set.

```python
# src/cifar/engine/evaluator.py
@dataclass
class EvalResult:
    loss: float
    top1: float
    top5: float
    per_class_acc: dict[str, float]
    confusion_matrix: np.ndarray       # (10, 10), rows = true
    num_samples: int

def evaluate(model: nn.Module, loader: DataLoader, cfg: Config,
             *, tta: bool = False) -> EvalResult:
    """tta=True averages softmax over {identity, horizontal flip}."""
```

---

## 6. Tracking

```python
# src/cifar/tracking/run.py
class Run:
    """Owns one run directory: runs/{run_name}/"""
    def __init__(self, cfg: Config) -> None: ...
    @property
    def dir(self) -> Path: ...
    @property
    def checkpoint_dir(self) -> Path: ...
    @property
    def figures_dir(self) -> Path: ...
    @property
    def tb_dir(self) -> Path: ...
    def write_manifest(self, extra: dict | None = None) -> None: ...
    def finalize(self, summary: dict[str, float], status: str = "completed") -> None: ...
```

`run_manifest.json` schema:

```json
{
  "run_name": "resnet18_cifar_20260731_142300",
  "config_hash": "a3f9c21b8e04",
  "config": { "...full resolved config..." },
  "git": { "sha": "abc1234", "branch": "main", "dirty": false },
  "env": { "python": "3.11.6", "torch": "2.3.0", "cuda": "12.1",
           "gpu": "NVIDIA GeForce RTX 4070", "platform": "Windows-11" },
  "runtime": { "device": "cuda", "channels_last": false, "seconds_per_epoch": 11.3 },
  "started_at": "2026-07-31T14:23:00Z",
  "finished_at": "2026-07-31T14:58:12Z",
  "total_seconds": 2112.4,
  "status": "completed",
  "summary": { "best_val_acc": 95.12, "best_epoch": 187, "final_val_acc": 95.08 }
}
```

`runtime` holds the three facts that decide whether two runs are comparable and are not
readable off the config alone: the *resolved* device (ADR-005 — `"auto"` never reaches
the manifest), the memory format (ADR-007 makes it an experimental variable, not a
detail), and the measured steady-state `seconds_per_epoch` — the median of the
per-epoch times in `metrics.jsonl`, excluding the first epoch, which on Windows carries
a fixed 60–75 s of DataLoader worker spawn. It is `null` until at least one epoch has
been logged.

```python
# src/cifar/tracking/logger.py
class Logger:
    def __init__(self, run: Run, cfg: Config) -> None: ...
    def log_scalars(self, metrics: dict[str, float], step: int,
                    prefix: str = "") -> None:
        """Writes to console, train.log, TensorBoard, and metrics.jsonl in one call."""
    def log_epoch(self, m: EpochMetrics) -> None: ...
    def log_figure(self, name: str, fig, step: int = 0) -> None: ...
    def info(self, message: str) -> None: ...
    def close(self) -> None: ...

# src/cifar/tracking/meters.py
class AverageMeter:
    def update(self, val: float, n: int = 1) -> None: ...
    @property
    def avg(self) -> float: ...

def accuracy(logits: Tensor, targets: Tensor,
             topk: tuple[int, ...] = (1,)) -> list[float]:
    """Top-k accuracy in percent."""
```

---

## 7. Scripts

```bash
python scripts/train.py --config configs/resnet18.yaml [--override optim.learning_rate=0.05 ...] [--resume runs/foo/ckpt/last.pth]
python scripts/eval.py  --run runs/foo [--checkpoint best] [--tta] [--ema]
python scripts/sweep.py --dir configs/ablations/ [--dry-run]
python scripts/make_report.py --runs runs/a runs/b --out docs/
```

All scripts exit non-zero on failure and print the run directory as their last line of stdout so they can be chained.

---

## 8. Data-flow invariants

These hold at every point in the project and are worth asserting in tests:

1. Normalization happens exactly once, in the transform pipeline. No layer inside a model normalizes input.
2. Cutout runs *after* normalization; Mixup/CutMix run on the *batch*, on the GPU, after collation.
3. The scheduler steps per optimizer step, never per epoch.
4. The AMP `GradScaler` wraps the backward and the step; `grad_clip_norm` unscales first.
5. EMA updates after the optimizer step, never before.
6. Val metrics select checkpoints. Test metrics are computed once, by `eval.py`, and never feed a decision.
