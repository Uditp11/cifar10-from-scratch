# Sprint 2 — Models & registry

**Goal:** A model registry plus two from-scratch architectures — `SimpleCNN` (fast baseline) and `ResNet-18/34-CIFAR` (the workhorse) — with parameter counts matching published references.

**Depends on:** Sprint 0. **Blocks:** Sprint 3. **Parallel with:** Sprint 1.
**Estimated effort:** 1–2 sessions.

---

## The CIFAR ResNet subtlety

ImageNet ResNet-18 opens with a 7×7 stride-2 conv followed by a 3×3 stride-2 maxpool. Applied to a 32×32 image, that throws away most of the spatial information before the first residual block — and costs several points of accuracy. The CIFAR variant replaces the stem with a single **3×3, stride 1, padding 1** conv and **removes the maxpool entirely**. Spatial resolution then goes 32 → 32 → 16 → 8 → 4 across the four stages.

This one change is worth more than most of the tricks in Sprints 6–7. Get it right, and put a comment on the stem saying why.

## Components

### C2.1 — Registry

`src/cifar/models/registry.py`

```python
_MODELS: dict[str, type[nn.Module]] = {}

def register_model(name: str): ...   # class decorator
def build_model(cfg: Config) -> nn.Module: ...
def list_models() -> list[str]: ...
```

`build_model` must: look up the name (raising with the list of valid names on a miss), construct with `cfg.model`, apply weight init, move to `cfg.device`, and apply `.to(memory_format=torch.channels_last)` when `cfg.train.channels_last`.

Go back to `config.py` and replace the Sprint 0 `# TODO(sprint-2)` with real validation: `cfg.model.name in list_models()`. Import carefully to avoid a cycle — validate lazily inside the function, not at module import.

### C2.2 — Shared layers

`src/cifar/models/layers.py`

- `conv3x3(in_ch, out_ch, stride=1)` and `conv1x1` helpers, `bias=False` always (BN follows).
- `BasicBlock` — two 3×3 convs, BN after each, ReLU after the first and after the residual add. Identity shortcut when shapes match; 1×1 conv + BN projection when stride ≠ 1 or channels change.
- Optional `Bottleneck` if you want ResNet-50 later; not required.

### C2.3 — SimpleCNN

`src/cifar/models/simple_cnn.py` — the fast baseline that makes Sprint 3's end-to-end test cheap.

Three VGG-ish blocks: `[Conv-BN-ReLU ×2 → MaxPool]` with channels 64 → 128 → 256, then global average pool → dropout → linear. Roughly 1.2M params. Should hit 85–88% in 30 epochs and train at ~4 s/epoch.

Do **not** use a giant flatten-into-4096-linear classifier head. GAP keeps the parameter count honest and trains more stably.

### C2.4 — ResNet-CIFAR

`src/cifar/models/resnet.py`

- Stem: `conv3x3(3, 64)` → BN → ReLU. No maxpool.
- Four stages of `BasicBlock`, channels `[64, 128, 256, 512] × width_multiplier`, strides `[1, 2, 2, 2]`, blocks `[2,2,2,2]` for ResNet-18 and `[3,4,6,3]` for ResNet-34.
- Head: global average pool → optional dropout → `Linear(512 × width, num_classes)`.
- Register both as `resnet18_cifar` and `resnet34_cifar`.

**Initialization:**
- Conv: Kaiming normal, `mode="fan_out"`, `nonlinearity="relu"`.
- BN: weight 1, bias 0.
- When `cfg.model.zero_init_residual`, zero the **last BN gamma in each BasicBlock**. This makes every residual branch start as an identity function, which measurably stabilizes early training and is worth ~0.2–0.3% on the final number. Cite the ResNet "bag of tricks" paper in the comment.

### C2.5 — Model introspection

Add to the base class or as a mixin: `num_parameters` property, and a `summary()` returning a string with per-stage output shapes and parameter counts. `scripts/model_summary.py --model resnet18_cifar` prints it. Useful for the README and for catching a wrong stem instantly.

---

## Acceptance criteria

`tests/test_models.py` — all CPU, all fast.

```python
@pytest.mark.parametrize("name", list_models())
def test_forward_shape(name):
    m = build_model(cfg_for(name, device="cpu"))
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)

@pytest.mark.parametrize("name", list_models())
def test_batch_size_one_works(name):
    m = build_model(cfg_for(name, device="cpu")).eval()
    assert m(torch.randn(1, 3, 32, 32)).shape == (1, 10)

def test_param_counts_match_reference():
    assert build_model(cfg_for("resnet18_cifar")).num_parameters == 11_173_962
    assert build_model(cfg_for("resnet34_cifar")).num_parameters == 21_282_122
    assert 1_000_000 < build_model(cfg_for("simple_cnn")).num_parameters < 1_500_000

def test_resnet_has_no_maxpool_in_stem():
    m = build_model(cfg_for("resnet18_cifar"))
    assert not any(isinstance(mod, nn.MaxPool2d) for mod in m.modules())

def test_stem_preserves_resolution():
    m = build_model(cfg_for("resnet18_cifar"))
    assert m.stem(torch.randn(1, 3, 32, 32)).shape[-2:] == (32, 32)

def test_zero_init_residual_zeros_last_bn():
    m = build_model(cfg_for("resnet18_cifar", zero_init_residual=True))
    for blk in all_basic_blocks(m):
        assert torch.all(blk.bn2.weight == 0)

def test_gradients_reach_every_parameter():
    m = build_model(cfg_for("resnet18_cifar", device="cpu"))
    m(torch.randn(2, 3, 32, 32)).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead}"

def test_unknown_model_name_raises_with_suggestions():
    with pytest.raises(ValueError, match="resnet18_cifar"):
        build_model(cfg_for("resnet18"))   # ImageNet-style name, not registered

def test_model_can_overfit_ten_samples():
    """The single most valuable model test: 200 steps on 10 fixed images
    must drive loss below 0.01. If it can't, the architecture or the
    init is broken and no amount of tuning will save it."""
    m = build_model(cfg_for("resnet18_cifar", device="cpu"))
    x, y = torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(); loss = F.cross_entropy(m(x), y); loss.backward(); opt.step()
    assert loss.item() < 0.01
```

## Gate

- All tests green, including the overfit test.
- `python scripts/model_summary.py --model resnet18_cifar` shows stage resolutions 32 → 32 → 16 → 8 → 4.

---

## Implementation prompt

> Read `docs/ARCHITECTURE.md` sections 1 and 4, and `docs/sprints/02-models.md`.
>
> Implement C2.1 through C2.5. The critical detail: this is the **CIFAR** ResNet variant — a 3×3 stride-1 stem and **no maxpool**, so the first stage operates at full 32×32 resolution. Do not copy the torchvision ImageNet architecture. Parameter count for ResNet-18-CIFAR must come out to exactly 11,173,962; if yours differs, the architecture is wrong somewhere — find it rather than adjusting the test.
>
> Also return to `src/cifar/config.py` and replace the Sprint 0 `TODO(sprint-2)` with real model-name validation against the registry, avoiding a circular import.
>
> Write all the acceptance tests, including the 10-sample overfit test, and report the parameter counts and the per-stage resolutions.
