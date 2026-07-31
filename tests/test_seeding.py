import pytest
import torch

from cifar.seeding import capture_rng_state, restore_rng_state, seed_everything


def test_seeding_is_reproducible():
    seed_everything(42)
    a = torch.randn(100)
    seed_everything(42)
    assert torch.equal(a, torch.randn(100))


def test_restore_rng_state_accepts_non_cpu_byte_tensors():
    """Regression: load_checkpoint(map_location="cuda") moves the RNG state
    tensors onto the GPU, and set_rng_state rejects anything that is not a CPU
    ByteTensor. Simulated here with a wrong dtype, which fails the same way."""
    state = capture_rng_state()
    state["torch"] = state["torch"].to(torch.int64)
    restore_rng_state(state)  # must not raise


@pytest.mark.gpu
def test_rng_state_survives_a_cuda_map_location_roundtrip(tmp_path):
    """The real path: save, reload onto the GPU, restore."""
    path = tmp_path / "rng.pth"
    torch.save(capture_rng_state(), path)
    restore_rng_state(torch.load(path, map_location="cuda", weights_only=False))
