import torch

from cifar.seeding import seed_everything


def test_seeding_is_reproducible():
    seed_everything(42)
    a = torch.randn(100)
    seed_everything(42)
    assert torch.equal(a, torch.randn(100))
