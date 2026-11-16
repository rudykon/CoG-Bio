import pytest
import torch
from torch import nn

from clingeo.model import DinoFeaturePyramid, LoRALinear


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.projection = nn.Conv2d(3, 8, 16, 16)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward_intermediates(self, image, **kwargs):
        feature = self.projection(image)
        return [feature] * 4


def test_frozen_encoder_keeps_eval_and_has_no_grad_while_pyramid_learns():
    encoder = FakeEncoder()
    model = DinoFeaturePyramid(encoder).train()
    features = model(torch.randn(2, 3, 64, 64))
    assert not encoder.training
    assert [tuple(x.shape) for x in features] == [
        (2, 96, 16, 16), (2, 192, 8, 8), (2, 384, 4, 4), (2, 768, 2, 2)
    ]
    sum(x.square().mean() for x in features).backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert model.projections[0][0].weight.grad.abs().sum() > 0


def test_lora_preserves_pretrained_function_at_initialization_and_only_updates_residual():
    base = nn.Linear(8, 24)
    original = base.weight.detach().clone()
    model = LoRALinear(base, rank=2, alpha=4)
    values = torch.randn(3, 5, 8)
    assert torch.equal(model(values), base(values))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(values).square().sum().backward()
    optimizer.step()
    assert torch.equal(base.weight, original)
    assert base.weight.grad is None
    assert not torch.equal(model(values), base(values))


def test_invalid_geometry_is_rejected_before_encoder_forward():
    with pytest.raises(ValueError, match="divisible by 32"):
        DinoFeaturePyramid(FakeEncoder())(torch.randn(1, 3, 70, 70))
