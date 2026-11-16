import pytest
import torch
from clingeo.training.mre_loss import case_mre


def test_case_mre_uses_euclidean_pixels_and_equal_case_weights():
    # First case: one valid 3-4-5 error at scale 2 => 10 pixels.
    # Second case: two valid unit errors at scale 1 => 1 pixel.
    predicted=torch.tensor([[[3.,4.],[100.,100.]],[[1.,0.],[0.,1.]]],requires_grad=True)
    mask=torch.tensor([[1.,0.],[1.,1.]])
    inverse=torch.eye(3).repeat(2,1,1);inverse[0,0,0]=inverse[0,1,1]=2
    loss=case_mre(predicted,torch.zeros_like(predicted),mask,inverse)
    assert float(loss.detach())==pytest.approx(5.5)
    loss.backward();assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0,1].abs().sum()==0


def test_case_mre_rejects_empty_case():
    with pytest.raises(ValueError):case_mre(torch.zeros(1,2,2),torch.zeros(1,2,2),torch.zeros(1,2),torch.eye(3)[None])
