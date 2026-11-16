import torch
from clingeo.training.spatial_loss import gaussian_cross_entropy


def test_spatial_loss_matches_decoder_and_masks_unsupervised_landmarks():
    logits=torch.zeros(1,2,3,3,requires_grad=True)
    target=torch.zeros_like(logits);target[0,0,1,1]=3
    mask=torch.tensor([[1.,0.]])
    loss=gaussian_cross_entropy(logits,target,mask)
    assert torch.allclose(loss,gaussian_cross_entropy(logits+17,target*5,mask),atol=1e-6)
    loss.backward()
    assert logits.grad[0,0,1,1]<0
    assert logits.grad[0,0,0,0]>0
    assert logits.grad[0,1].count_nonzero()==0
    assert gaussian_cross_entropy(logits,target,torch.zeros_like(mask))==0
