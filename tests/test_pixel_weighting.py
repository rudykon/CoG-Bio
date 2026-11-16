import pytest
import torch
from clingeo.training.pixel_weighting import original_pixel_scale,weight_in_original_pixels


def test_original_pixel_weights_respect_scale_and_image_point_count():
    inverse=torch.eye(3).repeat(2,1,1);inverse[0,:2,:2]*=2.;inverse[0,0,2]=17
    batch={"affine_inv":inverse,"sample_weight":torch.tensor([1/4,1/2])}
    assert original_pixel_scale(inverse).tolist()==[2.,1.]
    assert weight_in_original_pixels(batch)["sample_weight"].tolist()==[.5,.5]
    inverse[0,1,1]=3.
    with pytest.raises(ValueError,match="isotropic"):original_pixel_scale(inverse)
