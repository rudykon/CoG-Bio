"""Align normalized-input landmark training with original-pixel evaluation."""
from __future__ import annotations
import torch


def original_pixel_scale(inverse):
    if inverse.ndim != 3 or inverse.shape[1:] != (3,3):
        raise ValueError("batched 3x3 inverse affine required")
    scale_x,scale_y = inverse[:,0,0],inverse[:,1,1]
    if not torch.isfinite(inverse).all() or not (scale_x>0).all():
        raise ValueError("finite positive pixel scale required")
    if not torch.allclose(scale_x,scale_y) or (inverse[:,0,1].abs()>1e-6).any() or (inverse[:,1,0].abs()>1e-6).any():
        raise ValueError("this objective expects isotropic letterbox coordinates")
    return scale_x


def weight_in_original_pixels(batch):
    batch["sample_weight"] = batch["sample_weight"]*original_pixel_scale(batch["affine_inv"])
    return batch
