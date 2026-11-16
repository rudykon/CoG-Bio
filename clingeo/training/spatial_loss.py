"""Match the heatmap loss to the spatial-softmax coordinate decoder."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from clingeo.losses.landmark_loss import LandmarkCriterion, masked_mean


def gaussian_cross_entropy(logits, target, valid_mask, sample_weight=None):
    if logits.shape != target.shape: raise ValueError("heatmap shapes differ")
    target = target.float().flatten(2)
    mass = target.sum(-1)
    probability = target / mass[...,None].clamp_min(1e-12)
    loss = -(probability*F.log_softmax(logits.float().flatten(2),dim=-1)).sum(-1)
    valid = valid_mask*(mass>0).to(valid_mask.dtype)
    return masked_mean(loss,valid,sample_weight=sample_weight)


class SpatialLandmarkCriterion(LandmarkCriterion):
    def _heatmap_loss(self,pred,target,landmark_mask,sample_weight):
        return gaussian_cross_entropy(pred,target,landmark_mask,sample_weight)
