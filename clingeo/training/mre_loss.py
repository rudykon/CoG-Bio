"""Case-balanced Euclidean coordinate objective in original-image pixel units."""
import torch
import torch.nn.functional as F
from clingeo.training.pixel_weighting import original_pixel_scale


def case_mre(predicted, target, mask, inverse):
    valid = mask.float()
    count = valid.sum(-1)
    if (count<=0).any(): raise ValueError("each case requires a valid landmark")
    error = torch.linalg.vector_norm(predicted.float()-target.float(),dim=-1)
    return ((error*valid).sum(-1)/count*original_pixel_scale(inverse)).mean()


def mre_objective(outputs, batch, weights):
    target,mask,inverse = batch["landmarks"],batch["valid_mask"],batch["affine_inv"]
    losses = {"mre":case_mre(outputs["coords"],target,mask,inverse),
              "spatial_mre":case_mre(outputs["coord_heatmap"],target,mask,inverse),
              "query_mre":case_mre(outputs["coord_query"],target,mask,inverse)}
    heatmap = batch["heatmap"].float().flatten(2)
    mass = heatmap.sum(-1)
    valid = mask.float()*(mass>0).float()
    cross_entropy = -(heatmap/mass[...,None].clamp_min(1e-12)*
                      F.log_softmax(outputs["heatmap"].float().flatten(2),-1)).sum(-1)
    losses["spatial_ce"] = ((cross_entropy*valid).sum(-1)/valid.sum(-1).clamp_min(1)).mean()
    total = sum(weights[key]*value for key,value in losses.items())
    return total,{key:float(value.detach()) for key,value in losses.items()}
