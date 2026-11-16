from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


ELLIPSE_AUXILIARY_LOSS_NAMES = (
    "ellipse_latent",
    "ellipse_reconstruction",
    "ellipse_center",
    "ellipse_orthogonality",
)


def ellipse_auxiliary_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    task_names: Sequence[str],
    ellipse_task_names: set[str],
    required: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute target-only soft ellipse supervision for valid four-point samples.

    ``ellipse_latent`` is represented in pixels as ``(cx, cy, ax, ay, bx,
    by)``.  The corresponding auxiliary endpoints are reconstructed by the
    model, but neither tensor replaces the dense landmark prediction.
    """

    anchor = outputs["coords"]
    zero = anchor.sum() * 0.0
    empty = {name: zero for name in ELLIPSE_AUXILIARY_LOSS_NAMES}
    required_outputs = {"ellipse_latent", "ellipse_aux_coords", "ellipse_scale"}
    missing = sorted(required_outputs - set(outputs))
    if missing:
        if required:
            raise ValueError(
                "ellipse auxiliary loss is enabled but model outputs are missing: "
                + ", ".join(missing)
            )
        return empty

    target = batch["landmarks"].to(device=anchor.device, dtype=anchor.dtype)
    valid_mask = batch["valid_mask"].to(device=anchor.device, dtype=anchor.dtype)
    task_id = batch["task_id"].to(device=anchor.device)
    if target.shape[1] < 4 or valid_mask.shape[1] < 4:
        if required:
            raise ValueError("ellipse auxiliary loss requires four landmark slots")
        return empty

    ellipse_ids = [
        index for index, name in enumerate(task_names) if name in ellipse_task_names
    ]
    if not ellipse_ids:
        return empty
    task_selected = torch.zeros_like(task_id, dtype=torch.bool)
    for index in ellipse_ids:
        task_selected |= task_id == int(index)
    sample_mask = valid_mask[:, :4].prod(dim=1) * task_selected.to(valid_mask.dtype)
    if not bool(sample_mask.any()):
        return empty

    scale_tensor = outputs["ellipse_scale"].to(device=anchor.device, dtype=anchor.dtype)
    if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor).all()):
        raise ValueError("ellipse_scale must be one finite scalar")
    scale = scale_tensor.reshape(()).clamp_min(1.0)
    target_points = target[:, :4]
    target_latent = axis_points_to_ellipse_latent(target_points)
    pred_latent = outputs["ellipse_latent"].to(device=anchor.device, dtype=anchor.dtype)
    pred_aux = outputs["ellipse_aux_coords"].to(device=anchor.device, dtype=anchor.dtype)
    if pred_latent.shape != target_latent.shape:
        raise ValueError("ellipse_latent shape does not match the six-value target")
    if pred_aux.shape != target_points.shape:
        raise ValueError("ellipse_aux_coords shape does not match four target points")

    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=anchor.device, dtype=anchor.dtype)
    latent_per_sample = F.smooth_l1_loss(
        pred_latent / scale,
        target_latent / scale,
        reduction="none",
    ).mean(dim=1)
    reconstruction_per_sample = F.smooth_l1_loss(
        pred_aux / scale,
        target_points / scale,
        reduction="none",
    ).mean(dim=(1, 2))

    pred_points = anchor[:, :4]
    pred_center_delta = (
        (pred_points[:, 0] + pred_points[:, 1])
        - (pred_points[:, 2] + pred_points[:, 3])
    ) * 0.5
    target_center_delta = (
        (target_points[:, 0] + target_points[:, 1])
        - (target_points[:, 2] + target_points[:, 3])
    ) * 0.5
    center_per_sample = F.smooth_l1_loss(
        pred_center_delta / scale,
        target_center_delta / scale,
        reduction="none",
    ).mean(dim=1)

    axes = pred_latent[:, 2:].view(-1, 2, 2)
    dot = (axes[:, 0] * axes[:, 1]).sum(dim=1)
    denom = (
        torch.linalg.norm(axes[:, 0], dim=1)
        * torch.linalg.norm(axes[:, 1], dim=1)
    ).clamp_min(1e-6)
    orthogonality_per_sample = (dot / denom).square()

    return {
        "ellipse_latent": _weighted_sample_mean(
            latent_per_sample, sample_mask, sample_weight
        ),
        "ellipse_reconstruction": _weighted_sample_mean(
            reconstruction_per_sample, sample_mask, sample_weight
        ),
        "ellipse_center": _weighted_sample_mean(
            center_per_sample, sample_mask, sample_weight
        ),
        "ellipse_orthogonality": _weighted_sample_mean(
            orthogonality_per_sample, sample_mask, sample_weight
        ),
    }


def axis_points_to_ellipse_latent(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError("axis points must have shape (batch, 4, 2)")
    center_a = (points[:, 0] + points[:, 1]) * 0.5
    center_b = (points[:, 2] + points[:, 3]) * 0.5
    center = (center_a + center_b) * 0.5
    axis_a = (points[:, 1] - points[:, 0]) * 0.5
    axis_b = (points[:, 3] - points[:, 2]) * 0.5
    return torch.cat([center, axis_a, axis_b], dim=1)


def ellipse_latent_to_axis_points(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != 6:
        raise ValueError("ellipse latent must have shape (batch, 6)")
    center = latent[:, :2]
    axes = latent[:, 2:].view(-1, 2, 2)
    return torch.stack(
        [
            center - axes[:, 0],
            center + axes[:, 0],
            center - axes[:, 1],
            center + axes[:, 1],
        ],
        dim=1,
    )


def _weighted_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    if sample_weight is not None:
        weight = weight * sample_weight.reshape(-1).to(
            device=value.device, dtype=value.dtype
        )
    return (value * weight).sum() / weight.sum().clamp_min(eps)
