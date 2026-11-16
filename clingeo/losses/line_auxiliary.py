from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


LINE_AUXILIARY_LOSS_NAMES = (
    "line_latent",
    "line_reconstruction",
    "line_midpoint",
    "line_direction",
)


def line_auxiliary_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    task_names: Sequence[str],
    line_task_names: set[str],
    required: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute soft, endpoint-order-invariant supervision for line tasks.

    The latent is represented in pixels as ``(cx, cy, vx, vy)`` where the
    endpoints are ``centre - vector`` and ``centre + vector``.  This branch is
    training-only: it never replaces or projects the dense landmark output.
    """

    anchor = outputs["coords"]
    zero = anchor.sum() * 0.0
    empty = {name: zero for name in LINE_AUXILIARY_LOSS_NAMES}
    required_outputs = {"line_latent", "line_aux_coords", "line_scale"}
    missing = sorted(required_outputs - set(outputs))
    if missing:
        if required:
            raise ValueError(
                "line auxiliary loss is enabled but model outputs are missing: "
                + ", ".join(missing)
            )
        return empty

    target = batch["landmarks"].to(device=anchor.device, dtype=anchor.dtype)
    valid_mask = batch["valid_mask"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    task_id = batch["task_id"].to(device=anchor.device)
    if target.shape[1] < 2 or valid_mask.shape[1] < 2:
        if required:
            raise ValueError("line auxiliary loss requires two landmark slots")
        return empty

    line_ids = [
        index for index, name in enumerate(task_names) if name in line_task_names
    ]
    if not line_ids:
        return empty
    task_selected = torch.zeros_like(task_id, dtype=torch.bool)
    for index in line_ids:
        task_selected |= task_id == int(index)
    sample_mask = valid_mask[:, :2].prod(dim=1) * task_selected.to(valid_mask.dtype)
    if not bool(sample_mask.any()):
        return empty

    scale_tensor = outputs["line_scale"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor).all()):
        raise ValueError("line_scale must be one finite scalar")
    scale = scale_tensor.reshape(()).clamp_min(1.0)

    target_points = target[:, :2]
    target_latent = endpoints_to_line_latent(target_points)
    reversed_latent = torch.cat(
        [target_latent[:, :2], -target_latent[:, 2:]], dim=1
    )
    pred_latent = outputs["line_latent"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    pred_aux = outputs["line_aux_coords"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    if pred_latent.shape != target_latent.shape:
        raise ValueError("line_latent shape does not match the four-value target")
    if pred_aux.shape != target_points.shape:
        raise ValueError("line_aux_coords shape does not match two target points")

    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=anchor.device, dtype=anchor.dtype)

    latent_direct = F.smooth_l1_loss(
        pred_latent / scale, target_latent / scale, reduction="none"
    ).mean(dim=1)
    latent_reversed = F.smooth_l1_loss(
        pred_latent / scale, reversed_latent / scale, reduction="none"
    ).mean(dim=1)
    latent_per_sample = torch.minimum(latent_direct, latent_reversed)

    reconstruction_direct = F.smooth_l1_loss(
        pred_aux / scale, target_points / scale, reduction="none"
    ).mean(dim=(1, 2))
    reconstruction_reversed = F.smooth_l1_loss(
        pred_aux / scale, target_points.flip(1) / scale, reduction="none"
    ).mean(dim=(1, 2))
    reconstruction_per_sample = torch.minimum(
        reconstruction_direct, reconstruction_reversed
    )

    midpoint_per_sample = F.smooth_l1_loss(
        anchor[:, :2].mean(dim=1) / scale,
        target_points.mean(dim=1) / scale,
        reduction="none",
    ).mean(dim=1)

    pred_vector = pred_latent[:, 2:]
    target_vector = target_latent[:, 2:]
    dot = (pred_vector * target_vector).sum(dim=1)
    denominator = (
        torch.linalg.norm(pred_vector, dim=1)
        * torch.linalg.norm(target_vector, dim=1)
    ).clamp_min(1e-6)
    direction_per_sample = 1.0 - (dot / denominator).square().clamp(0.0, 1.0)

    return {
        "line_latent": _weighted_sample_mean(
            latent_per_sample, sample_mask, sample_weight
        ),
        "line_reconstruction": _weighted_sample_mean(
            reconstruction_per_sample, sample_mask, sample_weight
        ),
        "line_midpoint": _weighted_sample_mean(
            midpoint_per_sample, sample_mask, sample_weight
        ),
        "line_direction": _weighted_sample_mean(
            direction_per_sample, sample_mask, sample_weight
        ),
    }


def endpoints_to_line_latent(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3 or points.shape[1:] != (2, 2):
        raise ValueError("line endpoints must have shape (batch, 2, 2)")
    centre = points.mean(dim=1)
    half_vector = (points[:, 1] - points[:, 0]) * 0.5
    return torch.cat([centre, half_vector], dim=1)


def line_latent_to_endpoints(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != 4:
        raise ValueError("line latent must have shape (batch, 4)")
    centre = latent[:, :2]
    half_vector = latent[:, 2:]
    return torch.stack([centre - half_vector, centre + half_vector], dim=1)


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


__all__ = [
    "LINE_AUXILIARY_LOSS_NAMES",
    "endpoints_to_line_latent",
    "line_auxiliary_losses",
    "line_latent_to_endpoints",
]
