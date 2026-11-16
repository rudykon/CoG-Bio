from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


CARDIAC_TOPOLOGY_AUXILIARY_LOSS_NAMES = (
    "cardiac_topology_latent",
    "cardiac_topology_reconstruction",
    "cardiac_topology_pair_length",
    "cardiac_topology_midpoint",
)


def cardiac_topology_auxiliary_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    task_names: Sequence[str],
    cardiac_task_names: set[str],
    required: bool = False,
) -> dict[str, torch.Tensor]:
    """Apply pair-topology supervision without replacing dense predictions.

    Consecutive landmarks form the frozen topology pairs already used by the
    project evaluator.  The auxiliary latent stores each pair as
    ``(centre_x, centre_y, half_vector_x, half_vector_y)``.  Its reconstructed
    endpoints are training-only evidence; deployable coordinates remain the
    dense heatmap coordinates in ``outputs["coords"]``.
    """

    anchor = outputs["coords"]
    zero = anchor.sum() * 0.0
    empty = {name: zero for name in CARDIAC_TOPOLOGY_AUXILIARY_LOSS_NAMES}
    required_outputs = {
        "cardiac_topology_latent",
        "cardiac_topology_aux_coords",
        "cardiac_topology_scale",
    }
    missing = sorted(required_outputs - set(outputs))
    if missing:
        if required:
            raise ValueError(
                "cardiac topology auxiliary loss is enabled but model outputs "
                "are missing: " + ", ".join(missing)
            )
        return empty

    target = batch["landmarks"].to(device=anchor.device, dtype=anchor.dtype)
    valid_mask = batch["valid_mask"].to(device=anchor.device, dtype=anchor.dtype)
    task_id = batch["task_id"].to(device=anchor.device)
    even_slots = (target.shape[1] // 2) * 2
    if even_slots < 4 or valid_mask.shape[1] < even_slots:
        if required:
            raise ValueError(
                "cardiac topology auxiliary loss requires at least two pairs"
            )
        return empty

    cardiac_ids = [
        index for index, name in enumerate(task_names) if name in cardiac_task_names
    ]
    if not cardiac_ids:
        return empty
    task_selected = torch.zeros_like(task_id, dtype=torch.bool)
    for index in cardiac_ids:
        task_selected |= task_id == int(index)

    pair_count = even_slots // 2
    pair_mask = valid_mask[:, :even_slots].view(-1, pair_count, 2).prod(dim=-1)
    pair_mask = pair_mask * task_selected.to(dtype=pair_mask.dtype).unsqueeze(1)
    if not bool(pair_mask.any()):
        return empty

    target_pairs = target[:, :even_slots].view(-1, pair_count, 2, 2)
    target_latent = pair_points_to_topology_latent(target_pairs)
    pred_latent = outputs["cardiac_topology_latent"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    pred_aux = outputs["cardiac_topology_aux_coords"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    if (
        pred_latent.ndim != 3
        or pred_latent.shape[0] != target.shape[0]
        or pred_latent.shape[1] < pair_count
        or pred_latent.shape[2] != 4
    ):
        raise ValueError("cardiac_topology_latent shape is incompatible")
    if (
        pred_aux.ndim != 3
        or pred_aux.shape[0] != target.shape[0]
        or pred_aux.shape[1] < even_slots
        or pred_aux.shape[2] != 2
    ):
        raise ValueError("cardiac_topology_aux_coords shape is incompatible")
    pred_latent = pred_latent[:, :pair_count]
    pred_aux_pairs = pred_aux[:, :even_slots].view(-1, pair_count, 2, 2)

    scale_tensor = outputs["cardiac_topology_scale"].to(
        device=anchor.device, dtype=anchor.dtype
    )
    if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor).all()):
        raise ValueError("cardiac_topology_scale must be one finite scalar")
    scale = scale_tensor.reshape(()).clamp_min(1.0)
    latent_per_pair = F.smooth_l1_loss(
        pred_latent / scale,
        target_latent / scale,
        reduction="none",
    ).mean(dim=-1)
    reconstruction_per_pair = F.smooth_l1_loss(
        pred_aux_pairs / scale,
        target_pairs / scale,
        reduction="none",
    ).mean(dim=(-1, -2))
    pred_half_vector = pred_latent[..., 2:]
    target_half_vector = target_latent[..., 2:]
    pair_length_per_pair = F.smooth_l1_loss(
        torch.linalg.norm(pred_half_vector, dim=-1) / scale,
        torch.linalg.norm(target_half_vector, dim=-1) / scale,
        reduction="none",
    )

    dense_pairs = anchor[:, :even_slots].view(-1, pair_count, 2, 2)
    dense_midpoints = dense_pairs.mean(dim=2)
    target_midpoints = target_pairs.mean(dim=2)
    midpoint_per_pair = F.smooth_l1_loss(
        dense_midpoints / scale,
        target_midpoints / scale,
        reduction="none",
    ).mean(dim=-1)

    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=anchor.device, dtype=anchor.dtype)
    return {
        "cardiac_topology_latent": _weighted_pair_mean(
            latent_per_pair, pair_mask, sample_weight
        ),
        "cardiac_topology_reconstruction": _weighted_pair_mean(
            reconstruction_per_pair, pair_mask, sample_weight
        ),
        "cardiac_topology_pair_length": _weighted_pair_mean(
            pair_length_per_pair, pair_mask, sample_weight
        ),
        "cardiac_topology_midpoint": _weighted_pair_mean(
            midpoint_per_pair, pair_mask, sample_weight
        ),
    }


def pair_points_to_topology_latent(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 4 or points.shape[-2:] != (2, 2):
        raise ValueError("cardiac pair points must have shape (batch, pairs, 2, 2)")
    centre = points.mean(dim=2)
    half_vector = (points[:, :, 1] - points[:, :, 0]) * 0.5
    return torch.cat([centre, half_vector], dim=-1)


def topology_latent_to_pair_points(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 3 or latent.shape[-1] != 4:
        raise ValueError("cardiac topology latent must have shape (batch, pairs, 4)")
    centre = latent[..., :2]
    half_vector = latent[..., 2:]
    return torch.stack([centre - half_vector, centre + half_vector], dim=2)


def _weighted_pair_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    if sample_weight is not None:
        weight = weight * sample_weight.reshape(-1, 1).to(
            device=value.device, dtype=value.dtype
        )
    return (value * weight).sum() / weight.sum().clamp_min(eps)


__all__ = [
    "CARDIAC_TOPOLOGY_AUXILIARY_LOSS_NAMES",
    "cardiac_topology_auxiliary_losses",
    "pair_points_to_topology_latent",
    "topology_latent_to_pair_points",
]
