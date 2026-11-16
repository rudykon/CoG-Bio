from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from clingeo.biometry.measurement_functions import derive_params_torch
from clingeo.losses.cardiac_topology_auxiliary import (
    CARDIAC_TOPOLOGY_AUXILIARY_LOSS_NAMES,
    cardiac_topology_auxiliary_losses,
)
from clingeo.losses.ellipse_auxiliary import (
    ELLIPSE_AUXILIARY_LOSS_NAMES,
    ellipse_auxiliary_losses,
)
from clingeo.losses.line_auxiliary import (
    LINE_AUXILIARY_LOSS_NAMES,
    line_auxiliary_losses,
)
from clingeo.postprocess.shape_prior import load_shape_priors, priors_to_torch, shape_prior_loss_torch


class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha: float = 2.1, omega: float = 14.0, epsilon: float = 1.0, theta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmark_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        spatial_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        delta = (target - pred).abs()
        alpha_y = self.alpha - target
        small = self.omega * torch.log1p((delta / self.epsilon).pow(alpha_y))
        theta_ratio = pred.new_tensor(self.theta / self.epsilon)
        a = self.omega * (1.0 / (1.0 + theta_ratio.pow(alpha_y))) * alpha_y
        a = a * theta_ratio.pow(alpha_y - 1.0) / self.epsilon
        c = self.theta * a - self.omega * torch.log1p(theta_ratio.pow(alpha_y))
        large = a * delta - c
        loss = torch.where(delta < self.theta, small, large)
        if landmark_mask is None:
            return spatial_weighted_mean(loss, sample_weight)
        return spatial_landmark_weighted_mean(
            loss,
            landmark_mask,
            sample_weight=sample_weight,
            spatial_weight=spatial_weight,
        )


class WingLoss(nn.Module):
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * torch.log(torch.tensor(1.0 + w / epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        x = (pred - target).abs()
        c = self.c.to(device=x.device, dtype=x.dtype)
        loss = torch.where(x < self.w, self.w * torch.log1p(x / self.epsilon), x - c)
        return masked_mean(loss.sum(dim=-1), mask, sample_weight=sample_weight)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    if sample_weight is not None:
        weight = sample_weight.to(device=value.device, dtype=value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        mask = mask * weight
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def spatial_weighted_mean(value: torch.Tensor, sample_weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if sample_weight is None:
        return value.mean()
    per_sample = value.reshape(value.shape[0], -1).mean(dim=1)
    weight = sample_weight.to(device=value.device, dtype=value.dtype).reshape(-1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(eps)


def spatial_landmark_weighted_mean(
    value: torch.Tensor,
    landmark_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    spatial_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    mask = landmark_mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    denominator_weight = mask.expand_as(value)
    numerator_weight = denominator_weight
    if spatial_weight is not None:
        numerator_weight = numerator_weight * spatial_weight.to(device=value.device, dtype=value.dtype)
    if sample_weight is not None:
        sample = sample_weight.to(device=value.device, dtype=value.dtype).reshape(-1)
        while sample.ndim < value.ndim:
            sample = sample.unsqueeze(-1)
        sample = sample.expand_as(value)
        numerator_weight = numerator_weight * sample
        denominator_weight = denominator_weight * sample
    return (value * numerator_weight).sum() / denominator_weight.sum().clamp_min(eps)


class LandmarkCriterion(nn.Module):
    def __init__(self, schema, weights: dict | None = None, heatmap_type: str = "adaptive_wing", shape_prior_path: str | None = None):
        super().__init__()
        self.schema = schema
        self.weights = {
            "heatmap": 1.0,
            "coord_l1": 2.0,
            "coord_heatmap": 0.0,
            "coord_query": 0.0,
            "wing": 1.0,
            "uncertainty_nll": 0.2,
            "parameter": 1.5,
            "geometry": 0.5,
            "shape_prior": 0.0,
            "ellipse_latent": 0.0,
            "ellipse_reconstruction": 0.0,
            "ellipse_center": 0.0,
            "ellipse_orthogonality": 0.0,
            "cardiac_topology_latent": 0.0,
            "cardiac_topology_reconstruction": 0.0,
            "cardiac_topology_pair_length": 0.0,
            "cardiac_topology_midpoint": 0.0,
            "line_latent": 0.0,
            "line_reconstruction": 0.0,
            "line_midpoint": 0.0,
            "line_direction": 0.0,
        }
        self.heatmap_foreground_weight = 1.0
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items() if k in self.weights})
            self.heatmap_foreground_weight = float(weights.get("heatmap_foreground_weight", self.heatmap_foreground_weight))
        self.heatmap_loss = AdaptiveWingLoss() if heatmap_type == "adaptive_wing" else nn.MSELoss()
        self.wing = WingLoss()
        self.task_names = list(schema.tasks.keys())
        self.shape_priors = priors_to_torch(load_shape_priors(shape_prior_path)) if shape_prior_path else {}

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target_hm = batch["heatmap"].to(outputs["heatmap"].device)
        target_xy = batch["landmarks"].to(outputs["coords"].device)
        mask = batch["valid_mask"].to(outputs["coords"].device)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(outputs["coords"].device)
        pred_xy = outputs["coords"]
        losses: dict[str, torch.Tensor] = {}
        losses["heatmap"] = self._heatmap_loss(outputs["heatmap"], target_hm, mask, sample_weight)
        losses["coord_l1"] = masked_mean(F.smooth_l1_loss(pred_xy, target_xy, reduction="none").sum(dim=-1), mask, sample_weight=sample_weight)
        losses["coord_heatmap"] = self._auxiliary_coord_loss(
            outputs,
            ("coord_heatmap", "coords_heatmap"),
            target_xy,
            mask,
            sample_weight=sample_weight,
        )
        losses["coord_query"] = self._auxiliary_coord_loss(
            outputs,
            ("coord_query", "coords_query"),
            target_xy,
            mask,
            sample_weight=sample_weight,
        )
        losses["wing"] = self.wing(pred_xy, target_xy, mask, sample_weight=sample_weight)
        err = torch.linalg.norm(pred_xy - target_xy, dim=-1)
        log_sigma = outputs["log_sigma"].squeeze(-1)
        losses["uncertainty_nll"] = masked_mean(err * torch.exp(-log_sigma) + log_sigma, mask, sample_weight=sample_weight)
        losses["parameter"] = self._parameter_loss(outputs, batch, sample_weight=sample_weight)
        losses["geometry"] = self._geometry_loss(pred_xy, target_xy, mask, batch["task_id"].to(pred_xy.device), sample_weight=sample_weight)
        losses["shape_prior"] = shape_prior_loss_torch(pred_xy, batch["task_id"].to(pred_xy.device), self.task_names, self.shape_priors)
        ellipse_losses = ellipse_auxiliary_losses(
            outputs,
            batch,
            task_names=self.task_names,
            ellipse_task_names={
                name
                for name, task in self.schema.tasks.items()
                if task.geometry_type == "ellipse"
            },
            required=any(
                self.weights[name] != 0.0 for name in ELLIPSE_AUXILIARY_LOSS_NAMES
            ),
        )
        losses.update(ellipse_losses)
        cardiac_topology_losses = cardiac_topology_auxiliary_losses(
            outputs,
            batch,
            task_names=self.task_names,
            cardiac_task_names={
                name
                for name, task in self.schema.tasks.items()
                if task.geometry_type == "cardiac_topology"
            },
            required=any(
                self.weights[name] != 0.0
                for name in CARDIAC_TOPOLOGY_AUXILIARY_LOSS_NAMES
            ),
        )
        losses.update(cardiac_topology_losses)
        line_losses = line_auxiliary_losses(
            outputs,
            batch,
            task_names=self.task_names,
            line_task_names={
                name
                for name, task in self.schema.tasks.items()
                if task.geometry_type == "long_bone"
            },
            required=any(
                self.weights[name] != 0.0 for name in LINE_AUXILIARY_LOSS_NAMES
            ),
        )
        losses.update(line_losses)
        total = pred_xy.new_tensor(0.0)
        metrics = {}
        for key, value in losses.items():
            total = total + self.weights.get(key, 0.0) * value
            metrics[f"loss_{key}"] = float(value.detach().cpu())
        metrics["loss_total"] = float(total.detach().cpu())
        return total, metrics

    def _heatmap_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmark_mask: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        spatial_weight = None
        if self.heatmap_foreground_weight != 1.0:
            spatial_weight = 1.0 + (self.heatmap_foreground_weight - 1.0) * target
        if isinstance(self.heatmap_loss, AdaptiveWingLoss):
            return self.heatmap_loss(
                pred,
                target,
                landmark_mask=landmark_mask,
                sample_weight=sample_weight,
                spatial_weight=spatial_weight,
            )
        return spatial_landmark_weighted_mean(
            F.mse_loss(pred, target, reduction="none"),
            landmark_mask,
            sample_weight=sample_weight,
            spatial_weight=spatial_weight,
        )

    def _auxiliary_coord_loss(
        self,
        outputs: dict[str, torch.Tensor],
        names: tuple[str, ...],
        target_xy: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        pred = next((outputs[name] for name in names if name in outputs), None)
        if pred is None:
            return target_xy.sum() * 0.0
        pred = pred.to(device=target_xy.device, dtype=target_xy.dtype)
        return masked_mean(F.smooth_l1_loss(pred, target_xy, reduction="none").sum(dim=-1), mask, sample_weight=sample_weight)

    def _parameter_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        pred_params = outputs["params"]
        if pred_params.numel() == 0:
            return pred_params.sum() * 0.0
        target = batch["params"].to(pred_params.device)
        mask = batch["param_mask"].to(pred_params.device)
        direct = masked_mean(F.smooth_l1_loss(pred_params, target, reduction="none"), mask, sample_weight=sample_weight) if mask.sum() > 0 else pred_params.sum() * 0.0
        derived_losses = []
        for task_idx in batch["task_id"].unique().tolist():
            task_idx = int(task_idx)
            task_name = self.task_names[task_idx]
            task = self.schema.tasks[task_name]
            sample_mask = batch["task_id"].to(pred_params.device) == task_idx
            if not sample_mask.any() or not task.param_names:
                continue
            coords = outputs["coords"][sample_mask, : task.num_landmarks]
            derived, derived_mask = derive_params_torch(coords, task_name, task.param_names)
            n = min(derived.shape[1], target.shape[1])
            if n == 0:
                continue
            tgt = target[sample_mask, :n]
            pmask = mask[sample_mask, :n] * derived_mask[:, :n]
            if pmask.sum() > 0:
                # Normalize by target magnitude to avoid one parameter dominating.
                denom = tgt.abs().detach().median().clamp_min(1.0)
                local_weight = sample_weight[sample_mask] if sample_weight is not None else None
                derived_losses.append(masked_mean(F.smooth_l1_loss(derived[:, :n] / denom, tgt / denom, reduction="none"), pmask, sample_weight=local_weight))
        if derived_losses:
            return direct + torch.stack(derived_losses).mean()
        return direct

    def _geometry_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, task_id: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        losses = []
        for idx, task_name in enumerate(self.task_names):
            sample_mask = task_id == idx
            if not sample_mask.any():
                continue
            local_weight = sample_weight[sample_mask] if sample_weight is not None else None
            task = self.schema.tasks[task_name]
            p = pred[sample_mask, : task.num_landmarks]
            t = target[sample_mask, : task.num_landmarks]
            m = mask[sample_mask, : task.num_landmarks]
            if task.geometry_type == "long_bone" and task.num_landmarks >= 2:
                pred_len = torch.linalg.norm(p[:, 0] - p[:, 1], dim=-1)
                target_len = torch.linalg.norm(t[:, 0] - t[:, 1], dim=-1)
                losses.append(masked_mean(F.smooth_l1_loss(pred_len, target_len, reduction="none"), (m[:, 0] * m[:, 1]), sample_weight=local_weight))
            elif task.geometry_type == "angle" and task.num_landmarks >= 4:
                from clingeo.biometry.measurement_functions import angle_between_lines_torch

                ap = angle_between_lines_torch(p[:, 0], p[:, 1], p[:, 2], p[:, 3])
                at = angle_between_lines_torch(t[:, 0], t[:, 1], t[:, 2], t[:, 3])
                valid = m[:, :4].prod(dim=1)
                losses.append(masked_mean(F.smooth_l1_loss(torch.sin(torch.deg2rad(ap)), torch.sin(torch.deg2rad(at)), reduction="none"), valid, sample_weight=local_weight))
                losses.append(masked_mean(F.smooth_l1_loss(torch.cos(torch.deg2rad(ap)), torch.cos(torch.deg2rad(at)), reduction="none"), valid, sample_weight=local_weight))
            elif task.geometry_type == "ellipse" and task.num_landmarks >= 4:
                center_p = p.mean(dim=1, keepdim=True)
                center_t = t.mean(dim=1, keepdim=True)
                rp = torch.linalg.norm(p - center_p, dim=-1)
                rt = torch.linalg.norm(t - center_t, dim=-1)
                losses.append(masked_mean(F.smooth_l1_loss(rp, rt, reduction="none"), m, sample_weight=local_weight))
            elif task.geometry_type == "cardiac_topology" and task.num_landmarks >= 4:
                pairs_p = p[:, 0 : (task.num_landmarks // 2) * 2].view(p.shape[0], -1, 2, 2)
                pairs_t = t[:, 0 : (task.num_landmarks // 2) * 2].view(t.shape[0], -1, 2, 2)
                dp = torch.linalg.norm(pairs_p[:, :, 0] - pairs_p[:, :, 1], dim=-1)
                dt = torch.linalg.norm(pairs_t[:, :, 0] - pairs_t[:, :, 1], dim=-1)
                valid = m[:, 0 : (task.num_landmarks // 2) * 2].view(m.shape[0], -1, 2).prod(dim=-1)
                losses.append(masked_mean(F.smooth_l1_loss(dp / dp.detach().median().clamp_min(1.0), dt / dt.detach().median().clamp_min(1.0), reduction="none"), valid, sample_weight=local_weight))
        return torch.stack(losses).mean() if losses else pred.sum() * 0.0
