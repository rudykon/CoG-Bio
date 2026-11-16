from __future__ import annotations

import math

import numpy as np
import torch


def euclidean_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def angle_between_lines_torch(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    v1 = p1 - p0
    v2 = p3 - p2
    dot = (v1 * v2).sum(dim=-1)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    angle = torch.atan2(cross.abs(), dot.clamp_min(-1e8))
    return angle * (180.0 / math.pi)


def calibrated_aop_angle_torch(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    # Fallback calibrated on the released AOP training CSV:
    # target AOP ~= 0.977312 * angle(line p1-p2, line p3-p4) - 23.965934.
    # Keep the explicit calibration here instead of pretending the official
    # formula is known; schema still marks AOP as requiring official confirmation.
    angle = angle_between_lines_torch(p0, p1, p2, p3)
    return (0.9773120189814598 * angle - 23.96593414945698).clamp(0.0, 180.0)


def point_to_line_distance_torch(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    line = b - a
    dist = torch.abs(line[:, 0] * (a[:, 1] - p[:, 1]) - (a[:, 0] - p[:, 0]) * line[:, 1])
    return dist / torch.linalg.norm(line, dim=-1).clamp_min(1e-6)


def calibrated_hsd_torch(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    # Released AOP training CSV is closer to a calibrated distance from point 1
    # to the point_3-point_4 line than to the older point_1-to-point_2-point_3
    # fallback. Official formula still requires challenge-side confirmation.
    base = point_to_line_distance_torch(p0, p2, p3)
    return (1.087365918419145 * base + 3.871331305968764).clamp_min(0.0)


def four_point_ellipse_perimeter_torch(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-2] < 4:
        raise ValueError("four_point_ellipse_perimeter_torch expects at least 4 points")
    # Training annotations for HC/FA use p1-p2 and p3-p4 as the two ellipse axes.
    # This matches the provided FA target column; official HC spacing/formula remains TODO.
    axis_1 = euclidean_distance_torch(points[:, 0], points[:, 1]).clamp_min(1e-6)
    axis_2 = euclidean_distance_torch(points[:, 2], points[:, 3]).clamp_min(1e-6)
    major = torch.maximum(axis_1, axis_2)
    minor = torch.minimum(axis_1, axis_2)
    a = major / 2.0
    b = minor / 2.0
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-6)
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + torch.sqrt(4.0 - 3.0 * h).clamp_min(1e-6)))


def derive_params_torch(coords: torch.Tensor, task_name: str, param_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    b, k, _ = coords.shape
    values = []
    masks = []
    for i, name in enumerate(param_names):
        upper = name.upper()
        if k >= 2 and upper in {"FL", "IVC", "FUGC"}:
            values.append(euclidean_distance_torch(coords[:, 0], coords[:, 1]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif task_name == "PSAX" and k >= 4 and upper == "RVOT":
            values.append(euclidean_distance_torch(coords[:, 0], coords[:, 1]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif task_name == "PSAX" and k >= 4 and upper == "PA":
            values.append(euclidean_distance_torch(coords[:, 2], coords[:, 3]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif task_name in {"HC", "AC", "FA"} and k >= 4:
            values.append(four_point_ellipse_perimeter_torch(coords[:, :4]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif task_name == "AOP" and k >= 4 and upper == "AOP":
            values.append(calibrated_aop_angle_torch(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif task_name == "AOP" and k >= 4 and upper == "HSD":
            values.append(calibrated_hsd_torch(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        elif k >= 2:
            pairs = coords[:, 0 : (k // 2) * 2].view(b, -1, 2, 2)
            values.append(torch.linalg.norm(pairs[:, :, 0] - pairs[:, :, 1], dim=-1).mean(dim=-1))
            masks.append(torch.ones(b, device=coords.device, dtype=coords.dtype))
        else:
            values.append(torch.zeros(b, device=coords.device, dtype=coords.dtype))
            masks.append(torch.zeros(b, device=coords.device, dtype=coords.dtype))
    if not values:
        return coords.new_zeros((b, 0)), coords.new_zeros((b, 0))
    return torch.stack(values, dim=1), torch.stack(masks, dim=1)


def euclidean_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))
