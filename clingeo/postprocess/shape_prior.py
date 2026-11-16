from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def normalize_shape_np(points: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    centered = pts - center
    scale = float(np.sqrt((centered**2).sum(axis=1).mean()))
    scale = max(scale, eps)
    return centered / scale, center, scale


def denormalize_shape_np(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2) * float(scale) + np.asarray(center, dtype=np.float64)


def fit_pca_shapes(
    rows: list[dict[str, Any]],
    min_samples: int = 8,
    max_components: int = 8,
    explained_variance: float = 0.95,
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        domain = str(row["domain"])
        points = np.asarray(row["landmarks"], dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 3 or not np.isfinite(points).all():
            continue
        normalized, _center, _scale = normalize_shape_np(points)
        grouped.setdefault(domain, []).append(normalized.reshape(-1))

    priors: dict[str, dict[str, np.ndarray]] = {}
    for domain, vectors in grouped.items():
        if len(vectors) < min_samples:
            continue
        x = np.stack(vectors, axis=0)
        mean = x.mean(axis=0)
        centered = x - mean
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        denom = max(x.shape[0] - 1, 1)
        var = (s**2) / denom
        ratio = var / max(float(var.sum()), 1e-12)
        cumulative = np.cumsum(ratio)
        k = int(np.searchsorted(cumulative, explained_variance) + 1)
        k = max(1, min(k, max_components, vt.shape[0]))
        priors[domain] = {
            "mean": mean.astype(np.float32),
            "basis": vt[:k].astype(np.float32),
            "explained_variance_ratio": ratio[:k].astype(np.float32),
            "num_samples": np.asarray([len(vectors)], dtype=np.int32),
            "num_landmarks": np.asarray([mean.size // 2], dtype=np.int32),
        }
    return priors


def save_shape_priors(path: str | Path, priors: dict[str, dict[str, np.ndarray]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    domains = []
    for domain, prior in sorted(priors.items()):
        safe = _safe_key(domain)
        domains.append(domain)
        for key, value in prior.items():
            payload[f"{safe}__{key}"] = np.asarray(value)
    payload["__domains__"] = np.asarray(domains)
    np.savez_compressed(path, **payload)
    return path


def load_shape_priors(path: str | Path | None) -> dict[str, dict[str, np.ndarray]]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    domains = [str(x) for x in data["__domains__"].tolist()] if "__domains__" in data else []
    out: dict[str, dict[str, np.ndarray]] = {}
    for domain in domains:
        safe = _safe_key(domain)
        out[domain] = {
            "mean": data[f"{safe}__mean"],
            "basis": data[f"{safe}__basis"],
            "explained_variance_ratio": data[f"{safe}__explained_variance_ratio"],
            "num_samples": data[f"{safe}__num_samples"],
            "num_landmarks": data[f"{safe}__num_landmarks"],
        }
    return out


def project_shape_np(points: np.ndarray, prior: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    normalized, center, scale = normalize_shape_np(points)
    flat = normalized.reshape(-1)
    mean = np.asarray(prior["mean"], dtype=np.float64)
    basis = np.asarray(prior["basis"], dtype=np.float64)
    if flat.size != mean.size:
        return np.asarray(points, dtype=np.float64), float("inf")
    coeff = (flat - mean) @ basis.T
    recon = mean + coeff @ basis
    residual = float(np.sqrt(np.mean((flat - recon) ** 2)))
    return denormalize_shape_np(recon.reshape(-1, 2), center, scale), residual


def gated_shape_projection(
    points: np.ndarray,
    domain: str,
    priors: dict[str, dict[str, np.ndarray]],
    residual_threshold: float = 0.12,
    blend: float = 0.35,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    prior = priors.get(domain)
    if prior is None:
        return points, {"applied": False, "residual": float("nan")}
    projected, residual = project_shape_np(points, prior)
    if not np.isfinite(residual) or residual > residual_threshold:
        return points, {"applied": False, "residual": residual}
    blend = float(np.clip(blend, 0.0, 1.0))
    out = (1.0 - blend) * np.asarray(points, dtype=np.float64) + blend * projected
    return out.astype(np.float32), {"applied": True, "residual": residual}


def shape_prior_loss_torch(
    pred: torch.Tensor,
    task_id: torch.Tensor,
    task_names: list[str],
    priors: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    losses = []
    for idx, domain in enumerate(task_names):
        prior = priors.get(domain)
        if prior is None:
            continue
        sample_mask = task_id == idx
        if not sample_mask.any():
            continue
        n = int(prior["num_landmarks"].item())
        p = pred[sample_mask, :n]
        normalized = _normalize_shape_torch(p).flatten(1)
        mean = prior["mean"].to(device=pred.device, dtype=pred.dtype)
        basis = prior["basis"].to(device=pred.device, dtype=pred.dtype)
        if normalized.shape[1] != mean.numel():
            continue
        coeff = (normalized - mean) @ basis.T
        recon = mean + coeff @ basis
        losses.append(torch.nn.functional.smooth_l1_loss(normalized, recon, reduction="mean"))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def priors_to_torch(priors: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for domain, prior in priors.items():
        out[domain] = {k: torch.as_tensor(v) for k, v in prior.items()}
    return out


def parse_manifest_shape_rows(df) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        raw = row.get("landmarks", "")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            landmarks = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rows.append({"domain": str(row["domain"]), "case_id": str(row.get("case_id", "")), "landmarks": landmarks})
    return rows


def _normalize_shape_torch(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    center = points.mean(dim=1, keepdim=True)
    centered = points - center
    scale = torch.sqrt((centered**2).sum(dim=-1).mean(dim=1, keepdim=True)).clamp_min(eps)
    return centered / scale.unsqueeze(-1)


def _safe_key(domain: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in domain)
