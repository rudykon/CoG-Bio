from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from clingeo.postprocess.coordinate_transform import AffineTransform, apply_affine, make_letterbox_affine


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float32)
    cdf = hist.cumsum()
    denom = cdf[-1] - cdf[0]
    if denom <= 0:
        return image.astype(np.float32)
    lut = (cdf - cdf[0]) / denom
    return lut[img].astype(np.float32)


def clahe(image: np.ndarray, tiles: int = 8, clip_limit: float = 2.0, bins: int = 256) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    h, w = img.shape
    if h == 0 or w == 0:
        return img.astype(np.float32)
    n_tiles = max(1, min(int(tiles), h, w))
    n_bins = max(2, int(bins))
    y_edges = np.linspace(0, h, n_tiles + 1, dtype=np.int32)
    x_edges = np.linspace(0, w, n_tiles + 1, dtype=np.int32)
    luts = np.zeros((n_tiles, n_tiles, n_bins), dtype=np.float32)
    bin_idx = np.clip((img * float(n_bins - 1)).round().astype(np.int32), 0, n_bins - 1)

    for yi in range(n_tiles):
        for xi in range(n_tiles):
            patch = bin_idx[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            hist = np.bincount(patch.reshape(-1), minlength=n_bins).astype(np.float32)
            if patch.size == 0 or hist.sum() <= 0:
                luts[yi, xi] = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
                continue
            limit = max(float(clip_limit) * float(patch.size) / float(n_bins), 1.0)
            excess = np.maximum(hist - limit, 0.0)
            hist = np.minimum(hist, limit)
            hist += excess.sum() / float(n_bins)
            cdf = hist.cumsum()
            denom = float(cdf[-1])
            luts[yi, xi] = (cdf / denom).astype(np.float32) if denom > 0 else np.linspace(0.0, 1.0, n_bins, dtype=np.float32)

    out = np.empty_like(img, dtype=np.float32)
    for yi in range(n_tiles):
        y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
        y_alpha = np.linspace(0.0, 1.0, max(y1 - y0, 1), endpoint=False, dtype=np.float32)[:, None]
        yj = min(yi + 1, n_tiles - 1)
        for xi in range(n_tiles):
            x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
            x_alpha = np.linspace(0.0, 1.0, max(x1 - x0, 1), endpoint=False, dtype=np.float32)[None, :]
            xj = min(xi + 1, n_tiles - 1)
            idx = bin_idx[y0:y1, x0:x1]
            top = (1.0 - x_alpha) * luts[yi, xi][idx] + x_alpha * luts[yi, xj][idx]
            bottom = (1.0 - x_alpha) * luts[yj, xi][idx] + x_alpha * luts[yj, xj][idx]
            out[y0:y1, x0:x1] = (1.0 - y_alpha) * top + y_alpha * bottom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage

        sx = ndimage.sobel(image, axis=1, mode="nearest")
        sy = ndimage.sobel(image, axis=0, mode="nearest")
        mag = np.sqrt(sx * sx + sy * sy)
    except Exception:
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1, :] = image[2:, :] - image[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
    max_val = float(mag.max())
    return (mag / max_val).astype(np.float32) if max_val > 0 else mag.astype(np.float32)


def make_three_channel(
    image: np.ndarray,
    ch2: str = "global_he",
    clahe_tiles: int = 8,
    clahe_clip: float = 2.0,
) -> np.ndarray:
    raw = robust_normalize(image)
    if str(ch2) == "clahe":
        enhanced = clahe(raw, tiles=int(clahe_tiles), clip_limit=float(clahe_clip))
    elif str(ch2) == "global_he":
        enhanced = histogram_equalize(raw)
    else:
        raise ValueError(f"Unknown channel-2 preprocessing mode: {ch2}")
    edge = sobel_magnitude(raw)
    return np.stack([raw, enhanced, edge], axis=0).astype(np.float32)


def letterbox_image_tensor(chw: np.ndarray, output_size: int) -> tuple[torch.Tensor, AffineTransform]:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got {chw.shape}")
    c, h, w = chw.shape
    affine = make_letterbox_affine(h, w, output_size)
    scale = affine.matrix[0, 0]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = torch.zeros((1, c, output_size, output_size), dtype=tensor.dtype)
    pad_x = int(round(float(affine.matrix[0, 2])))
    pad_y = int(round(float(affine.matrix[1, 2])))
    out[:, :, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = tensor
    return out.squeeze(0), affine


@dataclass
class GeometryAugmentConfig:
    rotation_deg: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    translate: float = 0.0
    hflip: bool = False


def default_geometry_for_task(geometry_type: str, allow_hflip: bool) -> GeometryAugmentConfig:
    if geometry_type == "angle":
        return GeometryAugmentConfig(rotation_deg=15.0, scale_min=0.85, scale_max=1.2, translate=0.08, hflip=False)
    if geometry_type == "cardiac_topology":
        return GeometryAugmentConfig(rotation_deg=10.0, scale_min=0.8, scale_max=1.25, translate=0.08, hflip=False)
    if geometry_type == "long_bone":
        return GeometryAugmentConfig(rotation_deg=25.0, scale_min=0.8, scale_max=1.3, translate=0.10, hflip=allow_hflip)
    return GeometryAugmentConfig(rotation_deg=25.0, scale_min=0.75, scale_max=1.35, translate=0.10, hflip=allow_hflip)


def geometry_config_for_domain(
    domain: str,
    geometry_type: str,
    allow_hflip: bool,
    overrides: dict | None = None,
) -> GeometryAugmentConfig:
    cfg = default_geometry_for_task(geometry_type, allow_hflip)
    override = (overrides or {}).get(str(domain), {})
    if not override:
        return cfg
    allowed = {field.name for field in GeometryAugmentConfig.__dataclass_fields__.values()}
    values = {str(key): value for key, value in dict(override).items() if str(key) in allowed}
    return replace(cfg, **values)


def apply_hflip_tensor(image: torch.Tensor, coords: np.ndarray, flip_pairs: tuple[tuple[int, int], ...]) -> tuple[torch.Tensor, np.ndarray]:
    _, h, w = image.shape
    flipped = torch.flip(image, dims=[2])
    out = coords.copy()
    out[:, 0] = (w - 1) - out[:, 0]
    for a, b in flip_pairs:
        out[[a, b]] = out[[b, a]]
    return flipped, out


def random_affine_tensor(
    image: torch.Tensor,
    coords: np.ndarray,
    cfg: GeometryAugmentConfig,
) -> tuple[torch.Tensor, np.ndarray]:
    if cfg.rotation_deg <= 0 and cfg.scale_min == cfg.scale_max == 1.0 and cfg.translate <= 0:
        return image, coords
    return affine_tensor_from_normalized(
        image,
        coords,
        cfg,
        rotation=random.random(),
        scale=random.random(),
        translate_x=random.random(),
        translate_y=random.random(),
    )


def affine_tensor_from_normalized(
    image: torch.Tensor,
    coords: np.ndarray,
    cfg: GeometryAugmentConfig,
    *,
    rotation: float,
    scale: float,
    translate_x: float,
    translate_y: float,
) -> tuple[torch.Tensor, np.ndarray]:
    if cfg.rotation_deg <= 0 and cfg.scale_min == cfg.scale_max == 1.0 and cfg.translate <= 0:
        return image, coords
    rotation_unit = _normalized_draw(rotation, "rotation")
    scale_unit = _normalized_draw(scale, "scale")
    translate_x_unit = _normalized_draw(translate_x, "translate_x")
    translate_y_unit = _normalized_draw(translate_y, "translate_y")
    _, h, w = image.shape
    angle = math.radians((2.0 * rotation_unit - 1.0) * cfg.rotation_deg)
    scale_value = cfg.scale_min + scale_unit * (cfg.scale_max - cfg.scale_min)
    tx = (2.0 * translate_x_unit - 1.0) * cfg.translate * w
    ty = (2.0 * translate_y_unit - 1.0) * cfg.translate * h
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    cos_a = math.cos(angle) * scale_value
    sin_a = math.sin(angle) * scale_value
    matrix = np.array(
        [
            [cos_a, -sin_a, cx + tx - cos_a * cx + sin_a * cy],
            [sin_a, cos_a, cy + ty - sin_a * cx - cos_a * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    theta = torch.tensor(
        [
            [cos_a, -sin_a, (2.0 * matrix[0, 2] / max(w - 1, 1))],
            [sin_a, cos_a, (2.0 * matrix[1, 2] / max(h - 1, 1))],
        ],
        dtype=image.dtype,
        device=image.device,
    )
    # Use inverse mapping for grid_sample.
    inv = np.linalg.inv(matrix)
    norm = np.array([[2.0 / max(w - 1, 1), 0, -1], [0, 2.0 / max(h - 1, 1), -1], [0, 0, 1]], dtype=np.float64)
    denorm = np.array([[(w - 1) / 2.0, 0, (w - 1) / 2.0], [0, (h - 1) / 2.0, (h - 1) / 2.0], [0, 0, 1]], dtype=np.float64)
    theta_np = norm @ inv @ denorm
    theta = torch.from_numpy(theta_np[:2]).to(dtype=image.dtype, device=image.device).unsqueeze(0)
    grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=True)
    warped = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0)
    return warped, apply_affine(coords, matrix).astype(np.float32)


def ultrasound_physics_from_normalized(
    image: torch.Tensor,
    *,
    strength: str,
    speckle_gate: float,
    speckle_variance: float,
    gamma_gate: float,
    gamma: float,
    gain_gate: float,
    gain: float,
    bias: float,
    shadow_gate: float,
    shadow_x: float,
    shadow_width: float,
    shadow_attenuation: float,
    depth_gate: float,
    depth_alpha: float,
    noise_seed: int,
) -> torch.Tensor:
    if strength not in {"none", "weak", "strong"}:
        raise ValueError("ultrasound augmentation strength is invalid")
    values = {
        "speckle_gate": speckle_gate,
        "speckle_variance": speckle_variance,
        "gamma_gate": gamma_gate,
        "gamma": gamma,
        "gain_gate": gain_gate,
        "gain": gain,
        "bias": bias,
        "shadow_gate": shadow_gate,
        "shadow_x": shadow_x,
        "shadow_width": shadow_width,
        "shadow_attenuation": shadow_attenuation,
        "depth_gate": depth_gate,
        "depth_alpha": depth_alpha,
    }
    normalized = {name: _normalized_draw(value, name) for name, value in values.items()}
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int) or noise_seed < 0:
        raise ValueError("noise_seed must be a non-negative integer")
    if strength == "none":
        return image
    out = image.clone()
    if normalized["speckle_gate"] < 0.7:
        maximum = 0.04 if strength == "strong" else 0.015
        variance = 0.005 + normalized["speckle_variance"] * (maximum - 0.005)
        generator = torch.Generator(device=out.device)
        generator.manual_seed(noise_seed % (2**63 - 1))
        noise = torch.randn(
            out.shape,
            dtype=out.dtype,
            device=out.device,
            generator=generator,
        ) * math.sqrt(variance)
        out = out + out * noise
    if normalized["gamma_gate"] < 0.5:
        gamma_value = 0.6 + normalized["gamma"] * 1.2
        out = out.clamp(0, 1).pow(gamma_value)
    if normalized["gain_gate"] < 0.5:
        gain_value = 0.7 + normalized["gain"] * 0.7
        bias_value = -0.1 + normalized["bias"] * 0.2
        out = out * gain_value + bias_value
    if normalized["shadow_gate"] < 0.35:
        _, h, w = out.shape
        x0 = min(int(normalized["shadow_x"] * w), max(w - 1, 0))
        width = max(1, int(w * (0.05 + normalized["shadow_width"] * 0.2)))
        attenuation = 0.2 + normalized["shadow_attenuation"] * 0.6
        x1 = min(w, x0 + width)
        ramp = torch.linspace(
            1.0,
            attenuation,
            h,
            dtype=out.dtype,
            device=out.device,
        ).view(1, h, 1)
        out[:, :, x0:x1] = out[:, :, x0:x1] * ramp
    if normalized["depth_gate"] < 0.35:
        _, h, _w = out.shape
        alpha = 0.1 + normalized["depth_alpha"] * 0.7
        depth = torch.linspace(
            0, 1, h, dtype=out.dtype, device=out.device
        ).view(1, h, 1)
        out = out * torch.exp(-alpha * depth)
    return out.clamp(0.0, 1.0)


def _normalized_draw(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a normalized finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{field} must be in [0, 1)")
    return parsed


def apply_ultrasound_physics(image: torch.Tensor, strength: str = "strong") -> torch.Tensor:
    if strength == "none":
        return image
    out = image
    if random.random() < 0.7:
        var = random.uniform(0.005, 0.04 if strength == "strong" else 0.015)
        noise = torch.randn_like(out) * math.sqrt(var)
        out = out + out * noise
    if random.random() < 0.5:
        gamma = random.uniform(0.6, 1.8)
        out = out.clamp(0, 1).pow(gamma)
    if random.random() < 0.5:
        gain = random.uniform(0.7, 1.4)
        bias = random.uniform(-0.1, 0.1)
        out = out * gain + bias
    if random.random() < 0.35:
        _, h, w = out.shape
        x0 = random.randint(0, max(w - 1, 0))
        width = max(1, int(w * random.uniform(0.05, 0.25)))
        attenuation = random.uniform(0.2, 0.8)
        x1 = min(w, x0 + width)
        ramp = torch.linspace(1.0, attenuation, h, dtype=out.dtype, device=out.device).view(1, h, 1)
        out[:, :, x0:x1] = out[:, :, x0:x1] * ramp
    if random.random() < 0.35:
        _, h, _w = out.shape
        alpha = random.uniform(0.1, 0.8)
        depth = torch.linspace(0, 1, h, dtype=out.dtype, device=out.device).view(1, h, 1)
        out = out * torch.exp(-alpha * depth)
    return out.clamp(0.0, 1.0)
