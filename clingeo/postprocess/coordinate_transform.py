from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AffineTransform:
    matrix: np.ndarray
    inverse: np.ndarray
    output_size: tuple[int, int]
    input_size: tuple[int, int]

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return apply_affine(xy, self.matrix)

    def invert(self, xy: np.ndarray) -> np.ndarray:
        return apply_affine(xy, self.inverse)


def make_letterbox_affine(height: int, width: int, output_size: int | tuple[int, int]) -> AffineTransform:
    if isinstance(output_size, int):
        out_h = out_w = int(output_size)
    else:
        out_h, out_w = map(int, output_size)
    if height <= 0 or width <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("height, width and output_size must be positive")
    scale = min(out_w / float(width), out_h / float(height))
    new_w = width * scale
    new_h = height * scale
    pad_x = (out_w - new_w) / 2.0
    pad_y = (out_h - new_h) / 2.0
    matrix = np.array(
        [
            [scale, 0.0, pad_x],
            [0.0, scale, pad_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(matrix)
    return AffineTransform(matrix=matrix, inverse=inverse, output_size=(out_h, out_w), input_size=(height, width))


def apply_affine(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected xy with last dimension 2, got shape {arr.shape}")
    flat = arr.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([flat, ones], axis=1)
    mapped = hom @ np.asarray(matrix, dtype=np.float64).T
    return mapped[:, :2].reshape(arr.shape).astype(arr.dtype, copy=False)


def apply_affine_torch(xy: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(xy, torch.Tensor):
        raise TypeError(f"xy must be a torch.Tensor, got {type(xy).__name__}")
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"matrix must be a torch.Tensor, got {type(matrix).__name__}")
    if not xy.is_floating_point():
        raise TypeError(f"xy must have a floating dtype, got {xy.dtype}")
    if not matrix.is_floating_point():
        raise TypeError(f"matrix must have a floating dtype, got {matrix.dtype}")
    if xy.ndim not in (2, 3) or xy.shape[-1] != 2:
        raise ValueError(f"xy must have shape [K,2] or [B,K,2], got {tuple(xy.shape)}")
    if xy.shape[-2] == 0 or (xy.ndim == 3 and xy.shape[0] == 0):
        raise ValueError(f"xy must be nonempty, got shape {tuple(xy.shape)}")

    if matrix.ndim == 2:
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape [3,3] or [B,3,3], got {tuple(matrix.shape)}")
    elif matrix.ndim == 3:
        if matrix.shape[1:] != (3, 3):
            raise ValueError(f"matrix must have shape [3,3] or [B,3,3], got {tuple(matrix.shape)}")
        if xy.ndim != 3:
            raise ValueError("a batched matrix requires xy with shape [B,K,2]")
        if matrix.shape[0] != xy.shape[0]:
            raise ValueError(
                "matrix batch size must exactly match xy batch size; "
                f"got {matrix.shape[0]} and {xy.shape[0]}"
            )
    else:
        raise ValueError(f"matrix must have shape [3,3] or [B,3,3], got {tuple(matrix.shape)}")

    if not bool(torch.isfinite(xy).all()):
        raise ValueError("xy must contain only finite values")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")

    converted = matrix.to(device=xy.device, dtype=xy.dtype)
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f"matrix must remain finite after conversion to xy dtype {xy.dtype}")

    affine_last_row = converted.new_tensor([0.0, 0.0, 1.0])
    if converted.ndim == 2:
        last_rows = converted[2]
    else:
        last_rows = converted[:, 2, :]
        affine_last_row = affine_last_row.expand_as(last_rows)
    if not bool(torch.allclose(last_rows, affine_last_row, rtol=0.0, atol=0.0)):
        raise ValueError("matrix has a non-affine last row; expected [0, 0, 1]")

    determinant = converted[..., 0, 0] * converted[..., 1, 1] - converted[..., 0, 1] * converted[..., 1, 0]
    singular = determinant == 0
    if bool(singular.any()):
        indices = singular.nonzero(as_tuple=False).flatten().tolist()
        where = "" if converted.ndim == 2 else f" at batch indices {indices}"
        raise ValueError(f"matrix has a singular 2x2 linear block{where}")

    if converted.ndim == 2:
        return xy @ converted[:2, :2].transpose(0, 1) + converted[:2, 2]
    return torch.bmm(xy, converted[:, :2, :2].transpose(1, 2)) + converted[:, None, :2, 2]


def clamp_coords(xy: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float32).copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, max(float(width - 1), 0.0))
    out[..., 1] = np.clip(out[..., 1], 0.0, max(float(height - 1), 0.0))
    return out
