from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clingeo.data.image_io import read_grayscale
from clingeo.data.manifest import load_manifest, parse_json_array, resolve_image_path
from clingeo.data.task_schema import LandmarkSchema, load_landmark_schema
from clingeo.data.transforms import (
    apply_hflip_tensor,
    apply_ultrasound_physics,
    geometry_config_for_domain,
    letterbox_image_tensor,
    make_three_channel,
    random_affine_tensor,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BiometryDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        data_root: str | Path,
        schema: LandmarkSchema | str | Path = "configs/landmark_schema.yaml",
        input_size: int = 768,
        heatmap_stride: int = 4,
        training: bool = False,
        cache_dir: str | Path | None = "data/processed/image_cache",
        ultrasound_aug: str = "strong",
        geometry_overrides: dict[str, Any] | None = None,
        preprocess: dict[str, Any] | None = None,
        require_keyed_augmentation: bool = False,
    ):
        self.df = load_manifest(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True).copy()
        source_scope = self.df.get("source_scope", pd.Series("", index=self.df.index)).fillna("").astype(str)
        resolution = self.df.get("oof_ssl_resolution_state", pd.Series("", index=self.df.index)).fillna("").astype(str)
        forbidden = source_scope.isin({"public_validation", "public_validation_exclusion"}) | resolution.eq("public_overlap_excluded")
        if forbidden.any():
            raise ValueError("public-validation identity row cannot be loaded for training or inference")
        self.data_root = Path(data_root)
        self.schema = load_landmark_schema(schema) if not isinstance(schema, LandmarkSchema) else schema
        self.input_size = int(input_size)
        self.heatmap_stride = int(heatmap_stride)
        self.training = bool(training)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ultrasound_aug = ultrasound_aug
        self.geometry_overrides = dict(geometry_overrides or {})
        self.preprocess = dict(preprocess or {})
        self.require_keyed_augmentation = bool(require_keyed_augmentation)
        if self.require_keyed_augmentation and not self.training:
            raise ValueError("keyed augmentation can only be required for training")
        self.sigma_scale = 1.0
        self.task_to_idx = self.schema.task_index()
        self.max_landmarks = self.schema.max_landmarks
        self.max_params = self.schema.max_params

    def __len__(self) -> int:
        return len(self.df)

    def set_sigma_scale(self, scale: float) -> None:
        self.sigma_scale = float(scale)

    def __getitem__(self, index: int | tuple[int, Any]) -> dict[str, Any]:
        exposure = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("keyed dataset index must contain row index and exposure")
            index, exposure = index
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("dataset row index must be an integer")
        if exposure is not None and not self.training:
            raise ValueError("evaluation dataset cannot consume a training exposure")
        if self.require_keyed_augmentation and exposure is None:
            raise ValueError("production training requires a keyed exposure record")
        row = self.df.iloc[index]
        domain = str(row["domain"])
        task = self.schema.get(domain)
        task_idx = self.task_to_idx[task.name]
        image_path = resolve_image_path(self.data_root, "train" if str(row.get("split", "")).startswith("train") else "validation", domain, str(row["image_path"]))
        expected_sha256 = _manifest_image_sha256(row)
        image = read_grayscale(
            image_path,
            cache_dir=self.cache_dir,
            expected_sha256=expected_sha256,
        )
        chw = make_three_channel(
            image,
            ch2=str(self.preprocess.get("ch2", "global_he")),
            clahe_tiles=int(self.preprocess.get("clahe_tiles", 8)),
            clahe_clip=float(self.preprocess.get("clahe_clip", 2.0)),
        )
        image_tensor, affine = letterbox_image_tensor(chw, self.input_size)

        landmarks, valid_mask = self._load_landmarks(row, task.num_landmarks)
        if landmarks.size:
            landmarks = affine.apply(landmarks).astype(np.float32)
        elif exposure is not None:
            raise ValueError("keyed supervised exposure requires labeled landmarks")
        if self.training and landmarks.size:
            cfg = geometry_config_for_domain(domain, task.geometry_type, task.allow_hflip, self.geometry_overrides)
            if exposure is not None:
                from clingeo.training.keyed_transforms import (
                    apply_keyed_supervised_augmentation,
                )

                row_id = str(row.get("manifest_row_id", "")).strip()
                if not row_id or exposure.manifest_row_id != row_id:
                    raise ValueError("keyed exposure manifest_row_id mismatch")
                image_tensor, landmarks, transform_audit = (
                    apply_keyed_supervised_augmentation(
                        image_tensor,
                        landmarks,
                        geometry=cfg,
                        allow_hflip=task.allow_hflip,
                        flip_pairs=task.flip_pairs,
                        ultrasound_strength=self.ultrasound_aug,
                        exposure=exposure,
                    )
                )
            else:
                if task.allow_hflip and cfg.hflip and np.random.rand() < 0.5:
                    image_tensor, landmarks = apply_hflip_tensor(image_tensor, landmarks, task.flip_pairs)
                image_tensor, landmarks = random_affine_tensor(image_tensor, landmarks, cfg)
                image_tensor = apply_ultrasound_physics(image_tensor, strength=self.ultrasound_aug)

        padded_landmarks = np.zeros((self.max_landmarks, 2), dtype=np.float32)
        padded_valid = np.zeros((self.max_landmarks,), dtype=np.float32)
        if landmarks.size:
            n = min(task.num_landmarks, self.max_landmarks, landmarks.shape[0])
            padded_landmarks[:n] = landmarks[:n]
            padded_valid[:n] = valid_mask[:n]

        heatmaps = make_heatmaps(
            padded_landmarks,
            padded_valid,
            num_landmarks=self.max_landmarks,
            out_size=self.input_size // self.heatmap_stride,
            stride=self.heatmap_stride,
            sigma=float(task.heatmap_sigma) * float(self.sigma_scale),
        )
        clinical_params = self._load_clinical_params(row)
        params, param_mask = self._load_params_from_clinical(clinical_params, task)
        item = {
            "image": image_tensor.float(),
            "landmarks": torch.from_numpy(padded_landmarks),
            "valid_mask": torch.from_numpy(padded_valid),
            "heatmap": torch.from_numpy(heatmaps),
            "params": torch.from_numpy(params),
            "param_mask": torch.from_numpy(param_mask),
            "sample_weight": torch.tensor(_sample_weight(row), dtype=torch.float32),
            "task_id": torch.tensor(task_idx, dtype=torch.long),
            "domain": domain,
            "case_id": str(row["case_id"]),
            "height": torch.tensor(int(row.get("height", image.shape[0]) or image.shape[0]), dtype=torch.long),
            "width": torch.tensor(int(row.get("width", image.shape[1]) or image.shape[1]), dtype=torch.long),
            "affine": torch.from_numpy(affine.matrix.astype(np.float32)),
            "affine_inv": torch.from_numpy(affine.inverse.astype(np.float32)),
            "clean_eval_ready": torch.tensor(not self.training, dtype=torch.bool),
            "clinical_params": clinical_params,
            "image_path": str(image_path),
            "num_landmarks": torch.tensor(task.num_landmarks, dtype=torch.long),
        }
        identity_defaults = {
            "manifest_row_id": "",
            "hidden_identity_group": "",
            "oof_ssl_resolution_state": "",
            "source_scope": "",
            "identity_resolved": False,
            "partition_eligible": False,
            "ssl_eligible": False,
        }
        for key, default in identity_defaults.items():
            value = row.get(key, default)
            if isinstance(default, bool):
                item[key] = bool(value) if not pd.isna(value) else default
            else:
                item[key] = "" if pd.isna(value) else str(value)
        if exposure is not None:
            item["exposure_record"] = exposure
            item["augmentation_record_sha256"] = transform_audit.record_sha256
            item["augmentation_draw_sha256"] = transform_audit.draw_sha256
            item["augmentation_exposure_index"] = transform_audit.exposure_index
        return item

    def _load_landmarks(self, row: pd.Series, num_landmarks: int) -> tuple[np.ndarray, np.ndarray]:
        value = row.get("landmarks", "")
        if not isinstance(value, str) or not value:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        landmarks = parse_json_array(value, dtype=np.float32).reshape(-1, 2)
        valid_value = row.get("valid_mask", "")
        if isinstance(valid_value, str) and valid_value:
            valid = parse_json_array(valid_value, dtype=np.float32).reshape(-1)
        else:
            valid = np.isfinite(landmarks).all(axis=1).astype(np.float32)
        if landmarks.shape[0] != num_landmarks:
            raise ValueError(f"{row.get('case_id')} has {landmarks.shape[0]} landmarks, schema expects {num_landmarks}")
        landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
        return landmarks.astype(np.float32), valid.astype(np.float32)

    def _load_clinical_params(self, row: pd.Series) -> dict[str, float]:
        raw = row.get("clinical_params", "{}")
        clinical = json.loads(raw) if isinstance(raw, str) and raw else {}
        if not isinstance(clinical, dict):
            raise ValueError(f"{row.get('case_id')} clinical_params must be a JSON object")
        return {str(key): float(value) for key, value in clinical.items()}

    def _load_params_from_clinical(self, clinical: dict[str, float], task) -> tuple[np.ndarray, np.ndarray]:
        params = np.zeros((self.max_params,), dtype=np.float32)
        mask = np.zeros((self.max_params,), dtype=np.float32)
        for i, param_name in enumerate(task.param_names[: self.max_params]):
            if param_name in clinical:
                params[i] = float(clinical[param_name])
                mask[i] = 1.0
        return params, mask


def make_heatmaps(
    coords: np.ndarray,
    valid_mask: np.ndarray,
    num_landmarks: int,
    out_size: int,
    stride: int,
    sigma: float,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(out_size, dtype=np.float32), np.arange(out_size, dtype=np.float32), indexing="ij")
    heatmaps = np.zeros((num_landmarks, out_size, out_size), dtype=np.float32)
    scaled = coords.astype(np.float32) / float(stride)
    sigma = max(float(sigma) / float(stride), 0.5)
    denom = 2.0 * sigma * sigma
    for i in range(min(num_landmarks, scaled.shape[0])):
        if valid_mask[i] <= 0:
            continue
        x, y = scaled[i]
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom)
    return heatmaps


def collate_biometry(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = {
        "image",
        "landmarks",
        "valid_mask",
        "heatmap",
        "params",
        "param_mask",
        "task_id",
        "height",
        "width",
        "affine",
        "affine_inv",
        "clean_eval_ready",
        "num_landmarks",
        "direction_features",
        "sample_weight",
        "roi_anchor_landmarks",
        "roi_anchor_valid_mask",
    }
    for key in batch[0]:
        if key in tensor_keys:
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out


def _sample_weight(row: pd.Series) -> float:
    value = row.get("pseudo_weight", 1.0)
    if pd.isna(value):
        return 1.0
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 1.0


def _manifest_image_sha256(row: pd.Series) -> str | None:
    value = row.get("checksum", "")
    if pd.isna(value) or str(value).strip() == "":
        return None
    checksum = str(value).strip().lower()
    if _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError("manifest image checksum must be a lowercase SHA-256")
    return checksum
