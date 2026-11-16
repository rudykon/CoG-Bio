from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clingeo.data.image_io import read_grayscale, sha256_file


META_COLUMNS = {"image_path", "height", "width", "task_name", "num_classes", "task_id"}


def parse_xy(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError(f"Expected xy pair, got {value}")
        return [float(value[0]), float(value[1])]
    if pd.isna(value):
        return [float("nan"), float("nan")]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(f"Could not parse xy value {text!r}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"Expected xy pair, got {text!r}")
    return [float(parsed[0]), float(parsed[1])]


def parse_json_array(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    if isinstance(value, list):
        return np.asarray(value, dtype=dtype)
    if pd.isna(value):
        return np.zeros((0,), dtype=dtype)
    return np.asarray(json.loads(str(value)), dtype=dtype)


def infer_xy_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_xy") or c.startswith("point_") and c.endswith("_xy")]


def infer_target_columns(df: pd.DataFrame, xy_columns: list[str]) -> list[str]:
    excluded = META_COLUMNS.union(xy_columns)
    out = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def resolve_image_path(data_root: str | Path, split: str, domain: str, image_path: str) -> Path:
    root = Path(data_root)
    rel = Path(str(image_path))
    candidates = []
    if rel.is_absolute():
        candidates.append(rel)
    candidates.append(root / image_path)
    candidates.append(root / split / "labeled" / "images" / image_path)
    candidates.append(root / split / "images" / image_path)
    candidates.append(root / "train" / "labeled" / "images" / image_path)
    candidates.append(root / "train" / "labeled" / "images" / domain / rel.name)
    candidates.append(root / "validation" / "images" / image_path)
    candidates.append(root / "validation" / "images" / domain / rel.name)
    # Current HC CSV stores image_path as images/<file>, while files live under HC/<file>.
    if len(rel.parts) > 1:
        candidates.append(root / "train" / "labeled" / "images" / domain / rel.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def pseudo_group_id(source_folder: str, case_id: str, height: int | None = None, width: int | None = None) -> str:
    bucket = f"{height or 'h'}x{width or 'w'}"
    prefix = str(case_id).split("_frame")[0].split("_of_")[0]
    raw = f"{source_folder}|{bucket}|{prefix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_labeled_manifest(data_root: str | Path, checksum: bool = False) -> pd.DataFrame:
    data_root = Path(data_root)
    ann_root = data_root / "train" / "labeled" / "annotations"
    if not ann_root.exists():
        raise FileNotFoundError(f"Missing annotation directory: {ann_root}")
    rows: list[dict[str, Any]] = []
    for ann_path in sorted(ann_root.glob("*.csv")):
        domain = ann_path.stem
        df = pd.read_csv(ann_path)
        xy_columns = infer_xy_columns(df)
        if not xy_columns:
            raise ValueError(f"No xy columns found in {ann_path}")
        target_columns = infer_target_columns(df, xy_columns)
        for idx, row in df.iterrows():
            case_rel = str(row["image_path"])
            image_path = resolve_image_path(data_root, "train", domain, case_rel)
            case_id = Path(case_rel).stem
            height = int(row.get("height", 0) or 0)
            width = int(row.get("width", 0) or 0)
            if (height <= 0 or width <= 0) and image_path.exists():
                img = read_grayscale(image_path)
                height, width = int(img.shape[0]), int(img.shape[1])
            landmarks = [parse_xy(row[col]) for col in xy_columns]
            valid_mask = [int(np.isfinite(x).all()) for x in np.asarray(landmarks, dtype=np.float32)]
            clinical = {col: float(row[col]) for col in target_columns if pd.notna(row[col])}
            source_folder = str(Path("train/labeled/images") / domain)
            rows.append(
                {
                    "split": "train_labeled",
                    "domain": domain,
                    "case_id": case_id,
                    "image_path": str(image_path.relative_to(data_root) if image_path.is_relative_to(data_root) else image_path),
                    "source_folder": source_folder,
                    "annotation_path": str(ann_path.relative_to(data_root)),
                    "row_index": int(idx),
                    "height": height,
                    "width": width,
                    "num_landmarks": len(landmarks),
                    "landmarks": json.dumps(landmarks, separators=(",", ":")),
                    "valid_mask": json.dumps(valid_mask, separators=(",", ":")),
                    "xy_columns": "|".join(xy_columns),
                    "target_columns": "|".join(target_columns),
                    "clinical_params": json.dumps(clinical, separators=(",", ":")),
                    "pseudo_group_id": pseudo_group_id(source_folder, case_id, height, width),
                    "checksum": sha256_file(image_path) if checksum and image_path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def build_validation_manifest(data_root: str | Path, checksum: bool = False) -> pd.DataFrame:
    data_root = Path(data_root)
    existing = data_root / "manifests" / "validation.csv"
    rows: list[dict[str, Any]] = []
    if existing.exists():
        df = pd.read_csv(existing)
        iterator = df.iterrows()
    else:
        image_root = data_root / "validation" / "images"
        records = []
        for path in sorted(image_root.glob("*/*")):
            if path.is_file():
                records.append({"domain": path.parent.name, "case_id": path.stem, "image_path": str(path.relative_to(data_root))})
        iterator = pd.DataFrame(records).iterrows()
    for _idx, row in iterator:
        domain = str(row["domain"])
        image_path = resolve_image_path(data_root, "validation", domain, str(row["image_path"]))
        height = int(row.get("height", 0) or 0)
        width = int(row.get("width", 0) or 0)
        if (height <= 0 or width <= 0) and image_path.exists():
            img = read_grayscale(image_path)
            height, width = int(img.shape[0]), int(img.shape[1])
        rows.append(
            {
                "split": "validation",
                "domain": domain,
                "case_id": str(row["case_id"]),
                "image_path": str(image_path.relative_to(data_root) if image_path.is_relative_to(data_root) else image_path),
                "source_folder": str(Path("validation/images") / domain),
                "height": height,
                "width": width,
                "num_landmarks": "",
                "landmarks": "",
                "valid_mask": "",
                "xy_columns": "",
                "target_columns": "",
                "clinical_params": "{}",
                "pseudo_group_id": pseudo_group_id(str(Path("validation/images") / domain), str(row["case_id"]), height, width),
                "checksum": sha256_file(image_path) if checksum and image_path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def save_manifest(df: pd.DataFrame, output: str | Path, allow_csv_fallback: bool = True) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        try:
            df.to_parquet(output, index=False)
            return output
        except Exception as exc:
            if not allow_csv_fallback:
                raise RuntimeError("Writing parquet requires pyarrow or fastparquet") from exc
            fallback = output.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            marker = output.with_suffix(".format.txt")
            marker.write_text(f"Parquet unavailable; wrote CSV fallback: {fallback.name}\nError: {exc}\n", encoding="utf-8")
            return fallback
    df.to_csv(output, index=False)
    return output


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback)
            raise RuntimeError(f"Could not read parquet manifest {path}; install pyarrow or pass CSV") from exc
    return pd.read_csv(path, low_memory=False)
