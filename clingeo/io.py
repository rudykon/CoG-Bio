"""Portable input/output validation for image manifests and landmark records."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_rows(path, schema, root):
    rows = read_rows(path)
    if not rows:
        raise ValueError("empty image manifest")
    seen = set()
    for row in rows:
        identifier = row["sample_id"]
        if not identifier or identifier in seen:
            raise ValueError("empty or duplicate sample_id")
        seen.add(identifier)
        if row["task_id"] not in schema.tasks:
            raise ValueError("use canonical task_id values from landmark_schema.yaml")
        if int(row["num_landmarks"]) != schema.tasks[row["task_id"]].num_landmarks:
            raise ValueError("landmark count differs from task schema")
        if min(int(row["width"]), int(row["height"])) <= 0:
            raise ValueError("positive image dimensions required")
        rel = Path(row["image_relpath"])
        if rel.is_absolute() or not (Path(root) / rel).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("image_relpath must stay within the supplied image root")
        row["checksum"] = row.get("checksum") or None
    return rows


def validate_predictions(records, rows, schema):
    expected = {r["sample_id"]: r for r in rows}
    if len(expected) != len(rows) or len(records) != len(rows):
        raise ValueError("prediction inventory is incomplete or duplicated")
    seen = set()
    for record in records:
        key = record["image_path"]
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate prediction identifier")
        seen.add(key)
        row = expected[key]
        if record["task_id"] != row["task_id"]:
            raise ValueError("prediction task differs from image manifest")
        points = np.asarray(record["predicted_points_pixels"], dtype=np.float64)
        n = schema.tasks[row["task_id"]].num_landmarks
        if points.shape != (2 * n,) or not np.isfinite(points).all():
            raise ValueError("finite flattened coordinates with the task landmark count required")
    return records
