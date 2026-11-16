from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_manifest(tmp_path: Path, n: int = 6) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        arr = np.zeros((64, 80), dtype=np.uint8)
        arr[16 + i : 24 + i, 20:30] = 180
        path = image_dir / f"case_{i}.npy"
        np.save(path, arr)
        landmarks = [[20.0, 16.0], [30.0, 24.0]]
        rows.append(
            {
                "split": "train_labeled",
                "domain": "FUGC",
                "case_id": f"case_{i}",
                "image_path": str(path.relative_to(tmp_path)),
                "source_folder": "synthetic",
                "annotation_path": "",
                "row_index": i,
                "height": 64,
                "width": 80,
                "num_landmarks": 2,
                "landmarks": json.dumps(landmarks),
                "valid_mask": json.dumps([1, 1]),
                "xy_columns": "point_1_xy|point_2_xy",
                "target_columns": "",
                "clinical_params": "{}",
                "pseudo_group_id": f"group_{i}",
                "checksum": "",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
