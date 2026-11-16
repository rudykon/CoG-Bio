from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from clingeo.data.image_io import read_grayscale
from clingeo.data.transforms import letterbox_image_tensor,make_three_channel


class OfficialImages(Dataset):
    def __init__(self, release, rows, schema, input_size):
        self.release, self.rows = Path(release), rows
        self.task_index, self.input_size = schema.task_index(), input_size

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_grayscale(self.release / row["image_relpath"], cache_dir=None, expected_sha256=row["checksum"])
        height, width = image.shape
        tensor, affine = letterbox_image_tensor(make_three_channel(image, ch2="global_he"), self.input_size)
        return {"image": tensor, "task_id": self.task_index[row["task_id"]],
                "inverse": torch.tensor(affine.inverse, dtype=torch.float32), "index": index,
                "height": height, "width": width}


def original_coordinates(coords, inverse, width, height, count):
    points = np.asarray(coords, dtype=np.float32)[:count]
    if points.shape != (count, 2) or not np.isfinite(points).all():
        raise ValueError("invalid predicted coordinate shape or value")
    homogeneous = np.concatenate([points, np.ones((count, 1), dtype=np.float32)], axis=1)
    transformed = (homogeneous @ np.asarray(inverse, dtype=np.float32).T)[:, :2]
    transformed[:, 0] = transformed[:, 0].clip(0, width - 1)
    transformed[:, 1] = transformed[:, 1].clip(0, height - 1)
    return np.round(transformed.astype(np.float64), 4).reshape(-1).tolist()
