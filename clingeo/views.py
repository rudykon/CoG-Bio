"""Fixed, label-free views of a single DINO checkpoint."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F


def undo_flip(points, width, pairs):
    restored = points.clone()
    restored[..., 0] = width - 1 - restored[..., 0]
    for a, b in pairs:
        restored[..., [a, b], :] = restored[..., [b, a], :]
    return restored


def predicted_crop_box(points, size, minimum_fraction=.8, context=1.25):
    if points.ndim != 2 or points.shape[1] != 2 or not torch.isfinite(points).all():
        raise ValueError("finite unpadded predicted landmarks required")
    low, high = points.min(0).values, points.max(0).values
    side = min(size, max(2, round(max(size * minimum_fraction,
                                    float((high-low).max()+1) * context))))
    center = (low+high)/2
    left, top = [min(size-side, max(0, round(float(v)-(side-1)/2))) for v in center]
    return left, top, side


def undo_crop(points, box, input_size):
    left, top, side = box
    return points * ((side-1)/(input_size-1)) + points.new_tensor([left, top])


class FixedViewMean(torch.nn.Module):
    def __init__(self, model, schema, view, minimum_fraction=.8, context=1.25):
        super().__init__()
        if view not in {"flip_mean", "crop_mean"}: raise ValueError("unknown view")
        self.model, self.input_size, self.view = model, model.input_size, view
        self.tasks = {i:schema.tasks[name] for name,i in schema.task_index().items()}
        self.minimum_fraction, self.context = minimum_fraction, context

    def forward(self, image, task_id):
        native = self.model(image, task_id)["coords"].float()
        if self.view == "flip_mean":
            indices = [i for i,t in enumerate(task_id.tolist()) if self.tasks[t].allow_hflip]
            if not indices: return {"coords": native}
            flipped = self.model(image[indices].flip(-1), task_id[indices])["coords"].float()
            output = native.clone()
            for j,i in enumerate(indices):
                mapped = undo_flip(flipped[j], self.input_size, self.tasks[int(task_id[i])].flip_pairs)
                output[i] = (native[i]+mapped)*.5
            return {"coords": output}
        boxes = [predicted_crop_box(native[i,:self.tasks[t].num_landmarks], self.input_size,
                                    self.minimum_fraction, self.context)
                 for i,t in enumerate(task_id.tolist())]
        crops = torch.cat([F.interpolate(image[i:i+1,:,top:top+side,left:left+side],
                                        size=(self.input_size,self.input_size), mode="bilinear",
                                        align_corners=True)
                           for i,(left,top,side) in enumerate(boxes)])
        refined = self.model(crops,task_id)["coords"].float()
        restored = torch.stack([undo_crop(refined[i],box,self.input_size) for i,box in enumerate(boxes)])
        return {"coords": (native+restored)*.5}


def average_records(sources, id_key="manifest_row_id"):
    """Equal view means in original image coordinates after each view is restored."""
    first = sources[0]
    mapped = [{r[id_key]:r for r in source} for source in sources]
    expected = {r[id_key] for r in first}
    if any(set(m)!=expected or len(m)!=len(first) for m in mapped):
        raise ValueError("view coverage mismatch")
    result = []
    for row in first:
        peers = [m[row[id_key]] for m in mapped]
        if any(p["task_id"]!=row["task_id"] for p in peers): raise ValueError("view task mismatch")
        values = np.stack([np.asarray(p["predicted_points_pixels"], dtype=np.float64) for p in peers])
        if not np.isfinite(values).all(): raise ValueError("nonfinite view")
        result.append({**row,"predicted_points_pixels":np.round(values.mean(0),4).tolist()})
    return result
