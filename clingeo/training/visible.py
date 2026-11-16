"""Deterministic augmentation that keeps supervised landmarks in the image."""
from __future__ import annotations
import torch
from clingeo.data.dataset import BiometryDataset
from clingeo.training.exposure import ExposureDataset


def visible_mask(batch):
    height,width = batch["image"].shape[-2:]
    points = batch["landmarks"]
    return (torch.isfinite(points).all(-1) & (points[:,0]>=0) & (points[:,0]<=width-1)
            & (points[:,1]>=0) & (points[:,1]<=height-1))


def all_supervised_visible(batch):
    valid = batch["valid_mask"].bool()
    return bool(valid.any()) and bool(visible_mask(batch)[valid].all())


class VisibleExposureDataset(ExposureDataset):
    """Retry the same image with new keyed transforms, then use its clean view."""
    def __init__(self,dataset,seed):
        super().__init__(dataset,seed)
        self.clean = BiometryDataset(dataset.df,dataset.data_root,dataset.schema,
            input_size=dataset.input_size,heatmap_stride=dataset.heatmap_stride,
            training=False,cache_dir=dataset.cache_dir,ultrasound_aug=dataset.ultrasound_aug,
            preprocess=dataset.preprocess)

    def __getitem__(self,key):
        index,exposure = key
        for attempt in range(5):
            batch = super().__getitem__((index,exposure+attempt*1_000_000_000))
            if all_supervised_visible(batch): return batch
        batch = self.clean[int(index)]
        # Retain original annotation files; never supervise a point outside the
        # actual model input even if an original border annotation is invalid.
        mask = visible_mask(batch).to(batch["valid_mask"].dtype)
        batch["valid_mask"] = batch["valid_mask"]*mask
        batch["heatmap"] = batch["heatmap"]*mask[:,None,None]
        if not all_supervised_visible(batch):
            raise ValueError("no visible supervised point in clean fallback")
        return batch
