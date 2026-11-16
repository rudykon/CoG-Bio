from __future__ import annotations
from contextlib import contextmanager
import hashlib,random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from clingeo.data.dataset import BiometryDataset
from clingeo.model import FoundationBiometry

TASKS = ("A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur")


@contextmanager
def augmentation_rng(seed: int):
    """Sample augmentation independently of each model's internal random draws."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed % (2**32))
        random.seed(seed)
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class ExposureDataset(Dataset):
    def __init__(self, dataset: BiometryDataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key: tuple[int, int]):
        index, exposure = key
        digest = hashlib.sha256(f"{self.seed}:{exposure}".encode()).digest()
        with augmentation_rng(int.from_bytes(digest[:4], "little")):
            return self.dataset[int(index)]


def exposure_indices(frame: pd.DataFrame, count: int, seed: int, smoke: bool) -> list[int]:
    counts = frame.domain.value_counts()
    weights = torch.tensor([float(counts[task]) ** -0.5 for task in frame.domain], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.multinomial(weights, count, replacement=True, generator=generator).tolist()
    if smoke:
        # Exercise every available task explicitly, including the low-n tasks.
        first = [int(frame.index[frame.domain.eq(task)][0]) for task in TASKS if task in counts]
        if count < len(first):
            raise ValueError("smoke budget must cover every task")
        sampled[:len(first)] = first
    return sampled


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def frozen_encoder_digest(model: FoundationBiometry) -> str | None:
    encoder = getattr(model.backbone, "encoder", None)
    if encoder is None:
        return None
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        if ".lora_" in name:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
