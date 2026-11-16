from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import math
import torch


class LearnedEMA:
    """Average trainable parameters only; never copy the large frozen encoder."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)

    @contextmanager
    def applied(self, model):
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)


def schedule_factor(step, total, warmup, floor):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    phase = min(1., max(0., (step - warmup) / max(total - warmup - 1, 1)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * phase))


def sampled_exposures(frame, count, seed, exponent):
    counts = frame.domain.value_counts()
    weights = torch.tensor([float(counts[task]) ** -exponent for task in frame.domain], dtype=torch.double)
    indices = torch.multinomial(weights, count, replacement=True, generator=torch.Generator().manual_seed(seed)).tolist()
    return [(index, exposure) for exposure, index in enumerate(indices)]


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
