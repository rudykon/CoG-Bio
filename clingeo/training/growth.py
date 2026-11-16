"""Grow a LoRA adapter while retaining its existing learned function."""
from __future__ import annotations
import torch


def grow_lora_state(source, template):
    if set(source) != set(template): raise ValueError("adapter state keys differ")
    result = {}
    for key, target in template.items():
        old = source[key]
        if old.shape == target.shape:
            result[key] = old.detach().clone()
            continue
        value = target.detach().clone()
        if ".lora_a." in key and old.ndim == 2 and old.shape[1] == value.shape[1] and old.shape[0] < value.shape[0]:
            value[:old.shape[0]] = old.to(value)
        elif ".lora_b." in key and old.ndim == 2 and old.shape[0] == value.shape[0] and old.shape[1] < value.shape[1]:
            value.zero_(); value[:,:old.shape[1]] = old.to(value)
        else:
            raise ValueError(f"unsupported shape change for {key}")
        result[key] = value
    # DinoFeaturePyramid uses alpha = 2 * rank, hence scale is always 2.
    # Added B columns are zero; new A rows start from their fresh initialization.
    return result
