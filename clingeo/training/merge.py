"""Merge two adapted DINO checkpoints into one network without changing the base."""
from __future__ import annotations
import torch


def merge_lora_states(first,second):
    if set(first)!=set(second):raise ValueError("checkpoint parameter keys differ")
    result={}
    for key,a in first.items():
        b=second[key]
        if key.endswith("lora_a.weight"):
            if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError("incompatible LoRA A")
            result[key]=torch.cat([a,b],dim=0)
        elif key.endswith("lora_b.weight"):
            if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[0]:raise ValueError("incompatible LoRA B")
            result[key]=torch.cat([a*.5,b*.5],dim=1)
        else:
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError("incompatible shared parameters")
            if torch.equal(a,b):result[key]=a.clone()
            elif a.is_floating_point():result[key]=a*.5+b*.5
            else:raise ValueError("different non-floating buffers")
    return result


def merged_model_config(first,second):
    a={k:v for k,v in first.items() if k!="lora_rank"}
    b={k:v for k,v in second.items() if k!="lora_rank"}
    if a!=b:raise ValueError("merged models must have the same input and head architecture")
    return {**first,"lora_rank":first["lora_rank"]+second["lora_rank"]}
