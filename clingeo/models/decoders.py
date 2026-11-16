from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TaskFiLM(nn.Module):
    def __init__(self, channels: int, task_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(task_dim, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels * 2))

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(task_emb).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FPNDecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 192, task_dim: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                )
                for _ in in_channels
            ]
        )
        self.film = TaskFiLM(out_channels, task_dim)

    def forward(self, features: list[torch.Tensor], task_emb: torch.Tensor) -> torch.Tensor:
        feats = [lat(feat) for lat, feat in zip(self.lateral, features)]
        x = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + feats[i]
            x = self.smooth[i](x)
        return self.film(x, task_emb)
