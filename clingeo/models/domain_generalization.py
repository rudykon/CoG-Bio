from __future__ import annotations

import random

import torch
from torch import nn

from clingeo.config.validation import normalize_domain_generalization_config


class MixStyle(nn.Module):
    """Feature-statistics mixing for domain generalization."""

    def __init__(self, p: float = 0.3, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.shape[0] < 2:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        normalized = (x - mu) / sig
        perm = torch.randperm(x.shape[0], device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return normalized * sig_mix + mu_mix


class DSU(nn.Module):
    """Domain statistics uncertainty perturbation."""

    def __init__(self, p: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.shape[0] < 2:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu_std = mu.std(dim=0, keepdim=True, unbiased=False)
        sig_std = sig.std(dim=0, keepdim=True, unbiased=False)
        eps_mu = torch.randn_like(mu) * mu_std
        eps_sig = torch.randn_like(sig) * sig_std
        return (x - mu) / sig * (sig + eps_sig).clamp_min(self.eps) + (mu + eps_mu)


class FeatureDomainGeneralizer(nn.Module):
    def __init__(self, cfg: dict | None = None, *, strict_clean: bool = False):
        super().__init__()
        cfg = normalize_domain_generalization_config(cfg or {}, strict_clean=strict_clean)
        mix_cfg = dict(cfg.get("mixstyle", {}) or {})
        dsu_cfg = dict(cfg.get("dsu", {}) or {})
        self.mixstyle_layers = set(mix_cfg.get("layers", []) or [])
        self.dsu_layers = set(dsu_cfg.get("layers", []) or [])
        self.layers = set(self.mixstyle_layers) | set(self.dsu_layers)
        self.mixstyle = MixStyle(p=float(mix_cfg.get("p", 0.0)), alpha=float(mix_cfg.get("alpha", 0.1))) if mix_cfg.get("enabled", False) else None
        self.dsu = DSU(p=float(dsu_cfg.get("p", 0.0))) if dsu_cfg.get("enabled", False) else None

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.mixstyle is None and self.dsu is None:
            return features
        out = []
        for idx, feature in enumerate(features):
            if self.mixstyle is not None and idx in self.mixstyle_layers:
                feature = self.mixstyle(feature)
            if self.dsu is not None and idx in self.dsu_layers:
                feature = self.dsu(feature)
            out.append(feature)
        return out
