"""Offline DINOv3 features with the existing geometry-aware landmark heads.

The original backbone and deployment factories are intentionally not changed.
Every DINO construction requires the locally verified pretrained artifact.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from clingeo.models.clingeo_usbiometry import ClinGeoUSBiometry


ENCODER_NAME = "vit_base_patch16_dinov3.lvd1689m"
ENCODER_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
ENCODER_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_encoder(weight_dir: str | Path, input_size: int) -> tuple[nn.Module, dict]:
    """Load the pinned timm conversion strictly, without any network access."""
    import timm
    from safetensors.torch import load_file

    root = Path(weight_dir)
    provenance = json.loads((root / "provenance.json").read_text())
    expected = {
        "repo_id": f"timm/{ENCODER_NAME}",
        "model_name": ENCODER_NAME,
        "revision": ENCODER_REVISION,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"pretrained provenance mismatch: {key}")
    weights = root / "model.safetensors"
    if provenance["files"][weights.name]["sha256"] != ENCODER_SHA256:
        raise ValueError("pretrained provenance checksum mismatch")
    if sha256_file(weights) != ENCODER_SHA256:
        raise ValueError("pretrained weight checksum mismatch")
    if ENCODER_NAME not in timm.list_models("*dinov3*", pretrained=True):
        raise RuntimeError("DINOv3 requires the isolated timm==1.0.24 runtime")
    encoder = timm.create_model(
        ENCODER_NAME, pretrained=False, img_size=input_size, num_classes=0
    )
    encoder.load_state_dict(load_file(str(weights)), strict=True)
    return encoder, provenance


class LoRALinear(nn.Module):
    """Zero-initialized residual on a frozen QKV projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha) / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.scale * self.lora_b(self.lora_a(value))


class DinoFeaturePyramid(nn.Module):
    """Shared features from four depths, projected to strides 4/8/16/32."""

    def __init__(self, encoder: nn.Module, lora_rank: int = 0):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.out_channels = (96, 192, 384, 768)
        self.lora_rank = int(lora_rank)
        if self.lora_rank < 0:
            raise ValueError("LoRA rank cannot be negative")
        if self.lora_rank:
            for block in self.encoder.blocks:
                if not isinstance(block.attn.qkv, nn.Linear):
                    raise TypeError("expected a fused linear QKV projection")
                block.attn.qkv = LoRALinear(block.attn.qkv, self.lora_rank, 2 * self.lora_rank)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(encoder.embed_dim, channels, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            )
            for channels in self.out_channels
        )
        depth = len(self.encoder.blocks)
        if depth < 4 or depth % 4:
            raise ValueError("encoder depth must be divisible by four")
        self.indices = [depth * (i + 1) // 4 - 1 for i in range(4)]

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the original pretrained stochastic layers disabled, including
        # when gradients flow through trainable LoRA residuals.
        self.encoder.eval()
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        height, width = image.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError("foundation inputs must be divisible by 32")
        context = nullcontext() if self.lora_rank and torch.is_grad_enabled() else torch.no_grad()
        with context:
            features = self.encoder.forward_intermediates(
                image, indices=self.indices, norm=True, output_fmt="NCHW", intermediates_only=True
            )
        if len(features) != 4:
            raise ValueError("encoder must return exactly four feature maps")
        return [
            F.interpolate(
                projection(feature), size=(height // stride, width // stride),
                mode="bilinear", align_corners=False,
            )
            for projection, feature, stride in zip(
                self.projections, features, (4, 8, 16, 32), strict=True
            )
        ]


class FoundationBiometry(ClinGeoUSBiometry):
    """Matched landmark heads for the CNN control, frozen DINO, and DINO LoRA."""

    def __init__(self, schema, config: dict):
        super().__init__(
            num_tasks=len(schema.tasks), max_landmarks=schema.max_landmarks,
            max_params=schema.max_params, input_size=int(config["input_size"]),
            backbone="convnextv2_base", decoder_channels=192, task_dim=128,
        )
        self.candidate = str(config["candidate"])
        self.pretrained_provenance = None
        if self.candidate in {"dino_frozen", "dino_lora"}:
            encoder, self.pretrained_provenance = verified_encoder(
                config["weight_dir"], self.input_size
            )
            rank = int(config.get("lora_rank", 8)) if self.candidate == "dino_lora" else 0
            if self.candidate == "dino_lora" and rank <= 0:
                raise ValueError("dino_lora requires a positive LoRA rank")
            self.backbone = DinoFeaturePyramid(encoder, rank)
        elif self.candidate != "cnn_control":
            raise ValueError(f"unknown candidate: {self.candidate}")
        self.param_head.requires_grad_(False)  # Clinical formula learning is not active.
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image: torch.Tensor, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        # All three candidates use the same grayscale representation and
        # normalization. The old deployed preprocessing is a separate baseline.
        image = (image[:, :1].expand(-1, 3, -1, -1) - self.input_mean) / self.input_std
        return super().forward(image, task_id)

    def research_state_dict(self) -> dict[str, torch.Tensor]:
        """Store learned weights; pin the frozen encoder by its artifact hash."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.encoder.") or ".lora_" in key
        }

    def load_research_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.research_state_dict())
        if set(state) != expected:
            raise ValueError("research checkpoint keys do not match this architecture")
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or any(
            not key.startswith("backbone.encoder.") for key in incompatible.missing_keys
        ):
            raise ValueError("incompatible research checkpoint")
