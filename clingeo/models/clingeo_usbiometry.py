from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from clingeo.models.backbones import build_backbone
from clingeo.models.decoders import FPNDecoder
from clingeo.models.domain_generalization import FeatureDomainGeneralizer


class ClinGeoUSBiometry(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        max_landmarks: int,
        max_params: int,
        input_size: int = 768,
        backbone: str = "convnextv2_base",
        in_channels: int = 3,
        heatmap_stride: int = 4,
        decoder_channels: int = 192,
        task_dim: int = 128,
        lightweight: bool = False,
        domain_generalization: dict | None = None,
        domain_generalization_strict_clean: bool = False,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.max_landmarks = int(max_landmarks)
        self.max_params = int(max_params)
        self.input_size = int(input_size)
        self.heatmap_stride = int(heatmap_stride)
        self.task_embedding = nn.Embedding(num_tasks, task_dim)
        self.backbone = build_backbone(backbone, in_chans=in_channels, lightweight=lightweight)
        self.domain_generalizer = FeatureDomainGeneralizer(
            domain_generalization,
            strict_clean=domain_generalization_strict_clean,
        )
        self.decoder = FPNDecoder(tuple(self.backbone.out_channels), out_channels=decoder_channels, task_dim=task_dim)
        if self.heatmap_stride == 2:
            norm_groups = 8 if decoder_channels % 8 == 0 else 1
            self.detail = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                nn.GroupNorm(norm_groups, decoder_channels),
                nn.GELU(),
            )
        elif self.heatmap_stride == 4:
            self.detail = nn.Identity()
        else:
            raise ValueError(f"Unsupported heatmap_stride: {self.heatmap_stride}")
        self.heatmap_head = nn.Conv2d(decoder_channels, max_landmarks, kernel_size=1)
        self.offset_head = nn.Conv2d(decoder_channels, max_landmarks * 2, kernel_size=1)
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, decoder_channels),
            nn.GELU(),
            nn.Linear(decoder_channels, max_landmarks),
        )
        self.coord_query_head = nn.Sequential(
            nn.Linear(decoder_channels + task_dim, decoder_channels),
            nn.GELU(),
            nn.Linear(decoder_channels, max_landmarks * 2),
        )
        self.param_head = nn.Sequential(
            nn.Linear(decoder_channels + task_dim, decoder_channels),
            nn.GELU(),
            nn.Linear(decoder_channels, max(max_params, 1)),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(task_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        task_emb = self.task_embedding(task_id)
        features = self.backbone(image)
        features = self.domain_generalizer(features)
        dense = self.decoder(features, task_emb)
        dense = self.detail(dense)
        heatmap_logits = self.heatmap_head(dense)
        offset = self.offset_head(dense)
        pooled = F.adaptive_avg_pool2d(dense, 1).flatten(1)
        query_in = torch.cat([pooled, task_emb], dim=1)
        coord_query = torch.sigmoid(self.coord_query_head(query_in)).view(image.shape[0], self.max_landmarks, 2)
        coord_query = coord_query * float(self.input_size - 1)
        log_sigma = self.uncertainty_head(dense).view(image.shape[0], self.max_landmarks, 1).clamp(-5.0, 5.0)
        params = self.param_head(query_in)[:, : self.max_params] if self.max_params > 0 else image.new_zeros((image.shape[0], 0))
        coord_heatmap, heatmap_prob, peak, entropy = decode_heatmap_with_offset(
            heatmap_logits,
            offset,
            stride=self.heatmap_stride,
            input_size=self.input_size,
            max_landmarks=self.max_landmarks,
        )
        gate_features = torch.cat(
            [
                peak.unsqueeze(-1),
                entropy.unsqueeze(-1),
                log_sigma.exp(),
            ],
            dim=-1,
        )
        task_expand = task_emb.unsqueeze(1).expand(-1, self.max_landmarks, -1)
        gate = torch.sigmoid(self.gate_head(torch.cat([gate_features, task_expand], dim=-1)))
        coords = gate * coord_heatmap + (1.0 - gate) * coord_query
        return {
            "heatmap": heatmap_logits,
            "heatmap_prob": heatmap_prob,
            "offset": offset,
            "coord_heatmap": coord_heatmap,
            "coord_query": coord_query,
            "coords": coords,
            "log_sigma": log_sigma,
            "params": params,
            "gate": gate,
        }


def decode_heatmap_with_offset(
    heatmap_logits: torch.Tensor,
    offset: torch.Tensor,
    stride: int,
    input_size: int,
    max_landmarks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, k, h, w = heatmap_logits.shape
    prob = torch.softmax(heatmap_logits.view(b, k, -1), dim=-1).view(b, k, h, w)
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=heatmap_logits.dtype, device=heatmap_logits.device),
        torch.arange(w, dtype=heatmap_logits.dtype, device=heatmap_logits.device),
        indexing="ij",
    )
    x = (prob * xx.view(1, 1, h, w)).sum(dim=(2, 3))
    y = (prob * yy.view(1, 1, h, w)).sum(dim=(2, 3))
    off = offset.view(b, max_landmarks, 2, h, w)
    off_x = (prob * off[:, :, 0]).sum(dim=(2, 3))
    off_y = (prob * off[:, :, 1]).sum(dim=(2, 3))
    coords = torch.stack([(x + off_x) * stride, (y + off_y) * stride], dim=-1)
    coords = coords.clamp(0.0, float(input_size - 1))
    peak = prob.flatten(2).amax(dim=-1)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=(2, 3)) / float(h * w)
    return coords, prob, peak, entropy
