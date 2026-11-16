from __future__ import annotations

import torch
from torch import nn

SUPPORTED_BACKBONE_TOKENS = ("convnext", "resnet", "swin", "efficientnet")
UNSUPPORTED_FOUNDATION_BACKBONE_TOKENS = (
    "dinov2",
    "dino",
    "usfm",
    "medsam",
    "sam2",
    "segment_anything",
    "efficientnet",
)


class UnsupportedFoundationBackboneError(RuntimeError):
    """Raised when a foundation-backbone scaffold is requested before an adapter exists."""


def _matches_any_token(name: str, tokens: tuple[str, ...]) -> bool:
    return any(token in name for token in tokens)


class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.empty((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
            x = x * mask / keep
        return residual + x


class ConvNeXtV2Backbone(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.out_channels = dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            blocks = [ConvNeXtV2Block(dims[i], drop_path=dp_rates[cur + j]) for j in range(depth)]
            self.stages.append(nn.Sequential(*blocks))
            cur += depth

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
            features.append(x)
        return features


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
    ):
        super().__init__()
        self.out_channels = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8 if dims[0] >= 8 else 1, dims[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8 if dims[0] >= 8 else 1, dims[0]),
            nn.SiLU(inplace=True),
        )
        stages = []
        in_ch = dims[0]
        for i, (out_ch, depth) in enumerate(zip(dims, depths)):
            blocks = []
            if i == 0:
                blocks.append(ResNetBasicBlock(in_ch, out_ch, stride=1))
            else:
                blocks.append(ResNetBasicBlock(in_ch, out_ch, stride=2))
            for _ in range(depth - 1):
                blocks.append(ResNetBasicBlock(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        hidden = in_ch * expansion
        self.use_residual = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(8 if hidden >= 8 else 1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(8 if hidden >= 8 else 1, hidden),
            nn.SiLU(inplace=True),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.act(out)


class OfficialBaselineEfficientNetFPNBackbone(nn.Module):
    """Pure-PyTorch EfficientNet-style adapter for local smoke checks only."""

    def __init__(self, in_chans: int = 3, lightweight: bool = False):
        super().__init__()
        dims = (64, 128, 256, 512)
        depths = (1, 1, 2, 1) if lightweight else (2, 3, 5, 3)
        self.out_channels = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        stages = []
        in_ch = 32
        for out_ch, depth in zip(dims, depths):
            blocks = [MBConvBlock(in_ch, out_ch, stride=2)]
            blocks.extend(MBConvBlock(out_ch, out_ch, stride=1) for _ in range(depth - 1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def build_backbone(name: str, in_chans: int = 3, lightweight: bool = False) -> nn.Module:
    normalized = name.lower()
    if normalized == "official_baseline_efficientnet_b4_fpn":
        return OfficialBaselineEfficientNetFPNBackbone(in_chans=in_chans, lightweight=lightweight)
    if _matches_any_token(normalized, UNSUPPORTED_FOUNDATION_BACKBONE_TOKENS):
        raise UnsupportedFoundationBackboneError(
            f"Foundation backbone '{name}' is a config scaffold only. "
            "Install the required runtime package, record compliant local weights/provenance, "
            "and implement a real backbone adapter before instantiating it."
        )
    if lightweight or "tiny" in normalized or "smoke" in normalized:
        return ConvNeXtV2Backbone(in_chans=in_chans, dims=(32, 64, 128, 256), depths=(1, 1, 2, 1), drop_path_rate=0.02)
    if "resnet" in normalized:
        if "small" in normalized:
            return ResNetBackbone(in_chans=in_chans, dims=(64, 128, 256, 512), depths=(2, 2, 4, 2))
        return ResNetBackbone(in_chans=in_chans, dims=(96, 192, 384, 768), depths=(2, 2, 6, 2))
    if "base" in normalized or "convnext" in normalized or "swin" in normalized:
        # Pure PyTorch ConvNeXtV2-style fallback. If timm/pretrained weights are available,
        # this can be replaced behind the same interface without changing the train loop.
        return ConvNeXtV2Backbone(in_chans=in_chans, dims=(96, 192, 384, 768), depths=(2, 2, 8, 2), drop_path_rate=0.1)
    return ConvNeXtV2Backbone(in_chans=in_chans)
