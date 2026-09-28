
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_decoder_channels: Sequence[int] = (1024, 512, 256, 64)
    dropout_p: float = 0.20
    negative_slope: float = 0.01
    pretrained_encoder: bool = False
    attention: bool = False


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: Optional[int] = None,
                 negative_slope: float = 0.01):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Diagram-faithful gate based on the user's AG-x figure:
      g -> Wg ---+
                 add -> ReLU -> 1x1 conv -> Sigmoid -> resample -> multiply with x
      x -> Wx ---+
    """
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        self.w_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.w_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_proj = self.w_g(g)
        x_proj = self.w_x(x)
        if g_proj.shape[-2:] != x_proj.shape[-2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode='bilinear', align_corners=False)
        attn = self.relu(g_proj + x_proj)
        attn = self.psi(attn)
        attn = self.sigmoid(attn)
        if attn.shape[-2:] != x.shape[-2:]:
            attn = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * attn


class ResNet50Encoder(nn.Module):
    """
    Encoder aligned to the diagram and paper description:
    - input stem: 7x7 conv / 64 + BN + ReLU
    - maxpool
    - ResNet50 stages with Bottleneck block counts 3,4,6,3
    Returns skips f1..f5 as in the figure.
    """
    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = 'IMAGENET1K_V2' if pretrained else None
        backbone = resnet50(weights=weights)
        if in_channels != 3:
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                       stride=old.stride, padding=old.padding, bias=False)
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)          # 64, 120x120 if input 240x240
        x = self.maxpool(f1)
        f2 = self.layer1(x)        # 256, 60x60
        f3 = self.layer2(f2)       # 512, 30x30
        f4 = self.layer3(f3)       # 1024, 15x15
        f5 = self.layer4(f4)       # 2048, 8x8 approx
        return [f1, f2, f3, f4, f5]


class DecoderBlock(nn.Module):
    """
    Decoder block following the figure's repeated pattern:
    3x3 Conv -> 1x1 Conv -> Dropout -> 3x3 Conv -> Dropout -> ReLU
    Upstream feature is resized before concat; optional attention is applied to the skip.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dropout_p: float = 0.20, negative_slope: float = 0.01,
                 use_attention: bool = False):
        super().__init__()
        self.attn = AttentionGate(in_channels, skip_channels) if use_attention else None
        merged = in_channels + skip_channels
        self.conv3x3_a = nn.Conv2d(merged, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, bias=False)
        self.drop1 = nn.Dropout2d(dropout_p)
        self.conv3x3_b = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop2 = nn.Dropout2d(dropout_p)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        if self.attn is not None:
            skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3x3_a(x)
        x = self.conv1x1(x)
        x = self.drop1(x)
        x = self.conv3x3_b(x)
        x = self.drop2(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UResNet50Base(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResNet50Encoder(in_channels=cfg.in_channels, pretrained=cfg.pretrained_encoder)
        d1, d2, d3, d4 = cfg.base_decoder_channels
        self.dec1 = DecoderBlock(2048, 1024, d1, cfg.dropout_p, cfg.negative_slope, cfg.attention)
        self.dec2 = DecoderBlock(d1, 512, d2, cfg.dropout_p, cfg.negative_slope, cfg.attention)
        self.dec3 = DecoderBlock(d2, 256, d3, cfg.dropout_p, cfg.negative_slope, cfg.attention)
        self.dec4 = DecoderBlock(d3, 64, d4, cfg.dropout_p, cfg.negative_slope, cfg.attention)
        self.head = nn.Sequential(
            nn.Conv2d(d4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=cfg.negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=cfg.negative_slope, inplace=True),
            nn.Conv2d(64, cfg.out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4, f5 = self.encoder(x)
        x = self.dec1(f5, f4)
        x = self.dec2(x, f3)
        x = self.dec3(x, f2)
        x = self.dec4(x, f1)
        x = F.interpolate(x, size=(240, 240) if x.shape[-2:] != (240, 240) and x.shape[-2] <= 240 else x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)


class UResNet50(UResNet50Base):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 dropout_p: float = 0.20, negative_slope: float = 0.01,
                 pretrained_encoder: bool = False):
        super().__init__(ModelConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
            negative_slope=negative_slope,
            pretrained_encoder=pretrained_encoder,
            attention=False,
        ))


class AGUResNet50(UResNet50Base):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 dropout_p: float = 0.20, negative_slope: float = 0.01,
                 pretrained_encoder: bool = False):
        super().__init__(ModelConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
            negative_slope=negative_slope,
            pretrained_encoder=pretrained_encoder,
            attention=True,
        ))


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, bce_weight: float = 1.0, dice_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target)
    dloss = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dloss


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> torch.Tensor:
    pred = (pred >= threshold).float().view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denom + smooth)).mean()


@torch.no_grad()
def smoke_test(device: Optional[str] = None) -> List[Tuple[str, Tuple[int, ...]]]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 1, 240, 240, device=device)
    outputs = []
    for cls in (UResNet50, AGUResNet50):
        model = cls().to(device)
        y = model(x)
        assert y.shape == (2, 1, 240, 240), f'{cls.__name__} produced {tuple(y.shape)}'
        loss = bce_dice_loss(y, torch.rand_like(y))
        assert torch.isfinite(loss), f'{cls.__name__} loss is not finite'
        outputs.append((cls.__name__, tuple(y.shape)))
    return outputs


if __name__ == '__main__':
    print(smoke_test())
