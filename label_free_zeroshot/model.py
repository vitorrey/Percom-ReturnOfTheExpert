"""
model.py
--------
1D-CNN encoder for accelerometer pseudo-label prediction.

Architecture
------------
Input  : (B, C, W)  where C = n_sensors * 3, W = window_frames
Encoder: 4 × (Conv1d → BatchNorm → ReLU) with progressive downsampling
Head   : AdaptiveAvgPool → FC → output (B, n_features)

The encoder produces a fixed-size embedding that can be reused for
downstream fine-tuning (replace only the head).
"""

import torch
import torch.nn as nn
from typing import Tuple


# -----------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU, with optional strided downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """
    Two-layer residual block — same as the pattern used in ssl-wearables.
    Both convolutions keep spatial size (stride=1, same padding).
    The skip connection uses a 1×1 conv if channel dimensions differ.
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


# -----------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------

class Encoder1D(nn.Module):
    """
    Hierarchical 1D-CNN encoder.

    Stages
    ------
    stem      : wide conv → 64 ch, reduces W by 2
    stage 1   : ResBlock × 2 @ 64 ch
    downsample: stride-2 conv → 128 ch
    stage 2   : ResBlock × 2 @ 128 ch
    downsample: stride-2 conv → 256 ch
    stage 3   : ResBlock × 2 @ 256 ch
    pool      : AdaptiveAvgPool1d(1) → (B, 256)
    """

    def __init__(self, in_channels: int, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        self.stem = ConvBlock(in_channels, 64, kernel=7, stride=2, padding=3)

        self.stage1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down1  = ConvBlock(64, 128, kernel=3, stride=2, padding=1)

        self.stage2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down2  = ConvBlock(128, embed_dim, kernel=3, stride=2, padding=1)

        self.stage3 = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))

        self.pool   = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, W) → embedding: (B, embed_dim)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)            # (B, embed_dim)
        return x


# -----------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------

class PseudoLabelPredictor(nn.Module):
    """
    Encoder + regression head for predicting normalised feature profiles.

    Args:
        in_channels  : number of input channels = n_sensors × 3
        n_features   : number of output scalars = n_sensors × 8
        embed_dim    : encoder embedding dimension (default 256)
        dropout      : dropout rate in the regression head
    """

    def __init__(self, in_channels: int, n_features: int,
                 embed_dim: int = 256, dropout: float = 0.3,
                 encoder: nn.Module = None):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
            embed_dim    = encoder.embed_dim
        else:
            self.encoder = Encoder1D(in_channels, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, W) → predictions: (B, n_features)"""
        emb = self.encoder(x)       # (B, embed_dim)
        return self.head(emb)       # (B, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding only (for downstream tasks)."""
        return self.encoder(x)


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def build_model(n_sensors: int = 4,
                n_feature_dims: int = 8,
                embed_dim: int = 256,
                dropout: float = 0.3,
                encoder: nn.Module = None) -> PseudoLabelPredictor:
    """
    Convenience constructor.

    Args:
        n_sensors      : number of accelerometer sensors
        n_feature_dims : feature dimensions per sensor (default 8 from PROFILE_KEYS)
        embed_dim      : encoder output size (ignored when encoder is provided)
        dropout        : head dropout
        encoder        : pre-built encoder module; if None, Encoder1D is created
    """
    return PseudoLabelPredictor(
        in_channels = n_sensors * 3,
        n_features  = n_sensors * n_feature_dims,
        embed_dim   = embed_dim,
        dropout     = dropout,
        encoder     = encoder,
    )


def model_summary(model: nn.Module, in_channels: int, window_frames: int):
    """Print parameter count and output shape."""
    device = next(model.parameters()).device
    x      = torch.zeros(1, in_channels, window_frames, device=device)
    out    = model(x)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {total:,} total  |  {train:,} trainable")
