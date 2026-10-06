"""
model_pim.py
------------
PIM pseudo-label pre-training model with our Encoder1D backbone.

Head structure (per dataset):
  WEAR   (4 angle, 2 sym, 4 motion heads)
  MMFIT  (4 angle, 1 sym, 4 motion heads)
  PAMAP2 (3 angle, 0 sym, 3 motion heads)

Losses:
  angle heads  : BCELoss    (targets: float one-hot (3, N_BINS))
  sym/mot heads: CrossEntropyLoss (targets: float one-hot (N_BINS,))

Downstream use
--------------
  model.encoder(x)  →  (B, 256)   — same interface as all other methods
"""

import torch
import torch.nn as nn

from model import Encoder1D
from features_pim import PS_HEADS, N_BINS

EMBED_DIM = 256


class _AngleHead(nn.Module):
    """Three independent linear heads, one per angle dim (roll/pitch/yaw)."""
    def __init__(self, embed_dim: int, n_bins: int):
        super().__init__()
        self.heads    = nn.ModuleList([nn.Linear(embed_dim, n_bins) for _ in range(3)])
        self.sigmoid  = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.sigmoid(h(x)) for h in self.heads], dim=1)  # (B, 3, n_bins)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.forward(x), y.float())


class _ScalarHead(nn.Module):
    """Single linear head for symmetry or motion (one scalar → n_bins classes)."""
    def __init__(self, embed_dim: int, n_bins: int):
        super().__init__()
        self.head      = nn.Linear(embed_dim, n_bins)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, n_bins)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.forward(x), y.float())


class PIMModel(nn.Module):
    def __init__(self, in_channels: int, dataset: str,
                 embed_dim: int = EMBED_DIM, n_bins: int = N_BINS):
        super().__init__()
        self.encoder = Encoder1D(in_channels, embed_dim)
        ahn, shn, mhn = PS_HEADS[dataset]
        self.angle_heads  = nn.ModuleList([_AngleHead(embed_dim, n_bins)  for _ in range(ahn)])
        self.sym_heads    = nn.ModuleList([_ScalarHead(embed_dim, n_bins) for _ in range(shn)])
        self.motion_heads = nn.ModuleList([_ScalarHead(embed_dim, n_bins) for _ in range(mhn)])
        self.ahn, self.shn, self.mhn = ahn, shn, mhn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_losses(self, x: torch.Tensor, ps: list) -> torch.Tensor:
        """
        x  : (B, C, W)
        ps : list of tensors matching the ps list order for this dataset
             [ang_0..ahn-1, sym_0..shn-1, mot_0..mhn-1]
        """
        emb = self.encoder(x)

        angl_y = ps[:self.ahn]
        sym_y  = ps[self.ahn: self.ahn + self.shn]
        mots_y = ps[-self.mhn:] if self.mhn > 0 else []

        losses = []
        for h, y in zip(self.angle_heads,  angl_y): losses.append(h.loss(emb, y))
        for h, y in zip(self.sym_heads,    sym_y):  losses.append(h.loss(emb, y))
        for h, y in zip(self.motion_heads, mots_y): losses.append(h.loss(emb, y))

        return sum(losses) / len(losses)


def build_pim(dataset: str, n_sensors: int,
              embed_dim: int = EMBED_DIM, n_bins: int = N_BINS) -> PIMModel:
    return PIMModel(in_channels=n_sensors * 3,
                    dataset=dataset,
                    embed_dim=embed_dim,
                    n_bins=n_bins)
