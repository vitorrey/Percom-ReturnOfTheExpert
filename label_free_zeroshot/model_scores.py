"""
model_scores.py
---------------
Score predictor for pseudo-label score pre-training.

Uses the same Encoder1D from model.py — identical architecture, identical
embedding dimension — so fine-tuned results are directly comparable with
the feature-prediction baseline.

Only the head differs: instead of predicting normalised feature values
(unbounded regression), it predicts activity similarity scores ∈ [0, 1]
(sigmoid output, BCE/MSE loss).
"""

import torch
import torch.nn as nn

from model import Encoder1D   # shared encoder — same arch as all other methods


class ScorePredictor(nn.Module):
    """
    Encoder + sigmoid regression head for activity score prediction.

    Args:
        in_channels  : C = n_sensors × 3  (e.g. 12 for 4 sensors)
        n_activities : number of pseudo-label activity scores to predict
        embed_dim    : encoder embedding size — must match other methods (default 256)
        dropout      : dropout rate in the regression head
    """

    def __init__(self, in_channels: int, n_activities: int,
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
            nn.Linear(embed_dim // 2, n_activities),
            nn.Sigmoid(),   # outputs ∈ (0, 1) — activity scores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, W) → scores: (B, n_activities)"""
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding only (same interface as PseudoLabelPredictor)."""
        return self.encoder(x)


def build_score_model(n_sensors: int = 4,
                      n_activities: int = 19,
                      embed_dim: int = 256,
                      dropout: float = 0.3,
                      encoder: nn.Module = None) -> ScorePredictor:
    return ScorePredictor(
        in_channels  = n_sensors * 3,
        n_activities = n_activities,
        embed_dim    = embed_dim,
        dropout      = dropout,
        encoder      = encoder,
    )
