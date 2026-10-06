"""
dataset_pim.py
--------------
Dataset wrapper for PIM pseudo-label pre-training.

Each window produces 6 items (the window + 3 augmented versions, each paired
with the window, all sharing the same pseudo-label), following the PIM paper.

Augmentations:
  - horizontally_flipped : time reversal
  - permuted             : random segment shuffle
  - time_warped          : cubic-spline temporal warping

Usage
-----
  disc = PIMDiscretizer(fs=50).fit(train_windows, "wear")
  ds   = PIMDataset(train_windows, "wear", disc)
  loader = DataLoader(ds, batch_size=256, shuffle=True)
"""

import copy
import numpy as np
import scipy.interpolate
import torch
from torch.utils.data import Dataset

from features_pim import PIMDiscretizer, PS_HEADS


# -----------------------------------------------------------------------
# Augmentations used by PIM pre-training
# -----------------------------------------------------------------------

def horizontally_flipped(x: np.ndarray) -> np.ndarray:
    """Reverse the time axis.  x: (W, C)"""
    return x[::-1].copy()


def permuted(x: np.ndarray, num_segments: int = 4) -> np.ndarray:
    """Randomly shuffle num_segments sections of the signal.  x: (W, C)"""
    W = x.shape[0]
    cut_points = np.sort(np.random.choice(W, size=num_segments, replace=False))
    splits = np.split(x, cut_points, axis=0)
    np.random.shuffle(splits)
    return np.concatenate(splits, axis=0)


def time_warped(x: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    """Cubic-spline temporal warping.  x: (W, C)"""
    W, C = x.shape
    time_stamps = np.arange(W)
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (W - 1) / (num_knots + 1)

    out = np.empty_like(x)
    for c in range(C):
        spline_ys = np.random.normal(loc=1.0, scale=sigma, size=num_knots + 2)
        cs = scipy.interpolate.CubicSpline(knot_xs, spline_ys)
        warp = cs(time_stamps)
        cumsum = np.cumsum(warp)
        distorted = cumsum / cumsum[-1] * (W - 1)
        out[:, c] = np.interp(time_stamps, distorted, x[:, c])
    return out


AUGMENTATIONS = [horizontally_flipped, permuted, time_warped]


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class PIMDataset(Dataset):
    """
    Wraps raw windows (N, W, C) with PIM pseudo-labels and 6× augmentation.

    Parameters
    ----------
    windows    : (N, W, C) float32 — raw accelerometer windows
    dataset    : "wear" | "mmfit" | "pamap2"
    discretizer: fitted PIMDiscretizer
    augment    : if True apply 3 augmentations → 6× expansion (default True)
    """

    def __init__(self, windows: np.ndarray, dataset: str,
                 discretizer: PIMDiscretizer, augment: bool = True):
        self.dataset = dataset
        self.ahn, self.shn, self.mhn = PS_HEADS[dataset]

        xs, ys = [], []
        for win in windows:
            ps = discretizer.transform_window(win, dataset)
            if augment:
                for aug in AUGMENTATIONS:
                    xs.append(torch.from_numpy(win.T.copy()).float())   # un-augmented window
                    ys.append(ps)
                    xs.append(torch.from_numpy(aug(win).T.copy()).float())  # augmented
                    ys.append(ps)
            else:
                xs.append(torch.from_numpy(win.T.copy()).float())
                ys.append(ps)

        self.xs = xs
        self.ys = ys  # list of ps lists

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x  = self.xs[idx]                       # (C, W)
        ps = [torch.from_numpy(y) for y in self.ys[idx]]  # list of tensors
        return x, ps
