"""
dataset_pamap2.py
-----------------
Data loader for the PAMAP2 Physical Activity Monitoring dataset.

Layout on disk
--------------
  data/pamap2/Protocol/
    subject101.dat  …  subject109.dat

Each .dat file is whitespace-delimited with 54 columns per row:
  col 0   : timestamp (s)
  col 1   : activityID
  col 2   : heart rate (bpm)
  cols 3-19  : IMU hand  (17 ch: temp, acc16g xyz, acc6g xyz, gyro xyz, mag xyz, orient×4)
  cols 20-36 : IMU chest (same layout)
  cols 37-53 : IMU ankle (same layout)

We use only the ±16g accelerometer axes from each IMU (3 sensors × 3 axes = 9 channels).
Data is already in m/s².  Downsampled from 100 Hz → 33.3 Hz (every 3rd row).

Evaluation
----------
Leave-one-subject-out (LOSO) cross-validation across 9 subjects.
For each fold: test = one subject, val = next subject (circular), train = remaining 7.
Pre-training uses all 9 subjects (unsupervised — no class labels used).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from features import compute_feature_profile
from dataset import ProfileNormalizer, PROFILE_KEYS

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

PAMAP2_DATA_SUBDIR = "pamap2/Protocol"

# Columns to read from each .dat file (0-indexed pandas columns)
# label + acc16g from hand, chest, ankle
_USED_COLS = [1, 4, 5, 6, 21, 22, 23, 38, 39, 40]
_COL_NAMES = [
    "activity_id",
    "acc_x_hand",  "acc_y_hand",  "acc_z_hand",
    "acc_x_chest", "acc_y_chest", "acc_z_chest",
    "acc_x_ankle", "acc_y_ankle", "acc_z_ankle",
]

PAMAP2_SENSOR_NAMES = ["hand_acc", "chest_acc", "ankle_acc"]

FS            = 100.0 / 3.0   # Hz after downsampling (every 3rd row)
WINDOW_FRAMES = 67             # ~2 s at 33.3 Hz
STRIDE_FRAMES = 33             # ~50 % overlap

# 9 subjects
PAMAP2_SUBJECTS = [
    "subject101", "subject102", "subject103", "subject104", "subject105",
    "subject106", "subject107", "subject108", "subject109",
]

_FILE_ENCODING = {s: f"{s}.dat" for s in PAMAP2_SUBJECTS}

# Raw activity IDs → class index (class 0 = 'other'/background)
_LABEL_MAP = [
    (0,  "other"),
    (1,  "lying"),
    (2,  "sitting"),
    (3,  "standing"),
    (4,  "walking"),
    (5,  "running"),
    (6,  "cycling"),
    (7,  "nordic walking"),
    (12, "ascending stairs"),
    (13, "descending stairs"),
    (16, "vacuum cleaning"),
    (17, "ironing"),
    (24, "rope jumping"),
]
_RAW_ID_TO_IDX = {raw: idx for idx, (raw, _) in enumerate(_LABEL_MAP)}

PAMAP2_CLASSES = [name for _, name in _LABEL_MAP]   # 13 classes
PAMAP2_CLASS_TO_IDX = {c: i for i, c in enumerate(PAMAP2_CLASSES)}
PAMAP2_NULL_IDX = PAMAP2_CLASS_TO_IDX["other"]

# Limb groups: sensor indices into PAMAP2_SENSOR_NAMES
PAMAP2_LIMB_GROUPS: Dict[str, List[int]] = {
    "arm":       [0],   # hand
    "torso":     [1],   # chest
    "leg":       [2],   # ankle
}


# -----------------------------------------------------------------------
# LOSO splits
# -----------------------------------------------------------------------

def pamap2_loso_splits() -> List[Dict[str, List[str]]]:
    """
    Return a list of 9 LOSO fold dicts, each with 'train', 'val', 'test' keys.

    Fold i:
      test  = PAMAP2_SUBJECTS[i]
      val   = PAMAP2_SUBJECTS[(i+1) % 9]
      train = remaining 7 subjects
    """
    n = len(PAMAP2_SUBJECTS)
    folds = []
    for i in range(n):
        test  = [PAMAP2_SUBJECTS[i]]
        val   = [PAMAP2_SUBJECTS[(i + 1) % n]]
        train = [s for j, s in enumerate(PAMAP2_SUBJECTS) if j != i and j != (i + 1) % n]
        folds.append({"train": train, "val": val, "test": test})
    return folds


# -----------------------------------------------------------------------
# Subject loading
# -----------------------------------------------------------------------

def load_pamap2_subject(data_root: str, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one PAMAP2 subject .dat file.

    Returns
    -------
    acc    : (F, 9) float32  — 3 sensors × 3 axes, m/s², downsampled to 33.3 Hz
    labels : (F,)   int64   — per-frame class index (PAMAP2_CLASS_TO_IDX)
    """
    path = os.path.join(data_root, PAMAP2_DATA_SUBDIR, f"{subject}.dat")
    df = pd.read_table(path, header=None, sep=r'\s+')
    df = df.iloc[:, _USED_COLS]
    df.columns = _COL_NAMES

    # Interpolate missing values (NaN from sensor dropout)
    df = df.interpolate(method="linear", limit_direction="both")

    # Downsample: keep every 3rd row
    df = df.iloc[::3].reset_index(drop=True)

    # Map raw activity IDs → class indices (unknown IDs → 'other')
    df["activity_id"] = df["activity_id"].map(
        lambda x: _RAW_ID_TO_IDX.get(int(x), PAMAP2_NULL_IDX)
    )

    acc    = df[_COL_NAMES[1:]].values.astype(np.float32)   # (F, 9)
    labels = df["activity_id"].values.astype(np.int64)       # (F,)
    return acc, labels


# -----------------------------------------------------------------------
# Feature profile
# -----------------------------------------------------------------------

def compute_pamap2_window_profile(window: np.ndarray,
                                   fs: float = FS) -> np.ndarray:
    """(W, 9) → (24,) scalar feature profile (3 sensors × 8 features)."""
    parts = []
    for s in range(3):
        acc = window[:, s*3 : s*3+3]
        p   = compute_feature_profile(acc, fs)
        parts.append([p[k] for k in PROFILE_KEYS])
    return np.array(parts, dtype=np.float32).ravel()


# -----------------------------------------------------------------------
# Pre-training dataset
# -----------------------------------------------------------------------

class PAMAP2Dataset(Dataset):
    """
    Sliding-window dataset for PAMAP2 pseudo-label pre-training.

    Each item: (x, y) where
      x : (9, W) float32 tensor  — 3 sensors × 3 axes × window_frames
      y : (24,)  float32 tensor  — scalar feature profile (pseudo-label)
    """

    def __init__(
        self,
        data_root:     str,
        subjects:      List[str],
        window_frames: int = WINDOW_FRAMES,
        stride_frames: int = STRIDE_FRAMES,
        normalizer:    Optional[ProfileNormalizer] = None,
        cache_dir:     Optional[str] = None,
    ):
        self.normalizer    = normalizer
        self.window_frames = window_frames
        self.stride_frames = stride_frames

        all_windows:  List[np.ndarray] = []
        all_profiles: List[np.ndarray] = []

        for subj in subjects:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                tag        = f"pamap2_{subj}_W{window_frames}_S{stride_frames}"
                cache_path = os.path.join(cache_dir, f"{tag}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins   = cached["windows"]
                profs  = cached["profiles"]
            else:
                print(f"  Loading {subj}...", flush=True)
                acc, _ = load_pamap2_subject(data_root, subj)
                F      = len(acc)
                starts = np.arange(0, F - window_frames + 1, stride_frames)
                wins   = np.stack([acc[i : i + window_frames] for i in starts])
                profs  = np.stack([
                    compute_pamap2_window_profile(wins[j]) for j in range(len(wins))
                ])
                if cache_path:
                    np.savez_compressed(cache_path, windows=wins, profiles=profs)

            all_windows.append(wins)
            all_profiles.append(profs)

        self._windows  = np.concatenate(all_windows,  axis=0)
        self._profiles = np.concatenate(all_profiles, axis=0)

    def raw_profiles(self) -> np.ndarray:
        return self._profiles.copy()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._windows[idx].T.copy()).float()   # (9, W)
        y = self._profiles[idx].copy()
        if self.normalizer is not None:
            y = self.normalizer.transform(y[None])[0]
        return x, torch.from_numpy(y).float()


# -----------------------------------------------------------------------
# Labeled window loading (LOSO evaluation)
# -----------------------------------------------------------------------

def load_pamap2_labeled_windows(
    data_root:     str,
    subjects:      List[str],
    window_frames: int = WINDOW_FRAMES,
    stride_frames: int = STRIDE_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide windows over PAMAP2 sessions. Each window is assigned the majority
    label across its frames.

    Returns
    -------
    windows  : (N, W, 9) float32
    labels   : (N,)      int64
    subjects : (N,)      object
    """
    all_windows:  List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_subjects: List[str]        = []

    n_classes = len(PAMAP2_CLASSES)

    for subj in subjects:
        acc, frame_labels = load_pamap2_subject(data_root, subj)
        F      = len(acc)
        starts = np.arange(0, F - window_frames + 1, stride_frames)
        wins   = np.stack([acc[i : i + window_frames] for i in starts])

        counts = np.zeros((len(starts), n_classes), dtype=np.int32)
        for i, s in enumerate(starts):
            bc = np.bincount(frame_labels[s : s + window_frames],
                             minlength=n_classes)
            counts[i] = bc
        win_labels = counts.argmax(axis=1).astype(np.int64)

        all_windows.append(wins)
        all_labels.append(win_labels)
        all_subjects.extend([subj] * len(starts))

    windows  = np.concatenate(all_windows,  axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0)
    subjects = np.array(all_subjects)
    return windows, labels, subjects
