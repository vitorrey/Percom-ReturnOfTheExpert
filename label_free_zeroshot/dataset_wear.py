"""
dataset_wear.py
---------------
Data loader for the WEAR dataset.

Layout on disk
--------------
  data/ubi29.informatik.uni-siegen.de/wear_dataset/
    raw/inertial/50hz/
      sbj_0.csv  …  sbj_17.csv
        columns: sbj_id, right_arm_acc_{x,y,z}, right_leg_acc_{x,y,z},
                        left_leg_acc_{x,y,z},  left_arm_acc_{x,y,z}, label

Each CSV row is one frame at 50 Hz.  Accelerometer values are in g;
they are converted to m/s² (×9.81) on load to match the MMFIT pipeline.

Splits
------
No official train/test split exists for WEAR (the published JSONs define
18 LOSO folds with a single validation subject each, no separate test set).

We define a fixed split here, stratified by session length so each partition
has roughly equal total recording time:

  Train (10 subjects): sbj_1  sbj_2  sbj_5  sbj_6  sbj_7  sbj_8  sbj_9  sbj_11  sbj_15  sbj_17
  Val   (4 subjects) : sbj_0  sbj_4  sbj_12  sbj_14
  Test  (4 subjects) : sbj_3  sbj_10  sbj_13  sbj_16

Sensor → limb mapping (WEAR_LIMB_GROUPS):
  arm : right_arm (idx 0), left_arm (idx 3)
  leg : right_leg (idx 1), left_leg (idx 2)
"""

import os
import csv
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from features import compute_feature_profile
from dataset import ProfileNormalizer, PROFILE_KEYS   # reuse normaliser + keys

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

WEAR_DATA_SUBDIR = "ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz"

# Column order in the CSV (excluding sbj_id and label)
WEAR_SENSOR_NAMES = [
    "right_arm_acc",   # idx 0
    "right_leg_acc",   # idx 1
    "left_leg_acc",    # idx 2
    "left_arm_acc",    # idx 3
]
G_TO_MS2 = 9.81

FS            = 50.0   # Hz
WINDOW_FRAMES = 100    # 2 s at 50 Hz
STRIDE_FRAMES = 50     # 50 % overlap

# Limb groups: sensor indices (into WEAR_SENSOR_NAMES)
WEAR_LIMB_GROUPS: Dict[str, List[int]] = {
    "arm": [0, 3],   # right_arm, left_arm
    "leg": [1, 2],   # right_leg, left_leg
}

# Fixed train / val / test split (stratified by session length)
# Sorted by session length (rows): short→long
# short: sbj_5(107k), sbj_9(108k), sbj_11(111k), sbj_8(124k), sbj_6(125k)
#        sbj_15(130k), sbj_1(138k), sbj_0(139k), sbj_7(142k), sbj_17(154k)
# mid  : sbj_4(159k), sbj_14(160k), sbj_12(163k), sbj_4(163k)
# long : sbj_16(175k), sbj_2(178k), sbj_10(199k), sbj_13(208k), sbj_3(208k)
WEAR_SPLITS: Dict[str, List[str]] = {
    "train": ["sbj_1", "sbj_2", "sbj_5", "sbj_6", "sbj_7",
              "sbj_8", "sbj_9", "sbj_11", "sbj_15", "sbj_17"],
    "val":   ["sbj_0", "sbj_4", "sbj_12", "sbj_14"],
    "test":  ["sbj_3", "sbj_10", "sbj_13", "sbj_16"],
}

# Ordered class list (label string → int idx)
# Matches the label_map provided by the WEAR authors, plus null at index 0
WEAR_CLASSES = [
    "null",
    "jogging",
    "jogging (rotating arms)",
    "jogging (skipping)",
    "jogging (sidesteps)",
    "jogging (butt-kicks)",
    "stretching (triceps)",
    "stretching (lunging)",
    "stretching (shoulders)",
    "stretching (hamstrings)",
    "stretching (lumbar rotation)",
    "push-ups",
    "push-ups (complex)",
    "sit-ups",
    "sit-ups (complex)",
    "burpees",
    "lunges",
    "lunges (complex)",
    "bench-dips",
]
WEAR_CLASS_TO_IDX = {c: i for i, c in enumerate(WEAR_CLASSES)}
WEAR_NULL_IDX     = WEAR_CLASS_TO_IDX["null"]

# Map CSV label strings → pseudo_labels.py activity keys
WEAR_LABEL_TO_PROFILE_KEY = {
    "null":                       "null",
    "jogging":                    "jogging",
    "jogging (rotating arms)":    "jogging_rotating_arms",
    "jogging (skipping)":         "jogging_skipping",
    "jogging (sidesteps)":        "jogging_sidesteps",
    "jogging (butt-kicks)":       "jogging_butt_kicks",
    "stretching (triceps)":       "stretching_triceps",
    "stretching (lunging)":       "stretching_lunging",
    "stretching (shoulders)":     "stretching_shoulders",
    "stretching (hamstrings)":    "stretching_hamstrings",
    "stretching (lumbar rotation)": "stretching_lumbar_rotation",
    "push-ups":                   "push_ups",
    "push-ups (complex)":         "push_ups_complex",
    "sit-ups":                    "sit_ups",
    "sit-ups (complex)":          "sit_ups_complex",
    "burpees":                    "burpees",
    "lunges":                     "lunges",
    "lunges (complex)":           "lunges_complex",
    "bench-dips":                 "bench_dips",
}


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------

def load_wear_subject(data_root: str, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one WEAR subject CSV.

    Returns
    -------
    acc    : (F, 12) float32  — 4 sensors × 3 axes, m/s²
    labels : (F,)   int64    — per-frame WEAR_CLASS_TO_IDX label
    """
    path = os.path.join(data_root, WEAR_DATA_SUBDIR, f"{subject}.csv")
    acc_cols = []
    label_col = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_acc = []
            for sensor in WEAR_SENSOR_NAMES:
                def _parse(v):
                    s = v.strip()
                    return float("nan") if s == "" else float(s) * G_TO_MS2
                frame_acc += [
                    _parse(row[f"{sensor}_x"]),
                    _parse(row[f"{sensor}_y"]),
                    _parse(row[f"{sensor}_z"]),
                ]
            acc_cols.append(frame_acc)
            label_col.append(WEAR_CLASS_TO_IDX.get(row["label"], WEAR_NULL_IDX))

    acc    = np.array(acc_cols, dtype=np.float32)     # (F, 12)
    labels = np.array(label_col, dtype=np.int64)      # (F,)

    # Interpolate any NaN columns caused by sensor dropout
    if np.isnan(acc).any():
        for col in range(acc.shape[1]):
            col_data = acc[:, col]
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                idx = np.arange(len(col_data))
                col_data[nan_mask] = np.interp(
                    idx[nan_mask], idx[~nan_mask], col_data[~nan_mask])
                acc[:, col] = col_data

    return acc, labels


def compute_wear_window_profile(window: np.ndarray,
                                 n_sensors: int = 4,
                                 fs: float = FS) -> np.ndarray:
    """(W, 12) → (32,) scalar feature profile."""
    parts = []
    for s in range(n_sensors):
        acc = window[:, s*3 : s*3+3]
        p   = compute_feature_profile(acc, fs)
        parts.append([p[k] for k in PROFILE_KEYS])
    return np.array(parts, dtype=np.float32).ravel()


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class WEARDataset(Dataset):
    """
    Sliding-window dataset for WEAR pre-training (pseudo-label prediction).

    Each item: (x, y) where
      x : (12, W) float32 tensor — 4 sensors × 3 axes × window_frames
      y : (32,)  float32 tensor — scalar feature profile (pseudo-label)
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
                tag        = f"wear_{subj}_W{window_frames}_S{stride_frames}"
                cache_path = os.path.join(cache_dir, f"{tag}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins   = cached["windows"]
                profs  = cached["profiles"]
            else:
                print(f"  Loading {subj}...", flush=True)
                acc, _ = load_wear_subject(data_root, subj)
                F      = len(acc)
                starts = np.arange(0, F - window_frames + 1, stride_frames)
                wins   = np.stack([acc[i : i + window_frames] for i in starts])
                profs  = np.stack([
                    compute_wear_window_profile(wins[j]) for j in range(len(wins))
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
        x = torch.from_numpy(self._windows[idx].T.copy()).float()
        y = self._profiles[idx].copy()
        if self.normalizer is not None:
            y = self.normalizer.transform(y[None])[0]
        return x, torch.from_numpy(y).float()


# -----------------------------------------------------------------------
# Labeled window loading (for supervised / evaluation)
# -----------------------------------------------------------------------

def load_wear_labeled_windows(
    data_root:     str,
    subjects:      List[str],
    window_frames: int = WINDOW_FRAMES,
    stride_frames: int = STRIDE_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide windows over WEAR sessions. Each window is assigned the majority
    label across its frames.

    Returns
    -------
    windows  : (N, W, 12) float32
    labels   : (N,)       int64
    subjects : (N,)       object
    """
    all_windows:  List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_subjects: List[str]        = []

    for subj in subjects:
        acc, frame_labels = load_wear_subject(data_root, subj)
        F      = len(acc)
        starts = np.arange(0, F - window_frames + 1, stride_frames)
        wins   = np.stack([acc[i : i + window_frames] for i in starts])

        n_classes = len(WEAR_CLASSES)
        counts    = np.zeros((len(starts), n_classes), dtype=np.int32)
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
