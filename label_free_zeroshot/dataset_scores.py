"""
dataset_scores.py
-----------------
Datasets for pseudo-label score pre-training.

For each sliding-window segment, a score in [0, 1] is computed per
ActivityProfile by comparing the window's per-limb handcrafted feature
profile against the profile's expected low / medium / high feature levels.

Score = 1 - mean |normalised_actual - target_level|
  where target_level ∈ {0.0 (low), 0.5 (medium), 1.0 (high)}.

Supported datasets: WEAR (19 activities), MMFIT (11 activities).

For joint pre-training (--dataset both) the score vector has 30 dims:
  dims [0 :11] = MMFIT activity scores  (zero + masked out for WEAR windows)
  dims [11:30] = WEAR  activity scores  (zero + masked out for MMFIT windows)
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from features import compute_feature_profile
from dataset import (
    load_subject as _load_mmfit_subject,
    SENSOR_NAMES as MMFIT_SENSOR_NAMES,
    MMFIT_LIMB_GROUPS,
    MMFIT_SPLITS,
    WINDOW_FRAMES as MMFIT_WINDOW_FRAMES,
    STRIDE_FRAMES as MMFIT_STRIDE_FRAMES,
    FS as MMFIT_FS,
    PROFILE_KEYS,
)
from dataset_wear import (
    load_wear_subject,
    WEAR_SENSOR_NAMES, WEAR_LIMB_GROUPS, WEAR_SPLITS,
    WINDOW_FRAMES as WEAR_WINDOW_FRAMES,
    STRIDE_FRAMES as WEAR_STRIDE_FRAMES,
    FS as WEAR_FS,
)
from pseudo_labels import WEAR_ACTIVITIES, MMFIT_ACTIVITIES, PAMAP2_ACTIVITIES, ActivityProfile
from dataset_pamap2 import (
    load_pamap2_subject,
    PAMAP2_LIMB_GROUPS,
    WINDOW_FRAMES as PAMAP2_WINDOW_FRAMES,
    STRIDE_FRAMES as PAMAP2_STRIDE_FRAMES,
    FS as PAMAP2_FS,
)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

LEVEL_TARGETS = {"low": 0.0, "medium": 0.5, "high": 1.0}

N_WEAR_ACTIVITIES   = len(WEAR_ACTIVITIES)    # 19
N_MMFIT_ACTIVITIES  = len(MMFIT_ACTIVITIES)   # 11
N_PAMAP2_ACTIVITIES = len(PAMAP2_ACTIVITIES)  # 13
N_TOTAL_ACTIVITIES  = N_MMFIT_ACTIVITIES + N_WEAR_ACTIVITIES  # 30

# Binary masks selecting valid score dims in the 30-dim combined vector
MMFIT_MASK = np.zeros(N_TOTAL_ACTIVITIES, dtype=np.float32)
MMFIT_MASK[:N_MMFIT_ACTIVITIES] = 1.0
WEAR_MASK  = np.zeros(N_TOTAL_ACTIVITIES, dtype=np.float32)
WEAR_MASK[N_MMFIT_ACTIVITIES:]  = 1.0


# -----------------------------------------------------------------------
# LimbScoreNormalizer
# -----------------------------------------------------------------------

class LimbScoreNormalizer:
    """
    Per-(limb, feature) 5th / 95th percentile normalisation.

    Maps raw feature values to [0, 1] so that score distances are
    comparable across features with different physical units.
    """

    def __init__(self):
        # {limb_name: {feature_key: (p5, p95)}}
        self.bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit(self, limb_arrays: Dict[str, np.ndarray]) -> "LimbScoreNormalizer":
        """
        Args:
            limb_arrays : {limb_name: (N, n_feats) array of raw feature values}
        """
        for limb, arr in limb_arrays.items():
            self.bounds[limb] = {}
            for j, feat in enumerate(PROFILE_KEYS):
                vals = arr[:, j]
                p5   = float(np.percentile(vals, 5))
                p95  = float(np.percentile(vals, 95))
                self.bounds[limb][feat] = (p5, max(p95, p5 + 1e-8))
        return self

    def normalize(self, limb: str, feat: str, value: float) -> float:
        if limb not in self.bounds or feat not in self.bounds[limb]:
            return 0.5
        p5, p95 = self.bounds[limb][feat]
        return float(np.clip((value - p5) / (p95 - p5), 0.0, 1.0))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.bounds, f)

    @classmethod
    def load(cls, path: str) -> "LimbScoreNormalizer":
        with open(path, "rb") as f:
            bounds = pickle.load(f)
        obj = cls()
        obj.bounds = bounds
        return obj


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _slide_windows(data: np.ndarray, W: int, S: int) -> np.ndarray:
    starts = np.arange(0, len(data) - W + 1, S)
    return np.stack([data[s : s + W] for s in starts])   # (N, W, C)


def _compute_profiles(windows: np.ndarray, n_sensors: int,
                       fs: float) -> np.ndarray:
    """
    Compute per-sensor scalar feature profiles for a batch of windows.

    Returns: (N, n_sensors * 8) float32 — flat profile per window.
    """
    N, n_feats = len(windows), len(PROFILE_KEYS)
    out = np.zeros((N, n_sensors * n_feats), dtype=np.float32)
    for i, win in enumerate(windows):
        for s in range(n_sensors):
            acc = win[:, s*3 : (s+1)*3]
            fp  = compute_feature_profile(acc, fs)
            out[i, s*n_feats : (s+1)*n_feats] = [fp[k] for k in PROFILE_KEYS]
    return out


def _limb_arrays(profiles: np.ndarray, n_sensors: int,
                  limb_groups: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """
    Derive per-limb feature arrays by averaging sensors within each limb.

    Args:
        profiles : (N, n_sensors * n_feats)
        n_sensors: number of sensors (= channels / 3)
        limb_groups : {limb: [sensor_idx, ...]}

    Returns:
        {limb: (N, n_feats) float32}
    """
    n_feats = profiles.shape[1] // n_sensors
    arr     = profiles.reshape(-1, n_sensors, n_feats)   # (N, S, F)
    return {
        limb: arr[:, idxs, :].mean(axis=1)               # (N, F)
        for limb, idxs in limb_groups.items()
    }


def compute_scores_batch(
    limb_arrays: Dict[str, np.ndarray],
    activities:  Dict[str, ActivityProfile],
    normalizer:  LimbScoreNormalizer,
) -> np.ndarray:
    """
    Compute activity scores for N windows simultaneously.

    score[i, a] = 1 − mean over all (limb, feat) pairs of
                  |normalised_actual[i, feat] − target_level[feat]|

    Returns: (N, n_activities) float32 in [0, 1]
    """
    N       = next(iter(limb_arrays.values())).shape[0]
    n_acts  = len(activities)
    scores  = np.full((N, n_acts), 0.5, dtype=np.float64)

    for a_idx, (_, activity) in enumerate(activities.items()):
        total_diff = np.zeros(N, dtype=np.float64)
        n_pairs    = 0

        for limb, expected_feats in activity.limbs.items():
            if limb not in limb_arrays:
                continue
            arr = limb_arrays[limb]   # (N, n_feats)

            for feat, level in expected_feats.items():
                if feat not in PROFILE_KEYS:
                    continue
                f_idx = PROFILE_KEYS.index(feat)
                vals  = arr[:, f_idx]

                p5, p95 = normalizer.bounds.get(limb, {}).get(feat, (0.0, 1.0 + 1e-8))
                norm    = np.clip((vals - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)
                total_diff += np.abs(norm - LEVEL_TARGETS[level])
                n_pairs    += 1

        if n_pairs > 0:
            scores[:, a_idx] = 1.0 - total_diff / n_pairs

    return scores.astype(np.float32)


# -----------------------------------------------------------------------
# WEAR Score Dataset
# -----------------------------------------------------------------------

class WEARScoreDataset(Dataset):
    """
    Sliding-window WEAR dataset yielding (x, score_vector) pairs.

    x            : (12, W) float32 — 4 sensors × 3 axes × window_frames
    score_vector : (n_activities,) float32 ∈ [0, 1]

    Call set_normalizer_and_activities() after fitting the
    LimbScoreNormalizer on the training split.
    """

    N_SENSORS = 4

    def __init__(
        self,
        data_root:     str,
        subjects:      List[str],
        window_frames: int = WEAR_WINDOW_FRAMES,
        stride_frames: int = WEAR_STRIDE_FRAMES,
        limb_normalizer: Optional[LimbScoreNormalizer] = None,
        activities:    Optional[Dict[str, ActivityProfile]] = None,
        cache_dir:     Optional[str] = None,
    ):
        self._windows:  np.ndarray         # (N, W, 12)
        self._profiles: np.ndarray         # (N, 32)  — per-sensor, flat
        self._scores:   Optional[np.ndarray] = None   # (N, n_acts)

        self._build(data_root, subjects, window_frames, stride_frames, cache_dir)

        if limb_normalizer is not None and activities is not None:
            self.set_normalizer_and_activities(limb_normalizer, activities)

    def _build(self, data_root, subjects, W, S, cache_dir):
        wins_list, profs_list = [], []
        for subj in subjects:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir, f"scores_wear_{subj}_W{W}_S{S}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins  = cached["windows"]
                profs = cached["profiles"]
            else:
                print(f"  Loading {subj}...", flush=True)
                acc, _ = load_wear_subject(data_root, subj)
                wins   = _slide_windows(acc, W, S)
                profs  = _compute_profiles(wins, self.N_SENSORS, WEAR_FS)
                if cache_path:
                    np.savez_compressed(cache_path, windows=wins, profiles=profs)

            wins_list.append(wins)
            profs_list.append(profs)

        self._windows  = np.concatenate(wins_list,  axis=0)
        self._profiles = np.concatenate(profs_list, axis=0)

    # ------------------------------------------------------------------

    def raw_limb_arrays(self) -> Dict[str, np.ndarray]:
        """Per-limb (N, 8) arrays — used to fit LimbScoreNormalizer."""
        return _limb_arrays(self._profiles, self.N_SENSORS, WEAR_LIMB_GROUPS)

    def set_normalizer_and_activities(
        self,
        normalizer:  LimbScoreNormalizer,
        activities:  Dict[str, ActivityProfile],
    ):
        la           = _limb_arrays(self._profiles, self.N_SENSORS, WEAR_LIMB_GROUPS)
        self._scores = compute_scores_batch(la, activities, normalizer)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._windows[idx].T.copy()).float()   # (12, W)
        y = torch.from_numpy(self._scores[idx]).float()              # (n_acts,)
        return x, y


# -----------------------------------------------------------------------
# MMFIT Score Dataset
# -----------------------------------------------------------------------

class MMFitScoreDataset(Dataset):
    """
    Sliding-window MM-Fit dataset yielding (x, score_vector) pairs.

    x            : (12, W) float32 — 4 sensors × 3 axes × window_frames
    score_vector : (n_activities,) float32 ∈ [0, 1]
    """

    N_SENSORS = 4

    def __init__(
        self,
        data_dir:      str,
        subjects:      List[str],
        window_frames: int = MMFIT_WINDOW_FRAMES,
        stride_frames: int = MMFIT_STRIDE_FRAMES,
        limb_normalizer: Optional[LimbScoreNormalizer] = None,
        activities:    Optional[Dict[str, ActivityProfile]] = None,
        cache_dir:     Optional[str] = None,
    ):
        self._windows:  np.ndarray
        self._profiles: np.ndarray
        self._scores:   Optional[np.ndarray] = None

        self._build(data_dir, subjects, window_frames, stride_frames, cache_dir)

        if limb_normalizer is not None and activities is not None:
            self.set_normalizer_and_activities(limb_normalizer, activities)

    def _build(self, data_dir, subjects, W, S, cache_dir):
        wins_list, profs_list = [], []
        for subj in subjects:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir, f"scores_mmfit_{subj}_W{W}_S{S}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins  = cached["windows"]
                profs = cached["profiles"]
            else:
                print(f"  Loading {subj}...", flush=True)
                subj_dir = os.path.join(data_dir, subj)
                loaded   = _load_mmfit_subject(subj_dir, MMFIT_SENSOR_NAMES)
                data     = loaded["data"]                        # (F, 12)
                wins     = _slide_windows(data, W, S)
                profs    = _compute_profiles(wins, self.N_SENSORS, MMFIT_FS)
                if cache_path:
                    np.savez_compressed(cache_path, windows=wins, profiles=profs)

            wins_list.append(wins)
            profs_list.append(profs)

        self._windows  = np.concatenate(wins_list,  axis=0)
        self._profiles = np.concatenate(profs_list, axis=0)

    # ------------------------------------------------------------------

    def raw_limb_arrays(self) -> Dict[str, np.ndarray]:
        return _limb_arrays(self._profiles, self.N_SENSORS, MMFIT_LIMB_GROUPS)

    def set_normalizer_and_activities(
        self,
        normalizer:  LimbScoreNormalizer,
        activities:  Dict[str, ActivityProfile],
    ):
        la           = _limb_arrays(self._profiles, self.N_SENSORS, MMFIT_LIMB_GROUPS)
        self._scores = compute_scores_batch(la, activities, normalizer)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._windows[idx].T.copy()).float()
        y = torch.from_numpy(self._scores[idx]).float()
        return x, y


# -----------------------------------------------------------------------
# PAMAP2 Score Dataset
# -----------------------------------------------------------------------

class PAMAP2ScoreDataset(Dataset):
    """
    Sliding-window PAMAP2 dataset yielding (x, score_vector) pairs.

    x            : (9, W) float32 — 3 sensors × 3 axes × window_frames
    score_vector : (n_activities,) float32 ∈ [0, 1]
    """

    N_SENSORS = 3

    def __init__(
        self,
        data_root:     str,
        subjects:      List[str],
        window_frames: int = PAMAP2_WINDOW_FRAMES,
        stride_frames: int = PAMAP2_STRIDE_FRAMES,
        limb_normalizer: Optional[LimbScoreNormalizer] = None,
        activities:    Optional[Dict[str, ActivityProfile]] = None,
        cache_dir:     Optional[str] = None,
    ):
        self._windows:  np.ndarray
        self._profiles: np.ndarray
        self._scores:   Optional[np.ndarray] = None

        self._build(data_root, subjects, window_frames, stride_frames, cache_dir)

        if limb_normalizer is not None and activities is not None:
            self.set_normalizer_and_activities(limb_normalizer, activities)

    def _build(self, data_root, subjects, W, S, cache_dir):
        wins_list, profs_list = [], []
        for subj in subjects:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir, f"scores_pamap2_{subj}_W{W}_S{S}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins  = cached["windows"]
                profs = cached["profiles"]
            else:
                print(f"  Loading {subj}...", flush=True)
                acc, _ = load_pamap2_subject(data_root, subj)
                wins   = _slide_windows(acc, W, S)
                profs  = _compute_profiles(wins, self.N_SENSORS, PAMAP2_FS)
                if cache_path:
                    np.savez_compressed(cache_path, windows=wins, profiles=profs)

            wins_list.append(wins)
            profs_list.append(profs)

        self._windows  = np.concatenate(wins_list,  axis=0)
        self._profiles = np.concatenate(profs_list, axis=0)

    def raw_limb_arrays(self) -> Dict[str, np.ndarray]:
        return _limb_arrays(self._profiles, self.N_SENSORS, PAMAP2_LIMB_GROUPS)

    def set_normalizer_and_activities(
        self,
        normalizer:  LimbScoreNormalizer,
        activities:  Dict[str, ActivityProfile],
    ):
        la           = _limb_arrays(self._profiles, self.N_SENSORS, PAMAP2_LIMB_GROUPS)
        self._scores = compute_scores_batch(la, activities, normalizer)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._windows[idx].T.copy()).float()  # (9, W)
        y = torch.from_numpy(self._scores[idx]).float()             # (n_acts,)
        return x, y
