"""
dataset.py
----------
Efficient MMFIT data loader for pseudo-label pre-training.

Layout on disk
--------------
  data/mm-fit/
    w00/
      w00_sw_l_acc.npy   # shape (N, 5): [frame_idx, unix_ts_ms, ax, ay, az]
      w00_sw_r_acc.npy
      w00_sp_r_acc.npy
      w00_eb_l_acc.npy
      w00_labels.csv     # start_frame, end_frame, reps, class_name
    w01/ ...

Processing pipeline
-------------------
1. Group raw rows by frame_idx, keep mean of ax/ay/az.
2. Build a dense array over the common frame range of all sensors,
   interpolating missing frames (< 10 % in all sensors).
3. Slide a window of WINDOW_FRAMES with STRIDE_FRAMES over the session.
4. Compute a scalar feature profile per window (pseudo-label target).
5. During training, normalise targets with a StandardScaler fitted on
   the training subjects (leave-one-out split).
"""

import os
import csv
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from features import compute_feature_profile

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
SENSOR_NAMES = ["sw_l_acc", "sw_r_acc", "sp_r_acc", "eb_l_acc"]
FS           = 30.0          # approximate frame rate (Hz)
WINDOW_FRAMES = 60           # 2 seconds at 30 Hz
STRIDE_FRAMES = 30           # 50 % overlap

# Official MM-Fit subject splits (from KDMStromback/mm-fit repo)
MMFIT_SPLITS = {
    "train":       ["w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"],
    "val":         ["w14", "w15", "w19"],
    "test":        ["w09", "w10", "w11"],
    "test_unseen": ["w00", "w05", "w12", "w13", "w20"],
}

# Sensor index groups per anatomical limb
# Order matches SENSOR_NAMES: sw_l=0, sw_r=1, sp_r=2, eb_l=3
MMFIT_LIMB_GROUPS: Dict[str, List[int]] = {
    "wrist":      [0, 1],   # smartwatch left & right wrist
    "lower_body": [2],      # smartphone in right pocket (hip proxy)
    "head":       [3],      # earbuds left ear
}


# -----------------------------------------------------------------------
# Per-subject loading
# -----------------------------------------------------------------------

def _aggregate_frames(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate multiple rows with the same frame index into a single mean row.

    Args:
        raw : (N, 5) array — [frame_idx, ts_ms, ax, ay, az]

    Returns:
        frames : (F,)    unique frame indices, sorted
        acc    : (F, 3)  mean [ax, ay, az] per frame
    """
    frame_col = raw[:, 0].astype(np.int64)
    acc_cols  = raw[:, 2:5].astype(np.float32)

    order  = np.argsort(frame_col, kind="stable")
    frames_sorted = frame_col[order]
    acc_sorted    = acc_cols[order]

    unique_frames, idx_start = np.unique(frames_sorted, return_index=True)
    idx_end = np.append(idx_start[1:], len(frames_sorted))

    mean_acc = np.stack([
        acc_sorted[s:e].mean(axis=0)
        for s, e in zip(idx_start, idx_end)
    ])                                              # (F, 3)

    return unique_frames, mean_acc


def _fill_gaps(frames: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """
    Build a dense array from frames[0] to frames[-1], linearly interpolating
    any missing frame indices.

    Returns: (M, 3) dense array where M = frames[-1] - frames[0] + 1
    """
    start, end = int(frames[0]), int(frames[-1])
    M = end - start + 1
    dense = np.empty((M, 3), dtype=np.float32)

    # Fill known frames
    rel = frames - start
    dense[rel] = acc

    # Identify and interpolate gaps
    known_mask = np.zeros(M, dtype=bool)
    known_mask[rel] = True
    if known_mask.all():
        return dense

    all_idx = np.arange(M)
    for ch in range(3):
        dense[:, ch] = np.interp(all_idx, all_idx[known_mask], dense[known_mask, ch])

    return dense


def load_subject(subject_dir: str,
                 sensor_names: List[str] = SENSOR_NAMES) -> Dict[str, object]:
    """
    Load and align all accelerometer sensors for one MMFIT subject.

    Returns a dict with:
        "data"       : (F, 3 * S) float32 — aligned sensor channels
        "frame_start": int  — first frame index in common range
        "frame_end"  : int  — last frame index in common range
        "sensors"    : list of sensor names in column order
    """
    subject_id = os.path.basename(subject_dir)

    dense_per_sensor: Dict[str, Tuple[int, np.ndarray]] = {}
    for sensor in sensor_names:
        path = os.path.join(subject_dir, f"{subject_id}_{sensor}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing sensor file: {path}")
        raw = np.load(path)
        frames, acc = _aggregate_frames(raw)
        dense = _fill_gaps(frames, acc)                 # (Mi, 3)
        dense_per_sensor[sensor] = (int(frames[0]), dense)

    # Common frame range across sensors
    common_start = max(v[0]               for v in dense_per_sensor.values())
    common_end   = min(v[0] + len(v[1]) - 1 for v in dense_per_sensor.values())
    length = common_end - common_start + 1

    # Slice each sensor to the common range and concatenate
    channels = []
    for sensor in sensor_names:
        frame_start, dense = dense_per_sensor[sensor]
        s = common_start - frame_start
        channels.append(dense[s : s + length])          # (length, 3)

    data = np.concatenate(channels, axis=1)             # (length, 3*S)

    return {
        "data":        data,
        "frame_start": common_start,
        "frame_end":   common_end,
        "sensors":     sensor_names,
    }


# -----------------------------------------------------------------------
# Normaliser
# -----------------------------------------------------------------------

class ProfileNormalizer:
    """
    Fits a StandardScaler on feature profiles computed from training windows,
    then applies it to standardize targets to zero mean / unit variance.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, profiles: np.ndarray) -> "ProfileNormalizer":
        """profiles: (N, D) array of raw scalar feature profiles."""
        self.mean_ = profiles.mean(axis=0)
        self.std_  = profiles.std(axis=0) + 1e-8
        return self

    def transform(self, profiles: np.ndarray) -> np.ndarray:
        return (profiles - self.mean_) / self.std_

    def inverse_transform(self, profiles: np.ndarray) -> np.ndarray:
        return profiles * self.std_ + self.mean_

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)

    @classmethod
    def load(cls, path: str) -> "ProfileNormalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.mean_ = d["mean"]
        obj.std_  = d["std"]
        return obj

    def percentile_thresholds(self, profiles: np.ndarray,
                               q_lo: float = 33.0,
                               q_hi: float = 67.0,
                               profile_keys: Optional[List[str]] = None
                               ) -> Dict[str, Tuple[float, float]]:
        """
        Compute (q_lo, q_hi) percentile thresholds per feature dimension
        from raw (un-normalised) profiles — used by discretize_profile().

        Returns a dict keyed either by profile_keys or by int index.
        """
        thresholds = {}
        n_dims = profiles.shape[1]
        for i in range(n_dims):
            lo = float(np.percentile(profiles[:, i], q_lo))
            hi = float(np.percentile(profiles[:, i], q_hi))
            key = profile_keys[i] if profile_keys else i
            thresholds[key] = (lo, hi)
        return thresholds


# -----------------------------------------------------------------------
# Feature profile helpers
# -----------------------------------------------------------------------

PROFILE_KEYS = [
    "rms", "speed_mean", "distance_total", "impact_rate",
    "tilt_mean", "tilt_variability", "angular_range", "angular_rate",
]


def compute_window_profile(window: np.ndarray,
                            n_sensors: int,
                            fs: float = FS) -> np.ndarray:
    """
    Compute a flat feature-profile vector for a multi-sensor window.

    Args:
        window   : (W, 3*n_sensors) float32
        n_sensors: number of sensors (channels / 3)
        fs       : sampling rate

    Returns:
        profile : (8 * n_sensors,) float32
    """
    parts = []
    for s in range(n_sensors):
        acc = window[:, s*3 : s*3+3]            # (W, 3)
        p   = compute_feature_profile(acc, fs)   # dict of 8 scalars
        parts.append([p[k] for k in PROFILE_KEYS])
    return np.array(parts, dtype=np.float32).ravel()   # (8*n_sensors,)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class MMFitDataset(Dataset):
    """
    Sliding-window dataset over MMFIT accelerometer data.

    Pre-computes feature profiles (pseudo-label targets) at construction time
    and caches them; subsequent instantiations load from cache.

    Args:
        data_dir     : path to the mm-fit root directory
        subjects     : list of subject IDs to include, e.g. ["w00", "w01"]
        window_frames: window length in frames (default 60 = 2 s)
        stride_frames: stride between windows (default 30 = 50 % overlap)
        sensor_names : list of sensor file suffixes to load
        normalizer   : fitted ProfileNormalizer; if None, targets are raw
        cache_dir    : directory to write/read pre-computed profiles
    """

    def __init__(
        self,
        data_dir:      str,
        subjects:      List[str],
        window_frames: int = WINDOW_FRAMES,
        stride_frames: int = STRIDE_FRAMES,
        sensor_names:  List[str] = SENSOR_NAMES,
        normalizer:    Optional[ProfileNormalizer] = None,
        cache_dir:     Optional[str] = None,
    ):
        self.data_dir      = data_dir
        self.subjects      = subjects
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.sensor_names  = sensor_names
        self.normalizer    = normalizer
        self.n_sensors     = len(sensor_names)

        self._windows:  List[np.ndarray] = []   # raw signal windows
        self._profiles: List[np.ndarray] = []   # feature profile per window
        self._subject_ids: List[str] = []       # subject per window (for LOSO)

        self._build(cache_dir)

    # ------------------------------------------------------------------

    def _build(self, cache_dir: Optional[str]):
        for subj in self.subjects:
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                tag = f"{subj}_W{self.window_frames}_S{self.stride_frames}"
                cache_path = os.path.join(cache_dir, f"{tag}.npz")

            if cache_path and os.path.exists(cache_path):
                cached = np.load(cache_path)
                wins    = cached["windows"]     # (N, W, C)
                profs   = cached["profiles"]    # (N, D)
            else:
                print(f"  Loading {subj}...", flush=True)
                subj_dir = os.path.join(self.data_dir, subj)
                loaded   = load_subject(subj_dir, self.sensor_names)
                data     = loaded["data"]       # (F, 3*S)

                wins, profs = self._slide_and_compute(data)

                if cache_path:
                    np.savez_compressed(cache_path, windows=wins, profiles=profs)

            self._windows.append(wins)
            self._profiles.append(profs)
            self._subject_ids.extend([subj] * len(wins))

        self._windows  = np.concatenate(self._windows,  axis=0)  # (N, W, C)
        self._profiles = np.concatenate(self._profiles, axis=0)  # (N, D)

    def _slide_and_compute(self, data: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        F, C = data.shape
        W    = self.window_frames
        S    = self.stride_frames

        starts = range(0, F - W + 1, S)
        wins   = np.stack([data[i : i + W] for i in starts])   # (N, W, C)
        profs  = np.stack([
            compute_window_profile(wins[i], self.n_sensors)
            for i in range(len(wins))
        ])                                                       # (N, D)
        return wins, profs

    # ------------------------------------------------------------------

    def raw_profiles(self) -> np.ndarray:
        """Return all raw (un-normalised) feature profiles. Useful for fitting
        the normalizer on the training split."""
        return self._profiles.copy()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        window  = self._windows[idx]                          # (W, C)
        profile = self._profiles[idx].copy()                  # (D,)

        if self.normalizer is not None:
            profile = self.normalizer.transform(profile[None])[0]

        # (C, W) — Conv1d expects channels first
        x = torch.from_numpy(window.T.copy()).float()         # (C, W)
        y = torch.from_numpy(profile).float()                 # (D,)
        return x, y


# -----------------------------------------------------------------------
# Leave-one-subject-out split helper
# -----------------------------------------------------------------------

def loso_split(data_dir:      str,
               hold_out:      str,
               window_frames: int = WINDOW_FRAMES,
               stride_frames: int = STRIDE_FRAMES,
               sensor_names:  List[str] = SENSOR_NAMES,
               cache_dir:     Optional[str] = None,
               ) -> Tuple[MMFitDataset, MMFitDataset, ProfileNormalizer]:
    """
    Build train / val datasets for one LOSO fold.

    1. Loads all subjects.
    2. Fits a ProfileNormalizer on the training subjects.
    3. Returns (train_dataset, val_dataset, normalizer).

    Args:
        data_dir : path to mm-fit root
        hold_out : subject ID held out for validation (e.g. "w03")
    """
    all_subjects  = sorted(os.listdir(data_dir))
    all_subjects  = [s for s in all_subjects if s.startswith("w")]
    train_subjects = [s for s in all_subjects if s != hold_out]
    val_subjects   = [hold_out]

    print(f"LOSO fold: hold-out={hold_out}  "
          f"train={len(train_subjects)} subjects")

    # Build training set without normalizer first to collect raw profiles
    print("Building training set...")
    train_ds = MMFitDataset(data_dir, train_subjects,
                            window_frames, stride_frames,
                            sensor_names, normalizer=None,
                            cache_dir=cache_dir)

    # Fit normalizer on training profiles
    normalizer = ProfileNormalizer().fit(train_ds.raw_profiles())
    train_ds.normalizer = normalizer

    # Build validation set with the same normalizer
    print("Building validation set...")
    val_ds = MMFitDataset(data_dir, val_subjects,
                          window_frames, stride_frames,
                          sensor_names, normalizer=normalizer,
                          cache_dir=cache_dir)

    return train_ds, val_ds, normalizer


# -----------------------------------------------------------------------
# Labeled window loading (for supervised / evaluation)
# -----------------------------------------------------------------------

# Ordered class list — exercise classes alphabetically, null last
MMFIT_CLASSES = [
    "bicep_curls",
    "dumbbell_rows",
    "dumbbell_shoulder_press",
    "jumping_jacks",
    "lateral_shoulder_raises",
    "lunges",
    "pushups",
    "situps",
    "squats",
    "tricep_extensions",
    "null",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(MMFIT_CLASSES)}
NULL_IDX     = CLASS_TO_IDX["null"]


def _parse_labels(label_path: str, frame_start: int) -> List[Tuple[int, int, str]]:
    """
    Parse a labels CSV and return a list of (rel_start, rel_end, class_name)
    where frame indices are relative to frame_start.
    """
    segments = []
    with open(label_path) as f:
        for row in csv.reader(f):
            start, end, _, cls = int(row[0]), int(row[1]), int(row[2]), row[3]
            if cls not in CLASS_TO_IDX:
                continue
            segments.append((start - frame_start, end - frame_start, cls))
    return segments


def load_labeled_windows(
    data_dir:      str,
    subjects:      List[str],
    window_frames: int = WINDOW_FRAMES,
    stride_frames: int = STRIDE_FRAMES,
    sensor_names:  List[str] = SENSOR_NAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide windows over entire sessions and assign a class label to each window
    by majority vote: the class that occupies the most frames inside the window
    wins. Frames not covered by any labelled segment count as null.

    Returns
    -------
    windows  : (N, W, C) float32
    labels   : (N,)      int64   — class index (see MMFIT_CLASSES / CLASS_TO_IDX)
    subjects : (N,)      object  — subject ID per window
    """
    all_windows:  List[np.ndarray] = []
    all_labels:   List[int]        = []
    all_subjects: List[str]        = []

    for subj in subjects:
        subj_dir   = os.path.join(data_dir, subj)
        label_path = os.path.join(subj_dir, f"{subj}_labels.csv")

        if not os.path.exists(label_path):
            continue

        loaded      = load_subject(subj_dir, sensor_names)
        data_dense  = loaded["data"]       # (F, C)
        frame_start = loaded["frame_start"]

        F      = len(data_dense)
        starts = np.arange(0, F - window_frames + 1, stride_frames)
        data   = np.stack([data_dense[i : i + window_frames] for i in starts])

        segments = _parse_labels(label_path, frame_start)

        # Build a per-frame label array for the whole session (default: null)
        frame_labels = np.full(F, NULL_IDX, dtype=np.int64)
        for seg_start, seg_end, cls in segments:
            s = max(seg_start, 0)
            e = min(seg_end, F)
            if s < e:
                frame_labels[s:e] = CLASS_TO_IDX[cls]

        # Majority vote per window: count occurrences of each class in [start, start+W)
        # Using a (N_windows, N_classes) count matrix built via bincount over slices
        n_wins = len(starts)
        counts = np.zeros((n_wins, len(MMFIT_CLASSES)), dtype=np.int32)
        for i, s in enumerate(starts):
            bc = np.bincount(frame_labels[s : s + window_frames],
                             minlength=len(MMFIT_CLASSES))
            counts[i] = bc
        win_labels = counts.argmax(axis=1).astype(np.int64)

        all_windows.append(data)
        all_labels.append(win_labels)
        all_subjects.extend([subj] * len(starts))

    windows  = np.concatenate(all_windows,  axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0)
    subjects = np.array(all_subjects)
    return windows, labels, subjects
