"""
features_pim.py
---------------
Accelerometer-only pseudo-label features of PIM, implemented following the
formulation in the PIM paper (Nshimyimana et al., ABC 2025).

Three feature types (all computed from raw acc, no gyroscope):

  angles   : mean roll, pitch, fake-yaw per sensor  (3 scalars / sensor)
  motion   : path length from raw acc integration    (1 scalar  / sensor)
  symmetry : DTW distance between paired limbs       (1 scalar  / pair)

Discretization
--------------
Each scalar is independently binned into N_BINS=11 uniform bins fitted
on the training windows.  Bin indices are integer class labels (0–10)
used with CrossEntropyLoss.

Usage
-----
  disc = PIMDiscretizer(fs=50)
  disc.fit(train_windows, dataset="wear")          # (N, W, C)
  targets = disc.transform(windows, dataset="wear") # list of np arrays
  disc.save("runs/pim_disc_wear.pkl")
"""

import pickle
import numpy as np
from scipy import signal
from sklearn.preprocessing import KBinsDiscretizer

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

N_BINS = 11

# Paired sensor indices for symmetry (left vs right limb)
# Format: list of (sensor_idx_a, sensor_idx_b) tuples
# WEAR sensors:  right_arm(0), right_leg(1), left_leg(2), left_arm(3)
# MMFIT sensors: sw_l_acc(0),  sw_r_acc(1),  sp_r_acc(2), eb_l_acc(3)
# PAMAP2 sensors: hand(0), chest(1), ankle(2)
SYMMETRY_PAIRS = {
    "wear":   [(0, 3),   # right_arm vs left_arm  (= sym_sw)
               (1, 2)],  # right_leg vs left_leg  (= sym_sp)
    "mmfit":  [(0, 1)],  # sw_l vs sw_r           (= sym_sw)
    "pamap2": [],
}

N_SENSORS = {"wear": 4, "mmfit": 4, "pamap2": 3}

# (n_angle_heads, n_sym_heads, n_motion_heads) per dataset
PS_HEADS = {
    "wear":   (4, 2, 4),
    "mmfit":  (4, 1, 4),
    "pamap2": (3, 0, 3),
}


# -----------------------------------------------------------------------
# Signal utilities
# -----------------------------------------------------------------------

def _butter_lowpass(x: np.ndarray, cutoff: float, fs: float,
                    order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    wn  = min(cutoff / nyq, 0.999)
    sos = signal.butter(order, wn, btype="low", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _butter_bandpass(x: np.ndarray, lo: float, hi: float, fs: float,
                     order: int = 2) -> np.ndarray:
    nyq  = 0.5 * fs
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    sos  = signal.butter(order, [lo_n, hi_n], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


# -----------------------------------------------------------------------
# Per-sensor feature functions   (input: (W, 3) acc for one sensor)
# -----------------------------------------------------------------------

def gravity_angles(acc: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate roll, pitch, fake-yaw from gravity direction (acc-only).
    Returns mean over the window: shape (3,).
    """
    acc_f = _butter_lowpass(acc, cutoff=0.5, fs=fs)
    x, y, z = acc_f.T
    roll      = np.arctan2(y, z)
    pitch     = np.arctan2(x, z)
    yaw_fake  = np.arctan2(y, x)
    return np.array([np.mean(roll), np.mean(pitch), np.mean(yaw_fake)],
                    dtype=np.float32)


def motion_path(acc: np.ndarray, fs: float) -> float:
    """
    Path length from numerically integrated raw acceleration (no tilt comp).
    Returns a scalar.
    """
    period = 1.0 / fs
    try:
        acc_f = _butter_bandpass(acc, lo=0.4, hi=min(20.0, fs / 2 - 1), fs=fs)
    except Exception:
        acc_f = acc
    vels = np.cumsum(acc_f, axis=0) * period + acc_f * period
    diff = vels[1:] - vels[:-1]
    return float(np.sqrt(np.sum(diff ** 2)))


def symmetry_dtw(acc_a: np.ndarray, acc_b: np.ndarray, fs: float) -> float:
    """
    DTW distance between RMS-normalised acceleration of two sensors.
    Uses a simple O(n²) DTW to avoid the dtaidistance dependency.
    """
    def _rms(a): return np.sqrt(np.sum(a ** 2, axis=1))

    ra = _rms(_butter_lowpass(acc_a, cutoff=8.0, fs=fs))
    rb = _rms(_butter_lowpass(acc_b, cutoff=8.0, fs=fs))

    n, m = len(ra), len(rb)
    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ra[i - 1] - rb[j - 1])
            dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j],
                                       dtw_mat[i, j - 1],
                                       dtw_mat[i - 1, j - 1])
    return float(dtw_mat[n, m])


# -----------------------------------------------------------------------
# Batch feature extraction
# -----------------------------------------------------------------------

def extract_features(windows: np.ndarray, dataset: str, fs: float) -> dict:
    """
    Extract all PIM features for a batch of windows.

    Parameters
    ----------
    windows : (N, W, C) float32  — C = n_sensors * 3
    dataset : "wear" | "mmfit" | "pamap2"
    fs      : sampling rate in Hz

    Returns
    -------
    dict with keys:
      "angles"    : (N, n_sensors, 3)   float32
      "motion"    : (N, n_sensors)       float32
      "symmetry"  : (N, n_pairs)         float32  (n_pairs may be 0)
    """
    N, W, C = windows.shape
    n_sen   = N_SENSORS[dataset]
    pairs   = SYMMETRY_PAIRS[dataset]

    angles   = np.zeros((N, n_sen, 3),       dtype=np.float32)
    motion   = np.zeros((N, n_sen),           dtype=np.float32)
    symmetry = np.zeros((N, len(pairs)),      dtype=np.float32)

    for i, win in enumerate(windows):
        # win: (W, C)  with sensors stacked as [s0_x,s0_y,s0_z, s1_x, ...]
        for s in range(n_sen):
            acc_s = win[:, s * 3: s * 3 + 3]
            angles[i, s] = gravity_angles(acc_s, fs)
            motion[i, s] = motion_path(acc_s, fs)
        for p, (a, b) in enumerate(pairs):
            acc_a = win[:, a * 3: a * 3 + 3]
            acc_b = win[:, b * 3: b * 3 + 3]
            symmetry[i, p] = symmetry_dtw(acc_a, acc_b, fs)

    return {"angles": angles, "motion": motion, "symmetry": symmetry}


# -----------------------------------------------------------------------
# Discretizer
# -----------------------------------------------------------------------

class PIMDiscretizer:
    """
    Fits one KBinsDiscretizer (onehot-dense, uniform) per scalar feature
    channel, as in the PIM formulation.

    transform() returns float one-hot arrays matching the shapes expected
    by the BCE / CrossEntropy heads:
      angles   : list of (3, N_BINS) arrays  — one per sensor
      motion   : list of (N_BINS,)   arrays  — one per sensor
      symmetry : list of (N_BINS,)   arrays  — one per pair
    """

    def __init__(self, fs: float, n_bins: int = N_BINS):
        self.fs     = fs
        self.n_bins = n_bins
        self.scalers: list = []   # flat list, one scaler per raw feature channel
                                   # order: [ang_s0_roll, ang_s0_pitch, ang_s0_yaw,
                                   #         ang_s1_..., ..., mot_s0, mot_s1, ...,
                                   #         sym_p0, sym_p1, ...]

    def _make_disc(self):
        return KBinsDiscretizer(n_bins=self.n_bins,
                                encode="onehot-dense",
                                strategy="uniform",
                                subsample=None)

    def fit(self, windows: np.ndarray, dataset: str) -> "PIMDiscretizer":
        """Fit one scaler per raw scalar channel, matching PIM's per-column fit."""
        feats = extract_features(windows, dataset, self.fs)
        n_sen  = feats["angles"].shape[1]
        n_pair = feats["symmetry"].shape[1]

        self.scalers = []
        for s in range(n_sen):
            for d in range(3):
                sc = self._make_disc()
                sc.fit(feats["angles"][:, s, d].reshape(-1, 1))
                self.scalers.append(sc)
        for s in range(n_sen):
            sc = self._make_disc()
            sc.fit(feats["motion"][:, s].reshape(-1, 1))
            self.scalers.append(sc)
        for p in range(n_pair):
            sc = self._make_disc()
            sc.fit(feats["symmetry"][:, p].reshape(-1, 1))
            self.scalers.append(sc)

        self._n_sen  = n_sen
        self._n_pair = n_pair
        return self

    def transform_window(self, window: np.ndarray, dataset: str) -> list:
        """
        Transform a single window (W, C) into the flat pseudo-label list
        matching PIM's ps ordering per dataset.

        Returns a list of np.float32 arrays in the same order as ps[dataset]:
          WEAR/MMFIT : [ang_bh, ang_spl, ang_spr, ang_swl, ang_swr,
                        sym_sp, sym_sw,
                        mot_bh, mot_spl, mot_spr, mot_swl, mot_swr]
          PAMAP2     : [ang_s0, ang_s1, ang_s2, mot_s0, mot_s1, mot_s2]
        """
        feats = extract_features(window[np.newaxis], dataset, self.fs)

        n_sen  = self._n_sen
        n_pair = self._n_pair
        sc_idx = 0

        # --- angles (one (3, N_BINS) per sensor) ---
        ang_out = []
        for s in range(n_sen):
            rows = []
            for d in range(3):
                oh = self.scalers[sc_idx].transform(
                    feats["angles"][0, s, d].reshape(1, 1))[0]
                rows.append(oh)
                sc_idx += 1
            ang_out.append(np.stack(rows).astype(np.float32))  # (3, N_BINS)

        # --- motion (one (N_BINS,) per sensor) ---
        mot_out = []
        for s in range(n_sen):
            oh = self.scalers[sc_idx].transform(
                feats["motion"][0, s].reshape(1, 1))[0]
            mot_out.append(oh.astype(np.float32))  # (N_BINS,)
            sc_idx += 1

        # --- symmetry (one (N_BINS,) per pair) ---
        sym_out = []
        for p in range(n_pair):
            oh = self.scalers[sc_idx].transform(
                feats["symmetry"][0, p].reshape(1, 1))[0]
            sym_out.append(oh.astype(np.float32))  # (N_BINS,)
            sc_idx += 1

        return _build_ps_list(ang_out, sym_out, mot_out, dataset)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PIMDiscretizer":
        with open(path, "rb") as f:
            return pickle.load(f)


# -----------------------------------------------------------------------
# Pseudo-label list ordering per dataset
# -----------------------------------------------------------------------

def _build_ps_list(ang: list, sym: list, mot: list, dataset: str) -> list:
    """
    Build the flat pseudo-label list, ordered to match PS_HEADS indexing.

    Layout per dataset  (matches PS_HEADS order: angles | sym | motion):

    WEAR   (4,2,4): [spl, spr, swl, swr | sym_sp, sym_sw | mot_spl, spr, swl, swr]
    MMFIT  (4,1,4): [ebl, spr, swl, swr | sym_sw         | mot_ebl, spr, swl, swr]
    PAMAP2 (3,0,3): [s0,  s1,  s2       |                | mot_s0,  s1,  s2      ]

    Sensor index reference
    ----------------------
    WEAR  : right_arm=0, right_leg=1, left_leg=2, left_arm=3
    MMFIT : sw_l=0, sw_r=1, sp_r=2, eb_l=3
    PAMAP2: hand=0, chest=1, ankle=2

    Symmetry pair index reference
    ------------------------------
    WEAR  : sym[0]=(0,3)=sw  sym[1]=(1,2)=sp
    MMFIT : sym[0]=(0,1)=sw
    """
    if dataset == "wear":
        return [
            ang[2],  # spl  = left_leg
            ang[1],  # spr  = right_leg
            ang[3],  # swl  = left_arm
            ang[0],  # swr  = right_arm
            sym[1],  # sym_sp = (right_leg, left_leg)
            sym[0],  # sym_sw = (right_arm, left_arm)
            mot[2],  # mot_spl
            mot[1],  # mot_spr
            mot[3],  # mot_swl
            mot[0],  # mot_swr
        ]
    elif dataset == "mmfit":
        return [
            ang[3],  # ebl  = head/ear
            ang[2],  # spr  = hip
            ang[0],  # swl  = wrist_l
            ang[1],  # swr  = wrist_r
            sym[0],  # sym_sw = (wrist_l, wrist_r)
            mot[3],  # mot_ebl
            mot[2],  # mot_spr
            mot[0],  # mot_swl
            mot[1],  # mot_swr
        ]
    elif dataset == "pamap2":
        return ang + mot   # [ang_s0, ang_s1, ang_s2, mot_s0, mot_s1, mot_s2]
    else:
        raise ValueError(dataset)
