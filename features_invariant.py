"""
features_invariant.py
---------------------
Biomechanical feature bank (paper Sections III-B and III-C, 63-d per sensor;
model name `featinv`): the statistical descriptor family computed on
orientation-INVARIANT physical signals rather than on the raw sensor axes.

Motivation
----------
The statistical bank (features_statistical.py) is computed on the
orientation-DEPENDENT x/y/z axes. This module derives rotation-invariant scalar
time-series from a window (gravity, linear acceleration, tilt, angular rate,
jerk, ...) and applies the statistical descriptors to each, so the bank is as
descriptive as the statistical one but invariant by construction.

Signal design (recovering the horizontal degree of freedom)
-----------------------------------------------------------
Acceleration is 3-DOF. Splitting it into a gravity-aligned (vertical) component
and a horizontal-magnitude scalar keeps only 2 DOF — it discards how the
horizontal vector *rotates* in its plane. The absolute horizontal heading is not
rotation-invariant, but the RATE at which the horizontal vector turns is. We add
that as `horiz_turn_rate`, recovering the third DOF invariantly.

  invariant_signal_batch(windows, fs) -> (N, W, K)
  compute_invariant_features(windows, fs) -> (N, D)   D = K * len(CURATED_STATS)
"""

import numpy as np
from scipy.signal import butter, filtfilt

from features_statistical import compute_statistical_features

EPS = 1e-9

# Rotation-invariant scalar signals (K = 7), chosen to preserve the acceleration
# content invariantly with little redundancy.
INVARIANT_SIGNAL_NAMES = [
    "lin_mag",          # ||acc - gravity||         (overall linear motion)
    "acc_vertical",     # acc . gravity_hat         (gravity-aligned component)
    "acc_horizontal",   # ||acc perpendicular to g||(horizontal magnitude)
    "horiz_turn_rate",  # turn rate of the horizontal vector (recovered DOF)
    "tilt",             # angle(g(t), mean g)
    "angular_rate",     # ||d gravity_hat|| * fs
    "jerk_mag",         # ||d acc|| * fs
]

# Curated, low-redundancy statistics per signal (9) -> 7 * 9 = 63 features,
# same ball-park as the 54-dim raw-axis statistical bank.
CURATED_STATS = [
    "mean", "std", "min", "max", "iqr", "rms",
    "abs_energy", "dom_freq", "spec_entropy",
]


def _lowpass_axis1(x: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Zero-phase Butterworth low-pass along the time axis (axis=1) of (N, W, C)."""
    nyq = 0.5 * fs
    wc = min(cutoff_hz / nyq, 0.99)
    b, a = butter(2, wc, btype="low")
    return filtfilt(b, a, x, axis=1)


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle (rad) between paired 3-vectors along the last axis. Shapes broadcast."""
    un = u / (np.linalg.norm(u, axis=-1, keepdims=True) + EPS)
    vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)
    return np.arccos(np.clip((un * vn).sum(axis=-1), -1.0, 1.0))


def _consecutive_turn(vec: np.ndarray, fs: float) -> np.ndarray:
    """Turn rate (rad/s) between consecutive rows of a (N, W, 3) vector series."""
    a, b = vec[:, 1:], vec[:, :-1]
    cos = (a * b).sum(axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS)
    rate = np.arccos(np.clip(cos, -1.0, 1.0)) * fs        # (N, W-1)
    return np.concatenate([rate[:, :1], rate], axis=1)    # pad to W


def invariant_signal_batch(windows: np.ndarray, fs: float) -> np.ndarray:
    """(N, W, 3) accelerometer windows -> (N, W, K) rotation-invariant signals."""
    X = np.asarray(windows, dtype=np.float64)
    assert X.ndim == 3 and X.shape[2] == 3, f"expected (N,W,3), got {X.shape}"

    g = _lowpass_axis1(X, fs)                                  # gravity (N,W,3)
    ghat = g / (np.linalg.norm(g, axis=-1, keepdims=True) + EPS)
    lin = X - g

    lin_mag = np.linalg.norm(lin, axis=-1)                    # (N,W)
    a_vert = (X * ghat).sum(axis=-1)                          # along gravity
    horiz_vec = X - a_vert[..., None] * ghat                  # (N,W,3) in tangent plane
    a_horiz = np.linalg.norm(horiz_vec, axis=-1)
    horiz_turn = _consecutive_turn(horiz_vec, fs)            # recovered 3rd DOF

    mean_g = g.mean(axis=1, keepdims=True)
    tilt = _angle(g, np.broadcast_to(mean_g, g.shape))
    ang_rate = _consecutive_turn(ghat, fs)
    jerk = np.linalg.norm(np.diff(X, axis=1), axis=-1) * fs
    jerk = np.concatenate([jerk[:, :1], jerk], axis=1)

    sig = np.stack([lin_mag, a_vert, a_horiz, horiz_turn,
                    tilt, ang_rate, jerk], axis=-1)          # (N,W,7)
    return np.nan_to_num(sig.astype(np.float32))


def compute_invariant_features(windows: np.ndarray, fs: float) -> np.ndarray:
    """(N, W, 3) -> (N, K*len(CURATED_STATS)) curated stats over invariant signals."""
    sig = invariant_signal_batch(windows, fs)                # (N, W, K)
    return compute_statistical_features(sig, fs, keep=CURATED_STATS)
