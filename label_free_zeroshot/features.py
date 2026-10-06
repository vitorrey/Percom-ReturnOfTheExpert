"""
features.py
-----------
Physically-grounded feature extraction from accelerometer windows.

The eight scalar features used for pre-training and zero-shot classification:

    rms              — mean RMS acceleration magnitude (overall motion intensity)
    speed_mean       — mean detrended integrated speed (m/s)
    distance_total   — total path length from double-integrated linear acceleration (m)
    impact_rate      — fraction of frames where ||acc|| > 2g (≈19.6 m/s²)
    tilt_mean        — mean tilt angle from mean gravity direction (rad)
    tilt_variability — std of tilt deviation within the window (rad)
    angular_range    — max cumulative rotation from window start (rad)
    angular_rate     — mean absolute angular rate across axes (rad/s)

Usage
-----
    from features import compute_feature_profile, discretize_profile, thresholds_from_data

    profile = compute_feature_profile(acc, fs=50.0)   # -> dict of 8 scalars
    thresholds = thresholds_from_data(list_of_profiles)
    levels = discretize_profile(profile, thresholds)  # -> low/medium/high
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lowpass(data: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass filter along axis 0."""
    nyq = 0.5 * fs
    wc = min(cutoff_hz / nyq, 0.99)
    b, a = butter(2, wc, btype="low")
    return filtfilt(b, a, data, axis=0)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize rows of a (..., 3) array to unit vectors."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Element-wise angle (rad) between paired rows of two (N, 3) arrays. Returns (N,)."""
    cos_theta = np.clip((_unit(u) * _unit(v)).sum(axis=1), -1.0, 1.0)
    return np.arccos(cos_theta)


def _gravity(acc: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Estimate the gravity vector via low-pass filtering. Returns (W, 3)."""
    return _lowpass(acc, fs, cutoff_hz)


# ---------------------------------------------------------------------------
# The eight features (scalar summaries of a window)
# ---------------------------------------------------------------------------

def compute_feature_profile(acc: np.ndarray, fs: float,
                             impact_threshold: float = 19.6) -> Dict[str, float]:
    """
    Compute the eight scalar features for one accelerometer window.

    Args:
        acc              : (W, 3) raw accelerometer signal [ax, ay, az] in m/s²
        fs               : sampling frequency in Hz
        impact_threshold : threshold in m/s² for impact detection (default 2g ≈ 19.6)

    Returns:
        dict with keys: rms, speed_mean, distance_total, impact_rate,
                        tilt_mean, tilt_variability, angular_range, angular_rate
    """
    assert acc.ndim == 2 and acc.shape[1] == 3, \
        f"Expected acc shape (W, 3), got {acc.shape}"

    g = _gravity(acc, fs)
    lin = acc - g                                           # linear acceleration

    # ── motion features ────────────────────────────────────────────────────
    # rms: mean RMS across the three axes
    rms_val = float(np.sqrt(np.mean(acc ** 2, axis=0)).mean())

    # speed_mean: integrate ||linear acc|| then detrend
    lin_mag = np.linalg.norm(lin, axis=1)                  # (W,)
    vel = np.cumsum(lin_mag) / fs
    t = np.arange(len(vel), dtype=float)
    vel_detrended = vel - np.polyval(np.polyfit(t, vel, 1), t)
    speed_mean = float(np.abs(vel_detrended).mean())

    # distance_total: double-integrate per-axis linear acc, detrend each axis
    vel3 = np.cumsum(lin, axis=0) / fs                     # (W, 3)
    for i in range(3):
        vel3[:, i] -= np.polyval(np.polyfit(t, vel3[:, i], 1), t)
    pos = np.cumsum(vel3, axis=0) / fs                     # (W, 3)
    distance_total = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    # impact_rate: fraction of frames where ||acc|| > threshold
    impact_rate = float((np.linalg.norm(acc, axis=1) > impact_threshold).mean())

    # ── tilt / angle features ───────────────────────────────────────────────
    # tilt_mean + tilt_variability: angle between instantaneous and mean gravity
    mean_g = g.mean(axis=0, keepdims=True)
    tilt = _angle_between(g, np.broadcast_to(mean_g, g.shape))  # (W,)
    tilt_mean = float(tilt.mean())
    tilt_variability = float(tilt.std())

    # angular_range: max cumulative rotation from window start
    ref = np.broadcast_to(g[0:1], g.shape)
    angular_range = float(_angle_between(g, ref).max())

    # angular_rate: mean |d(per-axis tilt)/dt|
    g_mag = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    angles = np.arcsin(np.clip(g / g_mag, -1.0, 1.0))     # (W, 3)
    d_angles = np.abs(np.diff(angles, axis=0)) * fs        # (W-1, 3) rad/s
    angular_rate = float(d_angles.mean())

    return {
        "rms":             rms_val,
        "speed_mean":      speed_mean,
        "distance_total":  distance_total,
        "impact_rate":     impact_rate,
        "tilt_mean":       tilt_mean,
        "tilt_variability": tilt_variability,
        "angular_range":   angular_range,
        "angular_rate":    angular_rate,
    }


# ---------------------------------------------------------------------------
# Utilities for zero-shot / LLM-profile matching
# ---------------------------------------------------------------------------

def thresholds_from_data(profiles: List[Dict[str, float]]) -> Dict[str, tuple]:
    """
    Compute per-feature (33rd, 67th) percentile thresholds from a list of
    scalar profiles, giving balanced low/medium/high bins.

    Args:
        profiles : list of dicts returned by compute_feature_profile()

    Returns:
        thresholds dict suitable for discretize_profile()
    """
    keys = list(profiles[0].keys())
    values = {k: np.array([p[k] for p in profiles]) for k in keys}
    return {
        k: (float(np.percentile(v, 33)), float(np.percentile(v, 67)))
        for k, v in values.items()
    }


def discretize_profile(profile: Dict[str, float],
                       thresholds: Dict[str, tuple]) -> Dict[str, str]:
    """
    Convert a scalar feature profile to low/medium/high levels.

    Args:
        profile    : output of compute_feature_profile()
        thresholds : dict of feature -> (low_hi, medium_hi) boundary values
                     (from thresholds_from_data)

    Returns:
        dict of feature -> "low" | "medium" | "high"
    """
    levels = {}
    for feat, value in profile.items():
        if feat not in thresholds:
            continue
        lo, hi = thresholds[feat]
        if value < lo:
            levels[feat] = "low"
        elif value < hi:
            levels[feat] = "medium"
        else:
            levels[feat] = "high"
    return levels
