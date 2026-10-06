"""
features_statistical.py
-----------------------
Statistical feature bank (paper Section III-B, "Statistical", 63-d per sensor;
model name `featstat`): generic statistical / signal-processing descriptors of
the raw accelerometer axes, of the same family that automated extractors such
as tsfresh produce: time-domain moments, dispersion, energy, zero-crossings,
frequency-domain summaries and cross-axis correlations.

Two extractors are provided:

  compute_statistical_features(windows, fs)
      Fast, fully-vectorised NumPy implementation (default; no extra deps).

  compute_tsfresh_features(windows, fs, params="minimal")
      Thin wrapper around the real tsfresh library (optional; slow). Use to
      show the built-in bank tracks a bona-fide tsfresh feature set.

Both take (N, W, C) windows (C = n_sensors * 3 accelerometer axes) and return
a dense (N, D) float32 matrix (regression target in pretrain.py; classifier
input in evaluate_features_benchmark.py).
"""

import numpy as np
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------------
# Built-in vectorised statistical bank
# ---------------------------------------------------------------------------

# Per-channel scalar descriptors, in output order.
_TIME_FEATURE_NAMES = [
    "mean", "std", "min", "max", "range", "median", "iqr", "mad",
    "rms", "abs_energy", "zero_cross_rate", "mean_abs_diff", "skew", "kurtosis",
    "var", "sma", "crest",
]
_FREQ_FEATURE_NAMES = ["dom_freq", "spec_centroid", "spec_entropy"]
STAT_FEATURE_NAMES = _TIME_FEATURE_NAMES + _FREQ_FEATURE_NAMES


def compute_statistical_features(windows: np.ndarray, fs: float,
                                 keep=None) -> np.ndarray:
    """
    Vectorised statistical feature bank.

    Args:
        windows : (N, W, C) accelerometer windows in m/s²
        fs      : sampling frequency in Hz (only used for frequency features)
        keep    : optional list of feature names (subset of STAT_FEATURE_NAMES)
                  to compute per channel; None = all. When given, the per-sensor
                  axis-correlation block is skipped (used by the invariant bank
                  to control feature count).

    Returns:
        (N, D) float32 feature matrix, D = C * (len(keep) or len(STAT_FEATURE_NAMES))
        (+ 3 axis-correlation features per sensor when C is a multiple of 3 and
        keep is None).
    """
    x = np.asarray(windows, dtype=np.float64)
    assert x.ndim == 3, f"Expected (N, W, C), got {x.shape}"
    N, W, C = x.shape

    mean = x.mean(axis=1)                              # (N, C)
    std  = x.std(axis=1)
    q25  = np.percentile(x, 25, axis=1)
    q75  = np.percentile(x, 75, axis=1)
    centered = x - mean[:, None, :]

    feats = [
        mean,
        std,
        x.min(axis=1),
        x.max(axis=1),
        np.ptp(x, axis=1),                             # range
        np.median(x, axis=1),
        q75 - q25,                                     # iqr
        np.mean(np.abs(centered), axis=1),             # mean abs deviation
        np.sqrt((x ** 2).mean(axis=1)),                # rms
        (x ** 2).sum(axis=1),                          # abs energy
        (np.diff(np.sign(centered), axis=1) != 0).mean(axis=1),  # zero-cross rate
        np.abs(np.diff(x, axis=1)).mean(axis=1),       # mean abs diff (mobility-ish)
        skew(x, axis=1),
        kurtosis(x, axis=1),
    ]

    # ── extra amplitude descriptors (var / signal-magnitude-area / crest) ──
    _rms = np.sqrt((x ** 2).mean(axis=1))
    feats += [
        x.var(axis=1),                                 # variance
        np.abs(x).mean(axis=1),                        # signal magnitude area
        np.abs(x).max(axis=1) / (_rms + 1e-12),        # crest factor
    ]

    # ── frequency domain (detrended) ──────────────────────────────────────
    spec  = np.abs(np.fft.rfft(centered, axis=1))      # (N, F, C)
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)             # (F,)
    power = spec ** 2
    psum  = power.sum(axis=1) + 1e-12                   # (N, C)

    dom_freq = freqs[np.argmax(spec, axis=1)]          # (N, C)
    centroid = (power * freqs[None, :, None]).sum(axis=1) / psum
    pnorm    = power / psum[:, None, :]
    entropy  = -(pnorm * np.log(pnorm + 1e-12)).sum(axis=1)
    feats += [dom_freq, centroid, entropy]

    if keep is not None:
        idx = [STAT_FEATURE_NAMES.index(k) for k in keep]
        feats = [feats[i] for i in idx]

    out = np.stack(feats, axis=-1).reshape(N, -1)      # (N, C * K)

    # ── per-sensor cross-axis correlations (xy, xz, yz) ───────────────────
    if C % 3 == 0 and keep is None:
        xs = centered.reshape(N, W, C // 3, 3)
        corrs = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = xs[..., i], xs[..., j]              # (N, W, S)
            num = (a * b).sum(axis=1)
            den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12
            corrs.append(num / den)                    # (N, S)
        out = np.concatenate([out] + corrs, axis=1)

    return np.nan_to_num(out.astype(np.float32))


# ---------------------------------------------------------------------------
# Optional: real tsfresh
# ---------------------------------------------------------------------------

def compute_tsfresh_features(windows: np.ndarray, fs: float,
                             params: str = "minimal") -> np.ndarray:
    """
    Extract features with the tsfresh library (optional dependency).

    NOTE: tsfresh builds a long-format table of N*W*C rows and is far slower
    than compute_statistical_features. `params` selects the feature set:
    "minimal" (MinimalFCParameters) or "efficient" (EfficientFCParameters).

    Args:
        windows : (N, W, C) windows
        fs      : sampling frequency (unused by tsfresh directly; kept for a
                  uniform signature)

    Returns:
        (N, D) float32 feature matrix with NaN/constant columns dropped.
    """
    try:
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import (
            MinimalFCParameters, EfficientFCParameters)
        from tsfresh.utilities.dataframe_functions import impute
    except ImportError as e:
        raise SystemExit(
            "tsfresh not installed. Run `pip install tsfresh` or use "
            "--mode stat for the built-in statistical bank.") from e

    x = np.asarray(windows, dtype=np.float64)
    N, W, C = x.shape
    fc = MinimalFCParameters() if params == "minimal" else EfficientFCParameters()

    # Long format: one row per (window, channel, timestep).
    ids  = np.repeat(np.arange(N), W)
    time = np.tile(np.arange(W), N)
    frames = []
    for c in range(C):
        frames.append(pd.DataFrame({
            "id": ids, "time": time,
            "kind": f"ch{c}", "value": x[:, :, c].reshape(-1),
        }))
    long_df = pd.concat(frames, ignore_index=True)

    feat_df = extract_features(
        long_df, column_id="id", column_sort="time",
        column_kind="kind", column_value="value",
        default_fc_parameters=fc, disable_progressbar=True, n_jobs=0)
    impute(feat_df)
    feat_df = feat_df.reindex(sorted(feat_df.index))   # restore window order
    return feat_df.to_numpy(dtype=np.float32)
