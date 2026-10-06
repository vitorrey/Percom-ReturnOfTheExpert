"""
features_google.py
------------------
Spectral-PCA feature bank (paper Section III-B, 11-d per sensor; model name
`featgoogle`): a compact descriptor set in the style of production wearable
pipelines (energy / PCA / zero-crossing / moment descriptors). Used as an
additional feature-regression pretraining target to test whether
the few-shot advantage of feature-regression is robust to the *choice* of
feature family, not just to featstat vs featinv.

The distinctive members are built on the 1st principal component of the 3-axis
window (a data-driven dominant-motion axis) — a middle ground between featstat
(raw-axis stats) and featinv (hand-designed orientation-invariant signals).

compute_google_features(windows, fs) -> (N, D) float32
  windows : (N, W, C), C a multiple of 3 (one accelerometer triplet per 3 chans)
  D       : 11 features per triplet, in GOOGLE_FEATURE_NAMES order

Notes on fidelity: `steps` is an APPROXIMATION (thresholded peak count on the
magnitude, not a validated pedometer); at 5-second windows a faithful step count
is ill-defined, and winsorisation of the target tames the resulting noise. All
ops are fully vectorised (no per-window Python loop) so this is cheap enough to
compute as a per-batch regression target during pretraining.
"""
import numpy as np
from scipy.stats import kurtosis

EPS = 1e-8

_PER_TRIPLET_NAMES = [
    "axis_mean_x", "axis_mean_y", "axis_mean_z",
    "log_energy", "log_energy_ratio", "cov_cond",
    "kurtosis_mag", "jerk_autocorr",
    "zero_cross_mean", "zero_cross_std", "steps",
]


def _zero_cross_timing(pc1: np.ndarray, fs: float):
    """Mean/std of inter-zero-crossing intervals (seconds) of each window's 1st PC.

    Fully vectorised via a flat (row, col) crossing list + segment reductions,
    so there is no Python loop over windows.
    """
    N = pc1.shape[0]
    sign = pc1 >= 0                                   # (N, W)
    cross = sign[:, 1:] != sign[:, :-1]               # (N, W-1) crossing at t->t+1
    rows, cols = np.nonzero(cross)                    # crossing coordinates
    zc_mean = np.zeros(N, dtype=np.float64)
    zc_std = np.zeros(N, dtype=np.float64)
    if rows.size >= 2:
        same = np.diff(rows) == 0                     # consecutive crossings, same window
        gap = np.diff(cols)[same].astype(np.float64)  # inter-crossing gap in samples
        grow = rows[1:][same]                         # window id per gap
        cnt = np.bincount(grow, minlength=N).astype(np.float64)
        s = np.bincount(grow, weights=gap, minlength=N)
        ss = np.bincount(grow, weights=gap * gap, minlength=N)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_gap = np.where(cnt > 0, s / cnt, 0.0)
            var_gap = np.where(cnt > 0, ss / cnt - mean_gap ** 2, 0.0)
        zc_mean = mean_gap / fs
        zc_std = np.sqrt(np.maximum(var_gap, 0.0)) / fs
    return zc_mean, zc_std


def _triplet_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Features for one accelerometer triplet. x: (N, W, 3) -> (N, 11)."""
    N, W, _ = x.shape

    axis_mean = x.mean(axis=1)                        # (N, 3)
    mag = np.sqrt((x ** 2).sum(axis=2))               # (N, W) resultant magnitude

    rms_axis = np.sqrt((x ** 2).mean(axis=1))         # (N, 3)
    log_energy = np.log(rms_axis.sum(axis=1) + EPS)   # (N,)

    # PCA per window on the centred 3-axis signal (batched 3x3 eigendecomposition).
    xc = x - x.mean(axis=1, keepdims=True)            # (N, W, 3)
    cov = np.einsum("nwi,nwj->nij", xc, xc) / max(W - 1, 1)   # (N, 3, 3)
    evals, evecs = np.linalg.eigh(cov)                # ascending; evecs cols
    lam1, lam3 = evals[:, -1], evals[:, 0]
    v1 = evecs[:, :, -1]                              # top eigenvector (N, 3)
    cov_cond = lam1 / (lam3 + EPS)                    # covariance condition number

    pc1 = np.einsum("nwi,ni->nw", xc, v1)            # 1st principal component (N, W)
    energy_pc1 = (pc1 ** 2).sum(axis=1)               # (N,)
    total_energy = (xc ** 2).sum(axis=(1, 2))         # (N,)
    log_energy_ratio = np.log(energy_pc1 / (total_energy + EPS) + EPS)

    # Jerk (derivative of 1st PC) lag-1 autocorrelation, normalised by PC energy.
    jerk = np.diff(pc1, axis=1)                        # (N, W-1)
    ac1 = (jerk[:, :-1] * jerk[:, 1:]).sum(axis=1)     # (N,)
    jerk_autocorr = ac1 / (energy_pc1 + EPS)

    kurt_mag = kurtosis(mag, axis=1)                  # (N,) kurtosis of magnitude

    zc_mean, zc_std = _zero_cross_timing(pc1, fs)

    # Approximate step count: thresholded local maxima of the magnitude.
    thr = mag.mean(1, keepdims=True) + 0.5 * mag.std(1, keepdims=True)
    peak = (mag[:, 1:-1] > mag[:, :-2]) & (mag[:, 1:-1] > mag[:, 2:]) & (mag[:, 1:-1] > thr)
    steps = peak.sum(axis=1).astype(np.float64)       # (N,)

    return np.stack([
        axis_mean[:, 0], axis_mean[:, 1], axis_mean[:, 2],
        log_energy, log_energy_ratio, cov_cond,
        kurt_mag, jerk_autocorr,
        zc_mean, zc_std, steps,
    ], axis=1)                                        # (N, 11)


def compute_google_features(windows: np.ndarray, fs: float) -> np.ndarray:
    """Spectral-PCA feature bank. (N, W, C) -> (N, 11 * C/3) float32."""
    x = np.asarray(windows, dtype=np.float64)
    assert x.ndim == 3, f"Expected (N, W, C), got {x.shape}"
    N, W, C = x.shape
    assert C % 3 == 0, f"C must be a multiple of 3, got {C}"

    blocks = [_triplet_features(x[:, :, 3 * k:3 * k + 3], fs) for k in range(C // 3)]
    out = np.concatenate(blocks, axis=1)
    return np.nan_to_num(out.astype(np.float32))


GOOGLE_FEATURE_NAMES = _PER_TRIPLET_NAMES
