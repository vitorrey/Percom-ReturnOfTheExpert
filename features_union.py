"""
features_union.py
-----------------
A-priori UNION of the three hand-crafted banks (statistical + biomechanical +
spectral-PCA), as a fourth feature-regression target (`featunion`). Tests whether
regressing a larger, more diverse target set beats the individual banks — a
robustness point, NOT a performance-tuned "optimal" set: the union is fixed in
advance and we only remove EXACT duplicates.

Duplicate handling: by inspection the only exact-value overlap across the three
banks is featgoogle's per-axis `axis_mean`, which equals the per-axis `mean`
already in the statistical bank (both are x.mean over time). The invariant bank
computes its stats on derived orientation-invariant signals, so it shares no
columns with the raw-axis banks. We drop the 3 axis_mean columns per sensor
triplet and concatenate the rest.

compute_union_features(windows, fs) -> (N, D) float32, windows (N, W, C),
C a multiple of 3.
"""
import numpy as np

from features_statistical import compute_statistical_features
from features_invariant import compute_invariant_features
from features_google import compute_google_features, GOOGLE_FEATURE_NAMES


def compute_union_features(windows: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    assert x.ndim == 3, f"Expected (N, W, C), got {x.shape}"
    N, W, C = x.shape
    assert C % 3 == 0, f"C must be a multiple of 3, got {C}"

    fstat = compute_statistical_features(x, fs)
    finv = compute_invariant_features(x, fs)
    fg = compute_google_features(x, fs)

    # Drop featgoogle axis_mean (first 3 columns per triplet) — exact duplicate
    # of the statistical bank's per-axis mean.
    g = len(GOOGLE_FEATURE_NAMES)                      # 11 columns per triplet
    drop = {k * g + j for k in range(C // 3) for j in range(3)}
    keep = [i for i in range(fg.shape[1]) if i not in drop]
    fg = fg[:, keep]

    out = np.concatenate([fstat, finv, fg], axis=1)
    return np.nan_to_num(out.astype(np.float32))
