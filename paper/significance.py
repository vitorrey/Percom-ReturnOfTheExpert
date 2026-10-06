#!/usr/bin/env python3
"""
significance.py — paired Wilcoxon signed-rank tests quoted in the paper
(Sections V-A and V-B), computed over the 18 downstream datasets from the stored
results. Run from the repository root:  python paper/significance.py

  1. Average axis: each feature-regression bank vs MTL and TimeChannel.
  2. Average axis: all pairwise comparisons between the four banks.
  3. Few-shot: each bank vs TimeChannel and MTL at every label ratio
     (mean difference, datasets won, p-value).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import aggregate_results as agg  # noqa: E402

BANKS = {"featstat_norot": "statistical", "featinv_norot": "biomechanical",
         "featgoogle_norot": "spectral-PCA", "featunion_norot": "union"}
BASELINES = {"timechannel": "TimeChannel", "mtl": "MTL"}
RATIOS = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]

jobs, res, _, _ = agg.collect(Path("results/benchmark"))
models = list(BANKS) + list(BASELINES)
avg = {m: {d["dataset"]: d["summary"]["mean_f1"] for d in res["average"][m].values()} for m in models}
fs = {m: {} for m in models}
for m in models:
    for d in res["fewshot"][m].values():
        fs[m].setdefault(str(d["hyperparameters"]["data_ratio"]), {})[d["dataset"]] = d["summary"]["mean_f1"]


def test(a, b):
    """Paired test on the datasets common to a and b -> (mean diff, wins, n, p)."""
    ds = sorted(set(a) & set(b))
    x, y = np.array([a[d] for d in ds]), np.array([b[d] for d in ds])
    return (x - y).mean(), int((x > y).sum()), len(ds), wilcoxon(x, y).pvalue


def show(label, a, b):
    diff, wins, n, p = test(a, b)
    print(f"  {label:42s} diff={diff:+.3f}  wins={wins:2d}/{n}  p={p:.4f}")


print("1. Average axis: bank vs baseline")
for m, name in BANKS.items():
    for bl, bname in BASELINES.items():
        show(f"{name} vs {bname}", avg[m], avg[bl])

print("\n2. Average axis: bank vs bank")
for m1, m2 in combinations(BANKS, 2):
    show(f"{BANKS[m1]} vs {BANKS[m2]}", avg[m1], avg[m2])

print("\n3. Few-shot: bank vs baseline per label ratio")
for r in RATIOS:
    print(f" ratio {float(r) * 100:g}%")
    for m, name in BANKS.items():
        for bl, bname in BASELINES.items():
            show(f"{name} vs {bname}", fs[m][r], fs[bl][r])
