#!/usr/bin/env python3
"""
evaluate_features_benchmark.py
==============================
Features-alone reference (paper Table IV, Fig. 3b): how useful are the hand-crafted
features themselves for classification, compared with the pretrained network?

For each downstream dataset it classifies DIRECTLY on a hand-crafted feature bank
(featstat / featinv / featgoogle) with a Random Forest and a small MLP — NO
pretraining, CPU-only — under the *identical* HARBench protocol:
  - 4-fold user CV (test=[1,2]/[3,4]/[5,6]/[7,8]; train = users not in test/val,
    matching the network's fine-tuning set),
  - the same few-shot ratios (1/2/5/10/20/50%) with the same stratified
    (>=1 per class) subsample,
  - macro-F1, averaged over folds then over datasets.

Output drops next to the network's numbers so features-alone vs pretrained-network
can be compared per axis (average + few-shot curve).

Usage:
  python evaluate_features_benchmark.py --datasets uschad          # quick test
  python evaluate_features_benchmark.py                            # all 18
  python evaluate_features_benchmark.py --classifiers rf --ratios 0.01 0.05
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb
from src.data.dataloader import load_dataset
from features_statistical import compute_statistical_features
from features_invariant import compute_invariant_features
from features_google import compute_google_features

FS = 30.0
FOLDS = [
    {"test": [1, 2], "val": [3, 4]},
    {"test": [3, 4], "val": [5, 6]},
    {"test": [5, 6], "val": [7, 8]},
    {"test": [7, 8], "val": [1, 2]},
]
DEFAULT_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
BANK_FN = {
    "featstat": compute_statistical_features,
    "featinv": compute_invariant_features,
    "featgoogle": compute_google_features,
}
# dataset -> default sensors used by run_benchmark's Average eval.
DATASET_SENSORS = {ds: cfg["sensors"] for dom in rb.DATASETS.values()
                   for ds, cfg in dom.items()}
DOMAIN = {ds: dom for dom, dsets in rb.DATASETS.items() for ds in dsets}


def extract(bank: str, w: np.ndarray) -> np.ndarray:
    """Apply a feature bank per 3-axis sensor triplet and concatenate (matches
    the network's shared per-sensor encoding). w: (N, T, C), C multiple of 3."""
    fn = BANK_FN[bank]
    N, T, C = w.shape
    if C == 3:
        return fn(w, FS)
    return np.concatenate([fn(w[:, :, 3 * k:3 * k + 3], FS) for k in range(C // 3)], axis=1)


def subsample(Y_train: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    """Replicate create_dataloaders few-shot subsample: >=1 per class + random rest."""
    n = len(Y_train)
    k = max(1, int(n * ratio))
    sel = [rng.choice(np.where(Y_train == c)[0], 1)[0] for c in np.unique(Y_train)]
    rem = k - len(sel)
    if rem > 0:
        avail = list(set(range(n)) - set(sel))
        if avail:
            sel.extend(rng.choice(avail, min(rem, len(avail)), replace=False).tolist())
    return np.array(sel)


def winsor_standardize(Xtr, Xte):
    """Winsorise to train p1/p99 then standardise (tames heavy-tailed features
    like cov_cond/kurtosis; the same treatment used for the regression targets)."""
    lo = np.percentile(Xtr, 1, axis=0)
    hi = np.percentile(Xtr, 99, axis=0)
    Xtr = np.clip(Xtr, lo, hi)
    Xte = np.clip(Xte, lo, hi)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def make_clf(name: str, seed: int):
    if name == "rf":
        # class_weight balanced == inverse-frequency weighting, matching the
        # network's class-balanced WeightedRandomSampler at that data size.
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed,
                                      class_weight="balanced")
    if name == "mlp":
        # sklearn MLP has no class weighting, so we feed it a class-balanced
        # resample (see balanced_resample) to mirror the network's sampler.
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                             early_stopping=True, random_state=seed)
    raise ValueError(name)


def balanced_resample(Y: np.ndarray, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Indices of a class-balanced draw WITH replacement (inverse-frequency
    weights), replicating the network's WeightedRandomSampler for one epoch."""
    classes, counts = np.unique(Y, return_counts=True)
    cls_w = {c: 1.0 / n for c, n in zip(classes, counts)}
    w = np.fromiter((cls_w[y] for y in Y), dtype=np.float64, count=len(Y))
    w /= w.sum()
    return rng.choice(len(Y), size=size, replace=True, p=w)


def eval_dataset(ds, banks, clfs, ratios, data_root, seed, max_train=None, verbose=True):
    sensors = DATASET_SENSORS[ds]
    X, Y, U = load_dataset(ds, sensors, data_root)          # (N, C, T)
    w = X.transpose(0, 2, 1)                                # (N, T, C)
    t0 = time.time()
    feats = {b: extract(b, w) for b in banks}
    t_feat = time.time() - t0

    settings = ["full"] + [f"ratio{r}" for r in ratios]
    acc = {(b, c, s): [] for b in banks for c in clfs for s in settings}

    for fi, fold in enumerate(FOLDS):
        test_mask = np.isin(U, fold["test"])
        val_mask = np.isin(U, fold["val"])
        train_mask = ~(test_mask | val_mask)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        for b in banks:
            Ftr_all, Ytr_all = feats[b][train_mask], Y[train_mask]
            Fte, Yte = feats[b][test_mask], Y[test_mask]
            for s in settings:
                rng = np.random.RandomState(seed + fi)
                if s == "full":
                    idx = np.arange(len(Ytr_all))
                    # RF/MLP saturate well before 500k samples; cap the full-label
                    # train set for tractability on the huge datasets.
                    if max_train is not None and len(idx) > max_train:
                        idx = rng.choice(idx, max_train, replace=False)
                else:
                    idx = subsample(Ytr_all, float(s[5:]), rng)
                Xtr, Xte = winsor_standardize(Ftr_all[idx], Fte)
                Ytr = Ytr_all[idx]
                for c in clfs:
                    m = make_clf(c, seed)
                    if c == "mlp":
                        # Balance via resampling (MLP can't take class weights).
                        bi = balanced_resample(Ytr, len(Ytr), np.random.RandomState(seed + fi))
                        m.fit(Xtr[bi], Ytr[bi])
                    else:
                        m.fit(Xtr, Ytr)   # RF handles balance via class_weight
                    acc[(b, c, s)].append(f1_score(Yte, m.predict(Xte), average="macro"))

    out = {f"{b}|{c}|{s}": float(np.mean(v)) for (b, c, s), v in acc.items() if v}
    if verbose:
        print(f"  {ds:<14} n={len(Y):>7} C={w.shape[2]}  feat={t_feat:.1f}s  "
              f"total={time.time()-t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_SENSORS))
    ap.add_argument("--banks", nargs="+", default=list(BANK_FN))
    ap.add_argument("--classifiers", nargs="+", default=["rf", "mlp"])
    ap.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    ap.add_argument("--data-root", default="har-datasets/data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap full-label train samples (e.g. 50000) for tractability")
    ap.add_argument("--out", default="results/features_alone.json")
    args = ap.parse_args()

    print(f"Features-alone benchmark: {len(args.datasets)} datasets x "
          f"{len(args.banks)} banks x {len(args.classifiers)} clfs")
    per_ds = {}
    t0 = time.time()
    for ds in args.datasets:
        try:
            per_ds[ds] = eval_dataset(ds, args.banks, args.classifiers,
                                      args.ratios, args.data_root, args.seed,
                                      max_train=args.max_train)
        except Exception as e:
            print(f"  [warn] {ds} failed: {e}")
    print(f"done in {time.time()-t0:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(per_ds, open(args.out, "w"), indent=2)
    print(f"[json] wrote {args.out}")

    # Aggregate: mean over datasets, per (bank, clf, setting).
    settings = ["full"] + [f"ratio{r}" for r in args.ratios]
    print("\n" + "=" * 70)
    print("FEATURES-ALONE  (mean macro-F1 over datasets)")
    print("=" * 70)
    hdr = f"{'bank|clf':<22}" + "".join(s.replace('ratio', '').rjust(8) for s in settings)
    print(hdr)
    print("-" * len(hdr))
    for b in args.banks:
        for c in args.classifiers:
            row = f"{b + '|' + c:<22}"
            for s in settings:
                vals = [per_ds[d].get(f"{b}|{c}|{s}") for d in per_ds
                        if per_ds[d].get(f"{b}|{c}|{s}") is not None]
                row += (f"{np.mean(vals):.4f}".rjust(8) if vals else "   -- ".rjust(8))
            print(row)


if __name__ == "__main__":
    main()
