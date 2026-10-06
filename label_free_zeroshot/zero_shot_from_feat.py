"""
zero_shot_from_feat.py
----------------------
ZS-Feat (paper Tables VII and VIII): label-free recognition that scores each
window's own physical feature profile against every LLM-authored class profile
(pseudo_labels.py) and predicts the best-matching class. No network and no
labels are used; the result is deterministic.

Usage
-----
  python zero_shot_from_feat.py --data_root data --out_dir runs
  # -> runs/zs_feat_{mmfit,wear,pamap2}.json (macro-F1 + per-class accuracy)
"""
import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

# Projects imports
from dataset import (
    MMFitDataset, MMFIT_CLASSES, CLASS_TO_IDX, 
    MMFIT_LIMB_GROUPS, SENSOR_NAMES, MMFIT_SPLITS,
    WINDOW_FRAMES as MMFIT_WF, STRIDE_FRAMES as MMFIT_SF,
    load_labeled_windows
)
from dataset_wear import (
    WEARDataset, WEAR_CLASSES, WEAR_CLASS_TO_IDX, 
    WEAR_LIMB_GROUPS, WEAR_SPLITS, WEAR_LABEL_TO_PROFILE_KEY,
    WINDOW_FRAMES as WEAR_WF, STRIDE_FRAMES as WEAR_SF,
    load_wear_labeled_windows
)
from dataset_pamap2 import (
    PAMAP2Dataset, PAMAP2_CLASSES, PAMAP2_CLASS_TO_IDX,
    PAMAP2_LIMB_GROUPS, pamap2_loso_splits,
    WINDOW_FRAMES as PAMAP2_WF, STRIDE_FRAMES as PAMAP2_SF,
    load_pamap2_labeled_windows
)
from pseudo_labels import MMFIT_ACTIVITIES, WEAR_ACTIVITIES, PAMAP2_ACTIVITIES
from dataset_scores import compute_scores_batch, LimbScoreNormalizer

# -----------------------------------------------------------------------
# Mappings (copied from evaluate_fulldata.py)
# -----------------------------------------------------------------------

_MMFIT_DATASET_TO_PROFILE_KEY = {
    "bicep_curls":             "bicep_curls",
    "dumbbell_rows":           "dumbbell_rows",
    "dumbbell_shoulder_press": "shoulder_press",
    "jumping_jacks":           "jumping_jacks",
    "lateral_shoulder_raises": "lateral_raises",
    "lunges":                  "lunges",
    "pushups":                 "pushups",
    "situps":                  "situps",
    "squats":                  "squats",
    "tricep_extensions":       "tricep_extensions",
    "null":                    "null",
}

_WEAR_PROFILE_KEY_TO_CLS = {
    prof: WEAR_CLASS_TO_IDX[cls]
    for cls, prof in WEAR_LABEL_TO_PROFILE_KEY.items()
    if cls in WEAR_CLASS_TO_IDX
}
WEAR_SCORE_DIM_TO_CLASS = [
    _WEAR_PROFILE_KEY_TO_CLS.get(k, 0) for k in WEAR_ACTIVITIES.keys()
]

_MMFIT_PROFILE_KEY_TO_CLS = {
    prof: CLASS_TO_IDX[cls]
    for cls, prof in _MMFIT_DATASET_TO_PROFILE_KEY.items()
    if cls in CLASS_TO_IDX
}
MMFIT_SCORE_DIM_TO_CLASS = [
    _MMFIT_PROFILE_KEY_TO_CLS.get(k, CLASS_TO_IDX["null"])
    for k in MMFIT_ACTIVITIES.keys()
]

PAMAP2_SCORE_DIM_TO_CLASS = list(range(len(PAMAP2_ACTIVITIES)))


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_limb_arrays(profiles, n_sensors, limb_groups):
    n_feats = profiles.shape[1] // n_sensors
    arr = profiles.reshape(-1, n_sensors, n_feats)
    return {
        limb: arr[:, idxs, :].mean(axis=1)
        for limb, idxs in limb_groups.items()
    }

def scores_to_preds(scores, n_classes, dim_to_class):
    cls_scores = np.zeros((scores.shape[0], n_classes))
    for di, ci in enumerate(dim_to_class):
        cls_scores[:, ci] = np.maximum(cls_scores[:, ci], scores[:, di])
    return cls_scores.argmax(axis=1)

def print_report(name, classes, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  Accuracy         : {acc:.3f}")
    print(f"  Balanced accuracy: {bacc:.3f}")
    print(f"  Macro F1         : {f1:.3f}")
    print("  Per-class accuracy:")
    per_class = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if not mask.any():
            continue
        cacc = accuracy_score(y_true[mask], y_pred[mask])
        per_class[cls] = {"acc": float(cacc), "n": int(mask.sum())}
        print(f"    {cls:35s}: {cacc:.3f}  (n={mask.sum()})")
    return {"acc": float(acc), "bacc": float(bacc), "f1": float(f1),
            "per_class": per_class}


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def eval_dataset(name, data_root="data"):
    print(f"\nEvaluating {name.upper()}...")
    cache_dir = os.path.join(data_root, "cache")
    
    if name == "wear":
        train_ds = WEARDataset(data_root, WEAR_SPLITS["train"], WEAR_WF, WEAR_SF, cache_dir=cache_dir)
        test_ds  = WEARDataset(data_root, WEAR_SPLITS["test"],  WEAR_WF, WEAR_SF, cache_dir=cache_dir)
        _, test_y, _ = load_wear_labeled_windows(data_root, WEAR_SPLITS["test"], WEAR_WF, WEAR_SF)
        
        train_profs = train_ds.raw_profiles()
        test_profs  = test_ds.raw_profiles()
        n_sensors, limb_groups = 4, WEAR_LIMB_GROUPS
        activities, n_classes, dim_to_class = WEAR_ACTIVITIES, len(WEAR_CLASSES), WEAR_SCORE_DIM_TO_CLASS
        classes = WEAR_CLASSES

    elif name == "mmfit":
        data_dir = os.path.join(data_root, "mm-fit")
        train_ds = MMFitDataset(data_dir, MMFIT_SPLITS["train"], MMFIT_WF, MMFIT_SF, SENSOR_NAMES, cache_dir=cache_dir)
        test_ds  = MMFitDataset(data_dir, MMFIT_SPLITS["test"],  MMFIT_WF, MMFIT_SF, SENSOR_NAMES, cache_dir=cache_dir)
        _, test_y, _ = load_labeled_windows(data_dir, MMFIT_SPLITS["test"], MMFIT_WF, MMFIT_SF, SENSOR_NAMES)
        
        train_profs = train_ds.raw_profiles()
        test_profs  = test_ds.raw_profiles()
        n_sensors, limb_groups = 4, MMFIT_LIMB_GROUPS
        activities, n_classes, dim_to_class = MMFIT_ACTIVITIES, len(MMFIT_CLASSES), MMFIT_SCORE_DIM_TO_CLASS
        classes = MMFIT_CLASSES

    elif name == "pamap2":
        subjects = [f"subject10{i}" for i in range(1, 10)]
        print(f"  Loading profiles for {len(subjects)} subjects...")
        
        # Load all profiles and true labels
        ds = PAMAP2Dataset(data_root, subjects, PAMAP2_WF, PAMAP2_SF, cache_dir=cache_dir)
        _, all_y, _ = load_pamap2_labeled_windows(data_root, subjects, PAMAP2_WF, PAMAP2_SF)
        
        all_profs = ds.raw_profiles()
        
        # In zero-shot, we fit the normalizer on the same pool of data 
        # (or the entire dataset statistics)
        print("  Fitting normalizer...")
        norm = LimbScoreNormalizer().fit(get_limb_arrays(all_profs, 3, PAMAP2_LIMB_GROUPS))
        
        print("  Computing scores...")
        la = get_limb_arrays(all_profs, 3, PAMAP2_LIMB_GROUPS)
        scores = compute_scores_batch(la, PAMAP2_ACTIVITIES, norm)
        
        print("  Evaluating...")
        preds = scores_to_preds(scores, len(PAMAP2_CLASSES), PAMAP2_SCORE_DIM_TO_CLASS)
        return print_report(f"Zero-shot from Features (PAMAP2 - All Subjects)", PAMAP2_CLASSES, all_y, preds)

    # For wear/mmfit (fixed split)
    print("  Fitting normalizer...")
    norm = LimbScoreNormalizer().fit(get_limb_arrays(train_profs, n_sensors, limb_groups))
    
    print("  Computing scores...")
    la = get_limb_arrays(test_profs, n_sensors, limb_groups)
    scores = compute_scores_batch(la, activities, norm)
    
    print("  Evaluating...")
    preds = scores_to_preds(scores, n_classes, dim_to_class)
    return print_report(f"Zero-shot from Features ({name})", classes, test_y, preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--datasets", nargs="+", default=["mmfit", "wear", "pamap2"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ds in args.datasets:
        res = eval_dataset(ds, data_root=args.data_root)
        out = os.path.join(args.out_dir, f"zs_feat_{ds}.json")
        with open(out, "w") as f:
            json.dump({"dataset": ds, "zero_shot_feat": res}, f, indent=2)
        print(f"  -> {out}")
