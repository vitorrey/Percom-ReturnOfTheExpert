#!/usr/bin/env python3
"""
make_tables.py
--------------
Print paper Tables VI, VII (ZS-Feat column) and VIII from the stored runs.

Run-file -> table-row mapping (runs/fulldata_<dataset>_<tag>_p<P>_f<F>.json):

  Table VI   From scratch               feature-regression runs, "supervised_scratch"
             PIM                        pim_*,                   "supervised_finetune"
             Feature regression (ours)  feature-regression runs, "supervised_finetune"
  Table VIII ZS-Net                     scores_*,                "zero_shot"
             ZS-Feat                    runs/zs_feat_<dataset>.json (zero_shot_from_feat.py)
             Feature regression         feature-regression runs, "supervised_finetune"
             Prototypes only            scores_*,                "supervised_finetune"
             Features then prototypes   scores_*_from_feat,      "supervised_finetune"

Feature-regression tags: mmfit (MM-Fit), wear_wear (WEAR), feat (PAMAP2).
Each cell: macro-F1 mean +- std over all 25 runs (5 pre-training x 5
fine-tuning seeds).

Usage:  python make_tables.py [--runs_dir runs]
"""
import argparse
import glob
import json
import os

import numpy as np

DATASETS = ["pamap2", "wear", "mmfit"]
NAMES = {"pamap2": "PAMAP2", "wear": "WEAR", "mmfit": "MM-Fit"}
TAGS = {  # role -> per-dataset run tag
    "feat":            {"pamap2": "feat",             "wear": "wear_wear",             "mmfit": "mmfit"},
    "pim":             {"pamap2": "pim",              "wear": "pim_wear",              "mmfit": "pim_mmfit"},
    "proto":           {"pamap2": "scores",           "wear": "scores_wear",           "mmfit": "scores_mmfit"},
    "feat_then_proto": {"pamap2": "scores_from_feat", "wear": "scores_wear_from_feat", "mmfit": "scores_mmfit_from_feat"},
}


def f1s(runs_dir, ds, role, cond):
    files = sorted(glob.glob(os.path.join(runs_dir, f"fulldata_{ds}_{TAGS[role][ds]}_p*_f*.json")))
    return [json.load(open(f))[cond]["f1"] for f in files]


def cell(v, std=True):
    if not v:
        return "—"
    return f"{np.mean(v):.3f}±{np.std(v):.3f}" if std else f"{np.mean(v):.2f}"


def table(title, rows, runs_dir):
    print(f"\n{title}")
    print(f"  {'Method':28s}" + "".join(f"{NAMES[d]:>16s}" for d in DATASETS))
    for label, fn in rows:
        print(f"  {label:28s}" + "".join(f"{fn(d):>16s}" for d in DATASETS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    r = ap.parse_args().runs_dir

    def zs_feat(ds):
        p = os.path.join(r, f"zs_feat_{ds}.json")
        return f"{json.load(open(p))['zero_shot_feat']['f1']:.2f}" if os.path.exists(p) else "—"

    table("TABLE VI  (within-dataset, full-data macro-F1, mean±std over 25 runs)", [
        ("From scratch",              lambda d: cell(f1s(r, d, "feat", "supervised_scratch"))),
        ("PIM",                       lambda d: cell(f1s(r, d, "pim", "supervised_finetune"))),
        ("Feature regression (ours)", lambda d: cell(f1s(r, d, "feat", "supervised_finetune"))),
    ], r)
    table("TABLE VIII  (macro-F1)", [
        ("ZS-Net (no labels)",        lambda d: cell(f1s(r, d, "proto", "zero_shot"), std=False)),
        ("ZS-Feat (no labels)",       zs_feat),
        ("Feature regression",        lambda d: cell(f1s(r, d, "feat", "supervised_finetune"))),
        ("Prototypes only",           lambda d: cell(f1s(r, d, "proto", "supervised_finetune"))),
        ("Features then prototypes",  lambda d: cell(f1s(r, d, "feat_then_proto", "supervised_finetune"))),
    ], r)

    p = os.path.join(r, "zs_feat_mmfit.json")
    if os.path.exists(p):
        print("\nTABLE VII  ZS-Feat per-class accuracy on MM-Fit (%)")
        for cls, v in json.load(open(p))["zero_shot_feat"]["per_class"].items():
            print(f"  {cls:28s}{100 * v['acc']:6.1f}   n={v['n']}")


if __name__ == "__main__":
    main()
