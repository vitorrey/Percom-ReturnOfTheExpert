#!/usr/bin/env python3
"""
reproduce_profiles.py
---------------------
Quantify how well a regenerated set of LLM-authored profiles matches the
committed ones in pseudo_labels.py. See PROFILE_GENERATION.md for the prompt.

A regenerated set is a JSON mapping:
    { "<dataset>/<label>": { "<limb>": { "<feature>": "low|medium|high" } } }

Usage:
    python reproduce_profiles.py regenerated_profiles.json
    python reproduce_profiles.py regenerated_profiles.json --dataset mmfit
"""
import argparse
import json
import sys

from pseudo_labels import ALL_ACTIVITIES, FEATURE_KEYS

LEVELS = {"low": 0, "medium": 1, "high": 2}


def cells(profile_limbs):
    """Yield ((limb, feature), level) for every assigned cell."""
    for limb, feats in profile_limbs.items():
        for f in FEATURE_KEYS:
            if f in feats:
                yield (limb, f), feats[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regenerated", help="JSON of regenerated profiles")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset (mmfit/wear/pamap2)")
    args = ap.parse_args()

    regen = json.load(open(args.regenerated))

    by_ds = {}
    for key, ref in ALL_ACTIVITIES.items():
        ds = ref.dataset
        if args.dataset and ds != args.dataset:
            continue
        if key not in regen:
            continue
        ref_cells = dict(cells(ref.limbs))
        gen_cells = dict(cells(regen[key]))
        common = set(ref_cells) & set(gen_cells)
        for c in common:
            rv, gv = ref_cells[c], gen_cells[c]
            d = abs(LEVELS[rv] - LEVELS[gv])
            st = by_ds.setdefault(ds, {"n": 0, "exact": 0, "ord_dist": 0, "classes": set()})
            st["n"] += 1
            st["exact"] += int(d == 0)
            st["ord_dist"] += d
            st["classes"].add(key)

    if not by_ds:
        print("No overlapping classes between the regenerated file and pseudo_labels.py.")
        sys.exit(1)

    print(f"\n  {'dataset':<10} {'classes':>7} {'cells':>6} {'exact-match':>12} {'mean ord.dist':>14}")
    print("  " + "-" * 52)
    tot_n = tot_e = tot_d = 0
    for ds, st in sorted(by_ds.items()):
        n, e, d = st["n"], st["exact"], st["ord_dist"]
        tot_n += n; tot_e += e; tot_d += d
        print(f"  {ds:<10} {len(st['classes']):>7} {n:>6} {e/n:>11.1%} {d/n:>14.3f}")
    if len(by_ds) > 1:
        print("  " + "-" * 52)
        print(f"  {'ALL':<10} {'':>7} {tot_n:>6} {tot_e/tot_n:>11.1%} {tot_d/tot_n:>14.3f}")
    print("\n  (ordinal distance: low<medium<high; 0 = identical, 1 = one band off, 2 = opposite)\n")


if __name__ == "__main__":
    main()
