#!/usr/bin/env python3
"""
verify_datasets.py
==================
Completeness check for all NON-NHANES HARBench datasets (NHANES has its own
verify_nhanes.py because it's a sparse remote server, not a fixed archive).

For each expected dataset it checks the processed/<dataset>/ folder:
  - exists and is non-empty
  - has a metadata.json
  - has at least one loadable X.npy (and reports total window count + a shape)

Prints a per-dataset table and a final PASS/FAIL. Any MISSING or EMPTY dataset
is a real gap to re-download; a loadable one with sane shapes is good.

Usage (from harbench root):
  python verify_datasets.py
  python verify_datasets.py --processed har-datasets/data/processed
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Mirror slurm/worker_download.sh DATASETS (task-id order).
PRETRAIN = [
    "adlrd", "chad", "capture24", "dog", "har70plus", "hhar", "imsb",
    "kddi_kitchen_left", "kddi_kitchen_right", "motionsense", "opportunity",
    "sbrhapt", "tmd", "wisdm",
]
DOWNSTREAM = [
    "dsads", "pamap2", "mhealth", "realdisp", "mex", "forthtrace", "harth",
    "imwsha", "paal", "realworld", "selfback", "ucaehar", "uschad", "ward",
    "lara", "openpack", "exoskeletons", "vtt_coniot",
]


def check_one(root: Path):
    """Return dict with status/n_users/n_windows/shape/has_meta for a dataset."""
    if not root.exists():
        return {"status": "MISSING", "n_users": 0, "n_windows": 0,
                "shape": None, "has_meta": False}

    x_files = list(root.glob("USER*/**/X.npy"))
    # Some datasets nest users differently; fall back to any X.npy.
    if not x_files:
        x_files = list(root.glob("**/X.npy"))

    if not x_files:
        return {"status": "EMPTY", "n_users": 0, "n_windows": 0,
                "shape": None, "has_meta": (root / "metadata.json").exists()}

    # Count windows via the header only (mmap) so this stays fast, and grab one
    # concrete shape to eyeball the (N, C, T) layout.
    n_windows, shape, bad = 0, None, 0
    user_dirs = {p.parts[len(root.parts)] for p in x_files
                 if len(p.parts) > len(root.parts)}
    for xf in x_files:
        try:
            arr = np.load(xf, mmap_mode="r")
            n_windows += arr.shape[0]
            if shape is None:
                shape = tuple(arr.shape)
        except Exception:
            bad += 1
    status = "OK" if bad == 0 else f"OK({bad} unreadable)"
    return {"status": status, "n_users": len(user_dirs), "n_windows": n_windows,
            "shape": shape, "has_meta": (root / "metadata.json").exists()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="har-datasets/data/processed")
    args = ap.parse_args()
    base = Path(args.processed)
    if not base.exists():
        print(f"[error] processed dir not found: {base}", file=sys.stderr)
        return 2

    hdr = f"{'dataset':<20} {'status':<16} {'users':>6} {'windows':>10} {'meta':>5}  shape"
    print(hdr)
    print("-" * len(hdr))

    failures = []
    for group, names in (("PRETRAIN", PRETRAIN), ("DOWNSTREAM", DOWNSTREAM)):
        print(f"# {group}")
        for ds in names:
            r = check_one(base / ds)
            meta = "yes" if r["has_meta"] else "NO"
            shape = "" if r["shape"] is None else str(r["shape"])
            print(f"{ds:<20} {r['status']:<16} {r['n_users']:>6} "
                  f"{r['n_windows']:>10} {meta:>5}  {shape}")
            if r["status"] in ("MISSING", "EMPTY") or not r["has_meta"]:
                failures.append((ds, r["status"], meta))

    print("=" * len(hdr))
    if failures:
        print(f"PROBLEMS ({len(failures)}): re-download these:")
        for ds, st, meta in failures:
            why = st if st in ("MISSING", "EMPTY") else "no metadata.json"
            print(f"  {ds:<20} {why}")
        # Emit the array indices for slurm/submit_download.sh convenience.
        order = PRETRAIN + DOWNSTREAM
        idxs = sorted(order.index(ds) for ds, _, _ in failures)
        print("re-run those array indices, e.g.:")
        print(f"  sbatch --array={','.join(map(str, idxs))} ... slurm/worker_download.sh")
        return 1
    print(f"ALL {len(PRETRAIN) + len(DOWNSTREAM)} datasets OK "
          f"(loadable X.npy + metadata.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
