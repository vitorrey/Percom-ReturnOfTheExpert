#!/usr/bin/env python3
"""
aggregate_results.py
====================
Roll the per-(model x dataset) benchmark shards under results/benchmark/ into the
featinv-vs-featstat comparison tables.

To keep the numbers identical to how HARBench reports them, this does NOT
re-implement the roll-ups: it reconstructs the (jobs, results) structures the
benchmark builds in-memory and calls run_benchmark.py's own aggregators
(aggregate_average_results / aggregate_domain_results / aggregate_position_results
/ aggregate_fewshot_results). Those read result["summary"]["mean_f1"] — the same
4-fold CV macro-F1 stored in each shard's results.json. On top of their per-model
output we print the head-to-head comparison.

Zero-shot has no standalone aggregator in run_benchmark (eval_zeroshot_performance
computes inline), so we read dataset_results[target].f1 directly — the value the
benchmark itself produced.

Usage:
  python aggregate_results.py [--root results/benchmark]
"""
import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import run_benchmark as rb

MODELS = ["featinv", "featstat", "featgoogle", "featunion",
          "featinv_norot", "featstat_norot", "featgoogle_norot", "featunion_norot",
          "timechannel", "mtl"]
STAGES = ("average", "position", "fewshot", "zeroshot")

# Published HARBench numbers (ResNet18) for the two baselines, for a reproduction
# check — our within-pipeline runs should land close to these.
PUBLISHED = {
    "timechannel": {"average": 0.791, "domain": 0.813, "position": 0.697,
                    "fewshot": 0.725, "zeroshot": 0.636},
    "mtl": {"average": 0.826, "domain": 0.839, "position": 0.743,
            "fewshot": 0.714, "zeroshot": 0.807},
}

# Lookups straight from the benchmark's own config (authoritative).
DOMAIN = {ds: dom for dom, dsets in rb.DATASETS.items() for ds in dsets}
POS_CAT = {(ds, se): cat for cat, cfgs in rb.POSITION_CONFIGS.items() for ds, se in cfgs}


def _stage(parts):
    for s in STAGES:
        if s in parts:
            return s
    return None


def collect(root: Path):
    """Reconstruct per-model {jobs, results} for each stage from the shards.

    Re-run shards leave duplicate results.json for the same logical run (e.g. the
    ward/openpack few-shot re-runs), so we first keep only the LATEST per logical
    key (resolved by the JSON 'timestamp'), then build the jobs/results the
    official aggregators expect. Without this, aggregate_fewshot_results would
    double-count the re-run ratios.
    """
    # best[key] = (timestamp, payload); key identifies one logical run.
    best = {}
    n_files = 0

    for jp in root.rglob("results.json"):
        parts = jp.parts
        stage = _stage(parts)
        if stage is None:
            continue
        try:
            d = json.load(open(jp))
        except Exception as e:
            print(f"[warn] unreadable {jp}: {e}")
            continue
        m = d.get("model")
        if m not in MODELS:
            continue
        n_files += 1
        ts = d.get("timestamp", "")

        if stage == "zeroshot":
            for tgt, r in d.get("dataset_results", {}).items():
                key = ("zeroshot", m, tgt)
                if key not in best or ts > best[key][0]:
                    best[key] = (ts, ("zeroshot", m, tgt, r["f1"]))
            continue
        if "summary" not in d:
            continue

        if stage == "average":
            key = ("average", m, d["dataset"])
        elif stage == "fewshot":
            ratio = d.get("hyperparameters", {}).get("data_ratio")
            key = ("fewshot", m, d["dataset"], ratio)
        elif stage == "position":
            sensor = d["sensors"][0] if d.get("sensors") else "?"
            key = ("position", m, d["dataset"], sensor)
        else:
            continue
        if key not in best or ts > best[key][0]:
            best[key] = (ts, (stage, m, d))

    # Build the structures the official aggregators consume, from deduped runs.
    jobs = {s: {m: [] for m in MODELS} for s in ("average", "position", "fewshot")}
    res = {s: {m: {} for m in MODELS} for s in ("average", "position", "fewshot")}
    zs = {m: {} for m in MODELS}

    for i, (_, payload) in enumerate(best.values()):
        if payload[0] == "zeroshot":
            _, m, tgt, f1 = payload
            zs[m][tgt] = f1
            continue
        stage, m, d = payload
        jid = f"job{i}"
        if stage == "average":
            ds = d["dataset"]
            jobs["average"][m].append({"id": jid, "dataset": ds, "domain": DOMAIN.get(ds)})
            res["average"][m][jid] = d
        elif stage == "fewshot":
            ratio = d.get("hyperparameters", {}).get("data_ratio")
            jobs["fewshot"][m].append({"id": jid, "ratio": ratio})
            res["fewshot"][m][jid] = d
        elif stage == "position":
            ds = d["dataset"]
            sensor = d["sensors"][0] if d.get("sensors") else "?"
            cat = POS_CAT.get((ds, sensor), "?")
            jobs["position"][m].append(
                {"id": jid, "dataset": ds, "sensors": [sensor], "category": cat})
            res["position"][m][jid] = d

    return jobs, res, zs, n_files


def official(jobs, res):
    """Call run_benchmark's own aggregators per model; return their result dicts.
    Their stdout (the benchmark's native breakdown) is captured and returned too."""
    out = {m: {} for m in MODELS}
    logs = {m: "" for m in MODELS}
    for m in MODELS:
        buf = io.StringIO()
        with redirect_stdout(buf):
            avg = rb.aggregate_average_results(jobs["average"][m], res["average"][m])
            dom = rb.aggregate_domain_results(avg)
            pos = rb.aggregate_position_results(jobs["position"][m], res["position"][m])
            few = rb.aggregate_fewshot_results(jobs["fewshot"][m], res["fewshot"][m])
        out[m] = {"average": avg, "domain": dom, "position": pos, "fewshot": few}
        logs[m] = buf.getvalue()
    return out, logs


def _f(x, w=8):
    return f"{x:.4f}".rjust(w) if isinstance(x, (int, float)) else "  --  ".rjust(w)


def _cell(x):
    return f"{x:.4f}".rjust(11) if isinstance(x, (int, float)) else "  --  ".rjust(11)


def _hdr(label_w=22):
    return f"{'':<{label_w}}" + "".join(m.rjust(11) for m in MODELS)


def compare(out, zs):
    zmean = {m: (np.mean(list(zs[m].values())) if zs[m] else None) for m in MODELS}

    # Headline: all axes x all models.
    print("\n" + "#" * (22 + 11 * len(MODELS)))
    print("#  FIVE-AXIS COMPARISON  (macro-F1, all within the same pipeline)")
    print("#" * (22 + 11 * len(MODELS)))
    print(_hdr())
    print("-" * (22 + 11 * len(MODELS)))
    axis_vals = {
        "Average (18 ds)": {m: out[m]["average"].get("average_f1") for m in MODELS},
        "Domain robustness": {m: out[m]["domain"].get("domain_robustness") for m in MODELS},
        "Position robustness": {m: out[m]["position"].get("position_robustness") for m in MODELS},
        "Few-shot average": {m: out[m]["fewshot"].get("fewshot_average") for m in MODELS},
        "Zero-shot mean": zmean,
    }
    for label, vals in axis_vals.items():
        print(f"{label:<22}" + "".join(_cell(vals[m]) for m in MODELS))

    # Reproduction check for the baselines vs their published numbers.
    print("\n  reproduction check (ours - published):")
    axis_key = {"Average (18 ds)": "average", "Domain robustness": "domain",
                "Position robustness": "position", "Few-shot average": "fewshot",
                "Zero-shot mean": "zeroshot"}
    for m in ("timechannel", "mtl"):
        if m not in MODELS:
            continue
        deltas = []
        for label, k in axis_key.items():
            ours = axis_vals[label][m]
            pub = PUBLISHED[m][k]
            if isinstance(ours, (int, float)):
                deltas.append(f"{k}={ours-pub:+.3f}")
        print(f"    {m:<12} " + "  ".join(deltas))

    # Few-shot per ratio — the headline table for the paper.
    print("\n" + "=" * (8 + 11 * len(MODELS)))
    print("FEW-SHOT by label ratio  (mean macro-F1 over datasets)")
    print("=" * (8 + 11 * len(MODELS)))
    print(f"{'ratio':>8}" + "".join(m.rjust(11) for m in MODELS))
    print("-" * (8 + 11 * len(MODELS)))
    ratio_keys = set()
    for m in MODELS:
        ratio_keys |= set(out[m]["fewshot"].get("ratios", {}))
    for k in sorted(ratio_keys, key=lambda s: float(s.rstrip("%"))):
        print(f"{k:>8}" + "".join(_cell(out[m]["fewshot"].get("ratios", {}).get(k)) for m in MODELS))

    # Zero-shot per target.
    print("\n" + "=" * (12 + 11 * len(MODELS)))
    print("ZERO-SHOT by target (LODO, macro-F1 over seeds)")
    print("=" * (12 + 11 * len(MODELS)))
    print(f"{'target':<12}" + "".join(m.rjust(11) for m in MODELS))
    print("-" * (12 + 11 * len(MODELS)))
    tgts = set()
    for m in MODELS:
        tgts |= set(zs[m])
    for t in sorted(tgts):
        print(f"{t:<12}" + "".join(_cell(zs[m].get(t)) for m in MODELS))

    # Per-dataset average F1.
    print("\n" + "=" * (26 + 11 * len(MODELS)))
    print("PER-DATASET average macro-F1")
    print("=" * (26 + 11 * len(MODELS)))
    print(f"{'dataset':<16}{'domain':<10}" + "".join(m.rjust(11) for m in MODELS))
    print("-" * (26 + 11 * len(MODELS)))
    all_ds = set()
    for m in MODELS:
        all_ds |= set(out[m]["average"].get("datasets", {}))
    for ds in sorted(all_ds, key=lambda x: (DOMAIN.get(x, "z"), x)):
        row = "".join(_cell(out[m]["average"].get("datasets", {}).get(ds, {}).get("f1")) for m in MODELS)
        print(f"{ds:<16}{DOMAIN.get(ds,'?'):<10}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/benchmark")
    ap.add_argument("--show-native", action="store_true",
                    help="also print run_benchmark's native per-model breakdown")
    args = ap.parse_args()
    root = Path(args.root)

    jobs, res, zs, n = collect(root)
    print(f"parsed {n} results.json under {root}")
    for s in ("average", "position", "fewshot"):
        for m in MODELS:
            print(f"  {s:<9} {m:<9}: {len(jobs[s][m])} runs")
    for m in MODELS:
        print(f"  zeroshot  {m:<9}: {len(zs[m])} targets")

    out, logs = official(jobs, res)
    if args.show_native:
        for m in MODELS:
            print("\n" + "#" * 60 + f"\n# run_benchmark native breakdown: {m}\n" + "#" * 60)
            print(logs[m])

    compare(out, zs)


if __name__ == "__main__":
    main()
