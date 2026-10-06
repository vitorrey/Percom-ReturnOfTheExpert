#!/usr/bin/env python3
"""
make_figures.py
===============
Paper figure for the feature-regression results. Two panels:
  (a) Few-shot macro-F1 vs label ratio: the four feature-regression banks
      (featstat/featinv/featgoogle/featunion, no-rotation) vs the SSL baselines (TimeChannel, MTL),
      all within the same reproduced HARBench pipeline.
  (b) Pretrained network vs features-alone (RF/MLP on the same features,
      class-balanced): the widening few-shot gap.

Numbers come from aggregate_results (network, via run_benchmark's own
aggregators) and results/features_alone.json (features-alone).

  python make_figures.py            # -> results/figures/featreg_results.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aggregate_results as agg

RATIO_KEYS = ["1%", "2%", "5%", "10%", "20%", "50%"]
RATIO_X = [1, 2, 5, 10, 20, 50]

# Consistent styling: our banks = solid saturated; baselines = dashed grey.
# Canonical method = no-rotation banks (as reported in the paper); rotation is
# only an ablation. Baselines (timechannel/mtl) are unaffected by rotation.
STYLE = {
    "featstat_norot":   dict(color="#1b6ca8", marker="o", ls="-",  lw=2.2, label="Feat-reg: statistical"),
    "featinv_norot":    dict(color="#2a9d3f", marker="s", ls="-",  lw=2.2, label="Feat-reg: biomechanical"),
    "featgoogle_norot": dict(color="#e08214", marker="^", ls="-",  lw=2.2, label="Feat-reg: spectral-PCA"),
    "featunion_norot":  dict(color="#9b3fb0", marker="P", ls="-",  lw=2.2, label="Feat-reg: union"),
    "timechannel":      dict(color="#8a8a8a", marker="v", ls="--", lw=1.6, label="TimeChannel (SSL)"),
    "mtl":              dict(color="#4d4d4d", marker="D", ls="--", lw=1.6, label="MTL (SSL)"),
}
ORDER = ["featstat_norot", "featinv_norot", "featgoogle_norot", "featunion_norot",
         "timechannel", "mtl"]


def load_stats():
    """Means and SEMs (over the 18 datasets) precomputed by paper/gen_tables.py.

    Using the same JSON that populates the tables keeps figure and tables exactly
    consistent. Error bars are +-1 SEM.
    """
    for p in (Path("paper/fig_stats.json"), Path("fig_stats.json")):
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError("fig_stats.json not found; run `python paper/gen_tables.py` first")


def main():
    st_data = load_stats()
    fs = {m: np.array(st_data["methods"][m]["mean"]) for m in st_data["methods"]}
    fs_sem = {m: np.array(st_data["methods"][m]["sem"]) for m in st_data["methods"]}
    feat_alone = np.array(st_data["features_alone"]["mean"])
    feat_alone_sem = np.array(st_data["features_alone"]["sem"])
    # network representative for panel (b): no-rotation statistical bank
    net_rep = fs["featstat_norot"]
    net_rep_sem = fs_sem["featstat_norot"]

    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) four banks vs SSL baselines, with +-1 SEM error bars
    for m in ORDER:
        y = fs[m]
        st = dict(STYLE[m])
        lbl = st.pop("label")
        ax1.errorbar(RATIO_X, y, yerr=fs_sem[m], markersize=5, label=lbl,
                     capsize=2, elinewidth=1.0, **st)
    ax1.set_xscale("log")
    ax1.set_xticks(RATIO_X)
    ax1.set_xticklabels([f"{r}%" for r in RATIO_X])
    ax1.set_xlabel("Labelled data per dataset")
    ax1.set_ylabel("Macro-F1 (mean over 18 datasets)")
    ax1.set_title("(a) Few-shot: feature-regression vs SSL baselines", fontsize=11)
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

    # (b) network vs features-alone, gap shaded, +-1 SEM error bars
    ax2.errorbar(RATIO_X, net_rep, yerr=net_rep_sem, color="#1b6ca8", marker="o",
                 lw=2.2, markersize=5, capsize=2, elinewidth=1.0,
                 label="Pretrained network (feat-reg)")
    ax2.errorbar(RATIO_X, feat_alone, yerr=feat_alone_sem, color="#c0392b", marker="x",
                 ls="--", lw=1.8, markersize=6, capsize=2, elinewidth=1.0,
                 label="Features alone (RF/MLP, balanced)")
    ax2.fill_between(RATIO_X, feat_alone, net_rep, color="#1b6ca8", alpha=0.12)
    for x, lo, hi in zip(RATIO_X, feat_alone, net_rep):
        ax2.annotate(f"+{hi-lo:.02f}", (x, (lo + hi) / 2), fontsize=7.5,
                     color="#1b6ca8", ha="center", va="center")
    ax2.set_xscale("log")
    ax2.set_xticks(RATIO_X)
    ax2.set_xticklabels([f"{r}%" for r in RATIO_X])
    ax2.set_xlabel("Labelled data per dataset")
    ax2.set_ylabel("Macro-F1 (mean over 18 datasets)")
    ax2.set_title("(b) Pretrained network vs. features alone", fontsize=11)
    ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    outdir = Path("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"featreg_results.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {outdir}/featreg_results.pdf and .png")


if __name__ == "__main__":
    main()
