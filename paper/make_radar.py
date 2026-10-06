#!/usr/bin/env python3
"""
make_radar.py
=============
Single-column radar chart summarizing the five HARBench generalization axes for
the feature-regression union bank (our best all-round variant) against the two
SSL baselines re-run in the same pipeline (MTL, TimeChannel). Numbers are the
FINAL no-rotation values reported in the paper (overall table + cross-dataset
mean), hardcoded here so the figure is reproducible without the results tree.

  python make_radar.py   # -> figs/radar_axes.pdf / .png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["Average", "Domain\nrobustness", "Position\nrobustness",
        "Few-shot\naverage", "Cross-dataset\ntransfer"]

# FINAL no-rotation macro-F1 (overall table; cross-dataset = LODO mean).
SERIES = {
    "Feat-reg (union)": dict(vals=[0.827, 0.845, 0.730, 0.763, 0.666],
                             color="#7d3c98", lw=2.4, ls="-", marker="o"),
    "MTL (SSL)":        dict(vals=[0.828, 0.842, 0.742, 0.715, 0.815],
                             color="#c0392b", lw=1.9, ls="--", marker="s"),
    "TimeChannel (SSL)":dict(vals=[0.802, 0.824, 0.688, 0.732, 0.556],
                             color="#7f8c8d", lw=1.9, ls=":", marker="^"),
}

RLIM = (0.50, 0.90)


def main():
    N = len(AXES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(4.6, 4.4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=9)
    ax.set_ylim(*RLIM)
    ax.set_yticks([0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.6", "0.7", "0.8"], fontsize=7.5, color="#555")
    ax.set_rlabel_position(180 / N)
    ax.grid(alpha=0.35, linewidth=0.7)
    ax.spines["polar"].set_alpha(0.3)

    for name, s in SERIES.items():
        v = s["vals"] + s["vals"][:1]
        ax.plot(angles, v, color=s["color"], lw=s["lw"], ls=s["ls"],
                marker=s["marker"], markersize=4, label=name)
        if s["ls"] == "-":
            ax.fill(angles, v, color=s["color"], alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.14),
              fontsize=8.5, frameon=False)
    fig.tight_layout()

    outdir = Path(__file__).parent / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"radar_axes.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {outdir}/radar_axes.pdf and .png")


if __name__ == "__main__":
    main()
