#!/usr/bin/env python3
"""
gen_tables.py — emit the LaTeX table bodies of paper Tables I-V (mean+-uncertainty)
and a JSON of per-ratio mean/SEM for the Fig. 3 error bars. All numbers come from
the stored 4-fold CV / 4-seed results (reusing aggregate_results' collectors).
Run from the repository root:  python paper/gen_tables.py
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import aggregate_results as agg

M = ["featstat_norot","featinv_norot","featgoogle_norot","featunion_norot","timechannel","mtl"]
BANKS = [("statistical","featstat"),("biomechanical","featinv"),("spectral-PCA","featgoogle"),("union","featunion")]
ALL = M + [b for _, b in BANKS]   # + rotation variants for Table V
DOMAIN = agg.DOMAIN
DOMNAME = {"daily":"Daily living","exercise":"Exercise","industry":"Industrial"}
DISP = {"vtt_coniot":"vtt\\_coniot","ucaehar":"ucaEHAR"}
RATIOS = ["0.01","0.02","0.05","0.1","0.2","0.5"]
RLAB = {"0.01":"1\\%","0.02":"2\\%","0.05":"5\\%","0.1":"10\\%","0.2":"20\\%","0.5":"50\\%"}

def sem(a):
    a = np.asarray(a, float)
    return a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else 0.0

def cell(mean, unc, best=False):
    v = f"\\mathbf{{{mean:.3f}}}" if best else f"{mean:.3f}"
    return f"${v}{{\\scriptstyle\\pm{unc:.3f}}}$"

def row(label, means, uncs, bold=True):
    b = int(np.argmax(means)) if bold else -1
    return f"{label} & " + " & ".join(cell(means[i], uncs[i], i == b) for i in range(len(M))) + r" \\"

jobs, res, zs, _ = agg.collect(Path("results/benchmark"))
out, _ = agg.official(jobs, res)

# per-dataset fold mean/std
pm = {m: {} for m in ALL}; pstd = {m: {} for m in ALL}
for m in ALL:
    for _, d in res["average"][m].items():
        pm[m][d["dataset"]] = d["summary"]["mean_f1"]
        pstd[m][d["dataset"]] = d["summary"].get("std_f1", 0.0)
all_ds = sorted(pm[M[0]], key=lambda x: (["daily","exercise","industry"].index(DOMAIN.get(x,"industry")), x))

# few-shot per (ratio, dataset)
fsd = {m: {} for m in ALL}
for m in ALL:
    for _, d in res["fewshot"][m].items():
        r = str(d["hyperparameters"]["data_ratio"])
        fsd[m].setdefault(r, {})[d["dataset"]] = d["summary"]["mean_f1"]

# ---------- Table I : overall (mean +- SEM over the axis unit) ----------
avg_m  = [np.mean([pm[m][d] for d in all_ds]) for m in M]
avg_s  = [sem([pm[m][d] for d in all_ds]) for m in M]
dom_m  = [out[m]["domain"]["domain_robustness"] for m in M]
dom_s  = [sem([out[m]["domain"]["domains"][k]["mean_f1"] for k in out[m]["domain"]["domains"]]) for m in M]
pos_m  = [out[m]["position"]["position_robustness"] for m in M]
def pos_unit(m):
    cats = out[m]["position"].get("categories")
    if cats:
        return [ (v["mean_f1"] if isinstance(v, dict) else v) for v in cats.values() ]
    return list(out[m]["position"]["configs"].values())
pos_s  = [sem(pos_unit(m)) for m in M]
fsavg_m, fsavg_s = [], []
for m in M:
    dss = set().union(*[set(fsd[m][r]) for r in RATIOS])
    per = [np.mean([fsd[m][r][d] for r in RATIOS if d in fsd[m][r]]) for d in dss]
    fsavg_m.append(np.mean(per)); fsavg_s.append(sem(per))
zs_m = [np.mean(list(zs[m].values())) for m in M]
zs_s = [sem(list(zs[m].values())) for m in M]

tab1 = "\n".join([
    row("Average", avg_m, avg_s),
    row("Domain rob.", dom_m, dom_s),
    row("Position rob.", pos_m, pos_s),
    row("Few-shot avg.", fsavg_m, fsavg_s),
    row("Cross-dataset", zs_m, zs_s),
])

# ---------- Table III : few-shot by ratio (mean +- SEM over datasets) ----------
lines = []
for r in RATIOS:
    means = [np.mean(list(fsd[m][r].values())) for m in M]
    uncs  = [sem(list(fsd[m][r].values())) for m in M]
    lines.append(row(RLAB[r], means, uncs))
tab3 = "\n".join(lines)

# ---------- Table II : per-dataset (mean +- fold std) ----------
def cell2(mean, unc, best):
    v = f"\\mathbf{{{mean:.3f}}}" if best else f"{mean:.3f}"
    return f"${v}{{\\scriptstyle\\pm{unc:.3f}}}$"
plines = []; last = None
for ds in all_ds:
    dom = DOMAIN.get(ds, "industry")
    if dom != last:
        plines.append(r"\midrule")
        plines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{DOMNAME[dom]}}}}} \\\\")
        last = dom
    means = [pm[m][ds] for m in M]; uncs = [pstd[m][ds] for m in M]
    b = int(np.argmax(means))
    plines.append(f"{DISP.get(ds,ds)} & " + " & ".join(cell2(means[i],uncs[i],i==b) for i in range(len(M))) + r" \\")
plines.append(r"\midrule")
mrow_m = [np.mean([pm[m][d] for d in all_ds]) for m in M]
mrow_s = [sem([pm[m][d] for d in all_ds]) for m in M]
b = int(np.argmax(mrow_m))
plines.append(r"\emph{Mean} & " + " & ".join(cell2(mrow_m[i],mrow_s[i],i==b) for i in range(len(M))) + r" \\")
tab2 = "\n".join(plines)

# ---------- Table IV : features-alone vs network (mean +- SEM over datasets) ----------
fa = json.load(open("results/features_alone.json"))
fa_ds = list(fa)
banks = ["featstat","featinv","featgoogle"]; clfs = ["rf","mlp"]
setmap = [("full","full"),("50\\%","ratio0.5"),("20\\%","ratio0.2"),("10\\%","ratio0.1"),
          ("5\\%","ratio0.05"),("2\\%","ratio0.02"),("1\\%","ratio0.01")]
# network per setting (statistical bank): full=average, ratios=few-shot
net_full = ([pm["featstat_norot"][d] for d in all_ds])
def net_ratio(r): return list(fsd["featstat_norot"][r].values())
t4 = []
for lbl, setting in setmap:
    # features-alone best (bank,clf) by mean over datasets
    best_mean, best_vals = -1, None
    for b in banks:
        for c in clfs:
            vals = [fa[d][f"{b}|{c}|{setting}"] for d in fa_ds if f"{b}|{c}|{setting}" in fa[d]]
            if vals and np.mean(vals) > best_mean:
                best_mean, best_vals = np.mean(vals), vals
    fa_m, fa_s = np.mean(best_vals), sem(best_vals)
    if setting == "full":
        nm, nvals = np.mean(net_full), net_full
    else:
        r = setting.replace("ratio",""); nvals = net_ratio(r); nm = np.mean(nvals)
    ns = sem(nvals)
    adv = nm - fa_m
    t4.append(f"{lbl} & ${fa_m:.3f}{{\\scriptstyle\\pm{fa_s:.3f}}}$ & "
              f"$\\mathbf{{{nm:.3f}}}{{\\scriptstyle\\pm{ns:.3f}}}$ & $+{adv:.3f}$ \\\\")
tab4 = "\n".join(t4)

# ---------- figure JSON: per-method per-ratio mean/SEM + features-alone-best ----------
figj = {"ratios_pct": [1,2,5,10,20,50], "methods": {}}
for m in M:
    figj["methods"][m] = {
        "mean": [float(np.mean(list(fsd[m][r].values()))) for r in RATIOS],
        "sem":  [float(sem(list(fsd[m][r].values()))) for r in RATIOS],
    }
# features-alone best per ratio (matching panel b)
fb_m, fb_s = [], []
for _, setting in [("1%","ratio0.01"),("2%","ratio0.02"),("5%","ratio0.05"),
                   ("10%","ratio0.1"),("20%","ratio0.2"),("50%","ratio0.5")]:
    best_mean, best_vals = -1, None
    for b in banks:
        for c in clfs:
            vals = [fa[d][f"{b}|{c}|{setting}"] for d in fa_ds if f"{b}|{c}|{setting}" in fa[d]]
            if vals and np.mean(vals) > best_mean:
                best_mean, best_vals = np.mean(vals), vals
    fb_m.append(float(np.mean(best_vals))); fb_s.append(float(sem(best_vals)))
figj["features_alone"] = {"mean": fb_m, "sem": fb_s}
json.dump(figj, open("paper/fig_stats.json","w"), indent=1)

# ---------- Table V : rotation ablation, Delta = no-rotation - rotation ----------
def axes(m):
    avg = np.mean([pm[m][d] for d in all_ds])
    fs = np.mean([np.mean([fsd[m][r][d] for r in RATIOS if d in fsd[m][r]])
                  for d in set().union(*[set(fsd[m][r]) for r in RATIOS])])
    return [avg, out[m]["domain"]["domain_robustness"], out[m]["position"]["position_robustness"],
            fs, np.mean(list(zs[m].values()))]
t5 = []
for name, b in BANKS:
    delta = np.array(axes(b + "_norot")) - np.array(axes(b))
    t5.append(f"{name:13s} & " + " & ".join(f"${v:+.3f}$" for v in delta) + r" \\")
tab5 = "\n".join(t5)

for name, body in [("TABLE I (overall)",tab1),("TABLE II (per-dataset)",tab2),
                   ("TABLE III (few-shot)",tab3),("TABLE IV (features-alone)",tab4),
                   ("TABLE V (rotation ablation: no-rotation - rotation)",tab5)]:
    print(f"\n%%%% {name}\n{body}")
print("\n[wrote paper/fig_stats.json]")
