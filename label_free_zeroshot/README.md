# Within-dataset comparison and label-free recognition (Tables VI–VIII)

This directory contains the code and stored results for:

- **Table VI** (Section V-F): within-dataset comparison with PIM,
- **Table VII** (Section VI): LLM-authored motion profiles and per-class zero-shot
  accuracy on MM-Fit,
- **Table VIII** (Section VI): label-free recognition (ZS-Net, ZS-Feat) and
  prototype pre-training.

> **Setup differs from the HARBench experiments.** As stated in the paper, these
> experiments use a **1D convolutional encoder** (`model.py::Encoder1D`) on
> **three datasets** (PAMAP2, WEAR, MM-Fit), pre-trained and fine-tuned within each
> dataset, not the ResNet18 / HARBench pipeline of the parent directory. The
> regression target is the per-sensor set of **eight physical features** in
> `features.py` (RMS, speed, distance, impact rate, tilt mean/variability, angular
> range/rate). These are the same kinds of quantity the biomechanical bank uses,
> but in a compact, named form that the LLM-authored profiles refer to. All methods
> in Tables VI and VIII share the encoder and the fine-tuning protocol.

Datasets are **not** shipped. Stored results are in `runs/`, so every table
regenerates without re-running anything.

## Contents

```
features.py                 the eight physical features (feature-regression target)
pseudo_labels.py            LLM-authored per-limb class profiles (low / medium / high)
PROFILE_GENERATION.md       how the profiles were generated: model, settings, exact prompt
reproduce_profiles.py       score a regenerated profile set against pseudo_labels.py
regenerated_profiles.json   sample MM-Fit regeneration (97.2% exact match)

dataset.py, dataset_wear.py, dataset_pamap2.py   MM-Fit / WEAR / PAMAP2 loaders
dataset_scores.py           profile-match scores used by prototype pre-training
model.py                    1D-CNN encoder + regression head
model_scores.py             encoder + class-profile score head (prototype pre-training)

train.py                    feature regression, MM-Fit
train_wear.py               feature regression, WEAR
train_pamap2.py             feature regression, PAMAP2 (one checkpoint per LOSO fold)
train_scores.py             prototype pre-training ("prototypes only"; with --init_from:
                            "features then prototypes")
train_pim.py, model_pim.py, dataset_pim.py, features_pim.py
                            PIM baseline, implemented from the formulation in the PIM paper

evaluate_fulldata.py        from-scratch / fine-tuned / zero-shot (ZS-Net) evaluation -> runs/fulldata_*.json
zero_shot_from_feat.py      ZS-Feat (no network) -> runs/zs_feat_*.json
make_tables.py              prints Tables VI, VII (ZS-Feat column) and VIII from runs/
run_all.sh                  regenerates every file in runs/ from scratch
runs/                       stored results
```

## Regenerate the tables from the stored results

```bash
python make_tables.py
```

How `runs/` maps to table rows (`runs/fulldata_<dataset>_<tag>_p<P>_f<F>.json`,
5 pre-training seeds × 5 fine-tuning seeds per cell):

| Table row | Run tag (PAMAP2 / WEAR / MM-Fit) | Field |
|---|---|---|
| VI: From scratch | `feat` / `wear_wear` / `mmfit` | `supervised_scratch` |
| VI: PIM | `pim` / `pim_wear` / `pim_mmfit` | `supervised_finetune` |
| VI, VIII: Feature regression (ours) | `feat` / `wear_wear` / `mmfit` | `supervised_finetune` |
| VIII: Prototypes only | `scores` / `scores_wear` / `scores_mmfit` | `supervised_finetune` |
| VIII: Features then prototypes | `scores_from_feat` / `scores_wear_from_feat` / `scores_mmfit_from_feat` | `supervised_finetune` |
| VIII: ZS-Net (no labels) | `scores` / `scores_wear` / `scores_mmfit` | `zero_shot` |
| VII, VIII: ZS-Feat (no labels) | `runs/zs_feat_<dataset>.json` | `zero_shot_feat` (incl. per-class accuracy) |

**Splits.** WEAR and MM-Fit use fixed train/validation/test subject splits
(`WEAR_SPLITS`, `MMFIT_SPLITS`). PAMAP2 uses 9-fold leave-one-subject-out
(`dataset_pamap2.pamap2_loso_splits`), and results are averaged over folds.

## Reproduce from scratch

Expected data layout under `data/`:

```
data/mm-fit/w00 ... w20/                                  MM-Fit
data/pamap2/Protocol/subject101.dat ... subject109.dat    PAMAP2
data/ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_*.csv   WEAR
```

```bash
pip install -r requirements.txt
DATA=data ./run_all.sh
python make_tables.py
```

`run_all.sh` lists every pre-training and evaluation command with the seeds and
output names used for `runs/`. Defaults: 50 epochs, batch size 256 (PIM: 200
epochs, SGD, as in the PIM paper).

## LLM-authored profiles

The class descriptions used for label-free matching and for prototype pre-training
are defined in `pseudo_labels.py`. `PROFILE_GENERATION.md` documents how they were
produced (model, decoding settings and the exact prompt) and includes a
reproduction check.
