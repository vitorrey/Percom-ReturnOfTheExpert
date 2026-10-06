# Return of the Expert: Classical Features as Self-Supervision for Human Activity Recognition

Code and stored results for the paper *"Return of the Expert: Classical Features
as Self-Supervision for Human Activity Recognition."* This repository is released
anonymously for peer review.

An encoder is pre-trained to **regress a bank of hand-crafted ("classical")
features** from a raw single-sensor, three-axis accelerometer window, with no labels
and no data augmentation. After pre-training the regression head is discarded and
the encoder is fine-tuned on downstream labels; at inference only the raw signal is
used. Evaluation follows the HARBench protocol (14 unlabelled pre-training
datasets, 18 labelled downstream datasets, five generalization axes).

The code is built on the public HARBench codebase (data preprocessing, ResNet18
backbone, fine-tuning and benchmark drivers). That codebase also contains other
SSL methods and backbones; **only the models listed below are used in the paper.**

## Paper names vs. code names

| Paper | Code (`--method` / `--model`) | Feature module | Dim |
|---|---|---|---|
| Statistical bank | `featstat_norot` | `features_statistical.py` | 63 |
| Biomechanical bank | `featinv_norot` | `features_invariant.py` | 63 |
| Spectral-PCA bank | `featgoogle_norot` | `features_google.py` | 11 |
| Union bank | `featunion_norot` | `features_union.py` | 134 |
| Rotation variants (Table V only) | `featstat`, `featinv`, `featgoogle`, `featunion` | same as above | |
| MTL baseline | `mtl` | HARBench released weights | |
| TimeChannel baseline | `timechannel` | HARBench released weights (time + channel masked reconstruction) | |
| Cross-dataset transfer axis | `zeroshot` (`--eval zeroshot`, `results/benchmark/*_zeroshot/`) | | |

The `_norot` models are the method as reported in the paper (no augmentation).
Model names **without** `_norot` were pre-trained with random 3D rotation and are
used only for the rotation ablation (Table V).

**Terminology.** HARBench calls the leave-one-dataset-out axis "zero-shot", and the
code keeps that name. The paper reports it as **cross-dataset transfer**, because
it still fine-tunes on the remaining datasets before testing the held-out one. It
is unrelated to the **label-free recognition** of Section VI, which uses no
downstream labels (see `label_free_zeroshot/`).

## Where each paper result comes from

| Paper item | Script | Source data |
|---|---|---|
| Tables I–V | `python paper/gen_tables.py` | `results/benchmark/`, `results/features_alone.json` |
| Wilcoxon tests (Sections V-A, V-B) | `python paper/significance.py` | `results/benchmark/` |
| Fig. 1 (method diagram) | drawn in the LaTeX source | |
| Fig. 2 (radar, five axes) | `python paper/make_radar.py` | values from Table I |
| Fig. 3 (few-shot, features-alone) | `python make_figures.py` | `results/benchmark/`, `results/features_alone.json` |
| Baseline reproduction check (Section IV) | `python aggregate_results.py` | `results/benchmark/` |
| Tables VI–VIII | see [`label_free_zeroshot/README.md`](label_free_zeroshot/README.md) | `label_free_zeroshot/runs/` |

## Repository layout

```
features_statistical.py         statistical bank            (63-d)
features_invariant.py           biomechanical bank          (63-d)
features_google.py              spectral-PCA bank           (11-d)
features_union.py               union of the three banks    (134-d)

pretrain.py                     feature-regression pre-training (ResNet18 + regression head)
finetune.py                     downstream fine-tuning (HARBench)
run_benchmark.py                HARBench five-axis evaluation driver
aggregate_results.py            aggregates result shards; prints the five-axis summary
evaluate_features_benchmark.py  features-alone reference (class-balanced RF/MLP on the features)
make_figures.py                 Fig. 3
preprocess.py, verify_datasets.py   HARBench dataset download / preprocessing

paper/                          table, significance and radar-chart scripts
results/                        stored evaluation output (datasets and weights NOT included)
src/                            backbones, data loading, metrics (HARBench)
label_free_zeroshot/            Tables VI-VIII (Sections V-F and VI; 1D-CNN, 3 datasets) (1D-CNN, 3 datasets)
Dockerfile                      reproducibility image
```

## Installation

```bash
pip install torch            # pick the CUDA build for your GPU
pip install -r requirements.txt
# or:
docker build -t roe-percom .
```

## Reproducing the paper numbers from the stored results (no training)

```bash
python paper/gen_tables.py      # Tables I-V (LaTeX bodies) + paper/fig_stats.json
python paper/significance.py    # paired Wilcoxon tests quoted in the text
python make_figures.py          # Fig. 3 -> results/figures/featreg_results.{pdf,png}
python paper/make_radar.py      # Fig. 2 (radar) -> paper/figs/radar_axes.{pdf,png}
python aggregate_results.py     # five-axis summary + baseline reproduction check
```

## Reproducing from scratch

Requires the HARBench datasets, downloaded and preprocessed with `preprocess.py`.

**1. Pre-training corpus (14 datasets).** KDDI Kitchen is stored as two
recordings (`kddi_kitchen_left`, `kddi_kitchen_right`) that together form the
single KDDI Kitchen dataset, so 15 names cover the 14 datasets. The list is
`DEFAULT_PRETRAIN_DATASETS` in `pretrain.py`.

```bash
python preprocess.py --dataset nhanes adlrd chad capture24 dog har70plus hhar imsb \
    kddi_kitchen_left kddi_kitchen_right motionsense opportunity sbrhapt tmd wisdm --download
```

**2. Downstream datasets (18).**

```bash
python preprocess.py --dataset dsads pamap2 mhealth realdisp mex forthtrace harth imwsha \
    paal realworld selfback ucaehar uschad ward lara openpack exoskeletons vtt_coniot --download
```

**3. Pre-train the four banks.** Checkpoints are written to
`results/pretrain/<timestamp>_<method>/best.pth`.

```bash
for M in featstat featinv featgoogle featunion; do
  python pretrain.py --method ${M}_norot --device cuda:0   # paper method (no augmentation)
  python pretrain.py --method ${M}       --device cuda:0   # rotation variant, Table V only
done
```

**4. Put the checkpoints where the benchmark expects them.** The baselines use
HARBench's released pre-trained weights (`mtl.pth`, `timechannel.pth`, shipped in
the `pretrained/` directory of the HARBench repository). Only fine-tuning and
evaluation of the baselines are re-run here.

```bash
mkdir -p pretrained
for M in featstat featinv featgoogle featunion featstat_norot featinv_norot featgoogle_norot featunion_norot; do
  cp "$(ls -d results/pretrain/*_${M} | tail -1)/best.pth" pretrained/${M}.pth
done
cp /path/to/HARBench/pretrained/{mtl,timechannel}.pth pretrained/
```

**5. Evaluate on all five axes.**

```bash
for M in featstat_norot featinv_norot featgoogle_norot featunion_norot \
         featstat featinv featgoogle featunion mtl timechannel; do
  python run_benchmark.py --model $M --eval all
done
```

**6. Features-alone reference and aggregation.**

```bash
python evaluate_features_benchmark.py   # -> results/features_alone.json
python paper/gen_tables.py
python paper/significance.py
```

## Settings used for the stored results

- **Windows:** single sensor, three axes, 5 s at 30 Hz (150 samples).
- **Pre-training:** Adam, learning rate 1e-4, weight decay 1e-5, up to 200 epochs
  with early stopping (patience 30), best-validation checkpoint. The target
  normaliser (winsorise to [p1, p99], standardise, clamp at ±10) is fitted on
  pre-training data only (`fit_feature_normalizer` in `pretrain.py`).
- **Fine-tuning** (`run_benchmark.py` defaults, recorded in every
  `results.json`): Adam, learning rate 1e-4, weight decay 1e-5, cosine annealing,
  batch size 128, up to 100 epochs, early stopping (patience 5), best-validation
  selection, encoder and classifier updated end-to-end.
- **Protocol:** 4-fold user cross-validation, macro-F1 averaged over folds.
  Few-shot ratios are 1/2/5/10/20/50% of each fold's training split. Cross-dataset
  transfer is leave-one-dataset-out over dsads/mhealth/pamap2 with 4 seeds.
