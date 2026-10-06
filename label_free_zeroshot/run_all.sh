#!/usr/bin/env bash
# Reproduce every run in runs/ (paper Tables VI and VIII) from scratch.
#
# 5 pre-training seeds (p) x 5 fine-tuning seeds (f) per method and dataset.
# Output: runs/fulldata_<dataset>_<tag>_p<p>_f<f>.json
# Checkpoints: runs/seed_<p>/
#
# Needs the three datasets under $DATA (see README.md). Sequential and slow;
# split the loops across GPUs/jobs as convenient.
set -euo pipefail
DATA=${DATA:-data}
SEEDS="0 1 2 3 4"

for P in $SEEDS; do
  CK=runs/seed_${P}
  # ---- pre-training ------------------------------------------------------
  # Feature regression (8 physical features per sensor)
  python train.py        --data_root "$DATA" --out_dir "$CK" --seed "$P"   # mmfit_best.pt
  python train_wear.py   --dataset wear --data_root "$DATA" --out_dir "$CK" --seed "$P"   # wear_wear_best.pt
  # Prototype pre-training: prototypes only / features then prototypes
  for DS in mmfit wear; do
    python train_scores.py --dataset $DS --data_root "$DATA" --out_dir "$CK" --seed "$P"
  done
  python train_scores.py --dataset mmfit --data_root "$DATA" --out_dir "$CK" --seed "$P" --init_from "$CK/mmfit_best.pt"
  python train_scores.py --dataset wear  --data_root "$DATA" --out_dir "$CK" --seed "$P" --init_from "$CK/wear_wear_best.pt"
  # PIM baseline
  for DS in mmfit wear; do
    python train_pim.py --dataset $DS --data_root "$DATA" --out_dir "$CK" --seed "$P"
  done
  # PAMAP2: leave-one-subject-out, one checkpoint per fold
  for FOLD in 0 1 2 3 4 5 6 7 8; do
    python train_pamap2.py --fold $FOLD --data_root "$DATA" --out_dir "$CK" --seed "$P"
    python train_scores.py --dataset pamap2 --fold $FOLD --data_root "$DATA" --out_dir "$CK" --seed "$P"
    python train_scores.py --dataset pamap2 --fold $FOLD --data_root "$DATA" --out_dir "$CK" --seed "$P" \
        --init_from "$CK/pamap2_fold${FOLD}_best.pt"
    python train_pim.py --dataset pamap2 --fold $FOLD --data_root "$DATA" --out_dir "$CK" --seed "$P"
  done

  # ---- evaluation ----------------------------------------------------------
  for F in $SEEDS; do
    ev() {  # ev <dataset> <type:checkpoint> <tag>
      python evaluate_fulldata.py --dataset "$1" --method "$2" --data_root "$DATA" \
          --out_dir runs --seed "$F" --out_suffix "$1_$3_p${P}_f${F}"
    }
    ev mmfit  "feat:$CK/mmfit_best.pt"                               mmfit
    ev mmfit  "scores:$CK/scores_mmfit_best.pt"                      scores_mmfit
    ev mmfit  "scores:$CK/scores_mmfit_from_feat_best.pt"            scores_mmfit_from_feat
    ev mmfit  "pim:$CK/pim_mmfit_best.pt"                            pim_mmfit
    ev wear   "feat:$CK/wear_wear_best.pt"                           wear_wear
    ev wear   "scores:$CK/scores_wear_best.pt"                       scores_wear
    ev wear   "scores:$CK/scores_wear_from_feat_best.pt"             scores_wear_from_feat
    ev wear   "pim:$CK/pim_wear_best.pt"                             pim_wear
    # PAMAP2: {fold} is substituted per LOSO fold by evaluate_fulldata.py
    ev pamap2 "feat:$CK/pamap2_fold{fold}_best.pt"                   feat
    ev pamap2 "scores:$CK/scores_pamap2_fold{fold}_best.pt"          scores
    ev pamap2 "scores:$CK/scores_pamap2_fold{fold}_from_feat_best.pt" scores_from_feat
    ev pamap2 "pim:$CK/pim_pamap2_fold{fold}_best.pt"                pim
  done
done

# ZS-Feat (no network, deterministic) -> runs/zs_feat_<dataset>.json
python zero_shot_from_feat.py --data_root "$DATA"
