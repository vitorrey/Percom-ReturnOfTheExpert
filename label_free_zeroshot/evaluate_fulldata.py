"""
evaluate_fulldata.py
--------------------
Full-data evaluation on WEAR, MM-Fit, or PAMAP2 (paper Tables VI and VIII).
Writes one runs/fulldata_<dataset>_<tag>.json per (pre-training seed, fine-tuning
seed) with the conditions below:

  zero_shot           label-free recognition (scores method: argmax of the
                      predicted class-profile scores = ZS-Net in Table VIII)
  supervised_scratch  classifier trained from a randomly initialised Encoder1D
                      ("From scratch" in Table VI)
  supervised_finetune encoder initialised from the pre-trained checkpoint and
                      fine-tuned end-to-end (Tables VI and VIII)

Supported method types
----------------------
  feat     feature regression       (train.py / train_wear.py / train_pamap2.py)
  scores   prototype pre-training   (train_scores.py)
  pim      PIM pseudo-label baseline (train_pim.py)

Usage
-----
  python evaluate_fulldata.py --method feat:runs/seed_0/wear_wear_best.pt --dataset wear
  python evaluate_fulldata.py --method scores:runs/seed_0/scores_mmfit_best.pt --dataset mmfit
  python evaluate_fulldata.py --method pim:runs/seed_0/pim_wear_best.pt --dataset wear
"""

import os
import argparse
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Models
from model import build_model, Encoder1D
from model_pim import build_pim
from model_scores import build_score_model

# Datasets — WEAR
from dataset_wear import (
    load_wear_labeled_windows, WEARDataset,
    WEAR_CLASSES, WEAR_CLASS_TO_IDX, WEAR_LABEL_TO_PROFILE_KEY,
    WEAR_SPLITS, WEAR_LIMB_GROUPS,
    WINDOW_FRAMES as WEAR_WINDOW_FRAMES,
    STRIDE_FRAMES as WEAR_STRIDE_FRAMES,
)
# Datasets — MMFIT
from dataset import (
    load_labeled_windows, MMFitDataset, ProfileNormalizer, PROFILE_KEYS,
    MMFIT_CLASSES, CLASS_TO_IDX, SENSOR_NAMES,
    WINDOW_FRAMES as MMFIT_WINDOW_FRAMES,
    STRIDE_FRAMES as MMFIT_STRIDE_FRAMES,
    MMFIT_SPLITS, MMFIT_LIMB_GROUPS,
)
# Datasets — PAMAP2
from dataset_pamap2 import (
    load_pamap2_labeled_windows, pamap2_loso_splits,
    PAMAP2_CLASSES,
    WINDOW_FRAMES as PAMAP2_WINDOW_FRAMES,
    STRIDE_FRAMES as PAMAP2_STRIDE_FRAMES,
)
# Pseudo-labels (for zero-shot profile matching)
from pseudo_labels import WEAR_ACTIVITIES, MMFIT_ACTIVITIES
from dataset_scores import N_WEAR_ACTIVITIES, N_MMFIT_ACTIVITIES, N_PAMAP2_ACTIVITIES

N_SENSORS       = 4   # default for wear/mmfit; pamap2 uses 3
N_FEATURES      = len(PROFILE_KEYS)   # 8 per sensor
EMBED_DIM       = 256

# MMFIT dataset class → ActivityProfile key
_MMFIT_DATASET_TO_PROFILE_KEY = {
    "bicep_curls":             "bicep_curls",
    "dumbbell_rows":           "dumbbell_rows",
    "dumbbell_shoulder_press": "shoulder_press",
    "jumping_jacks":           "jumping_jacks",
    "lateral_shoulder_raises": "lateral_raises",
    "lunges":                  "lunges",
    "pushups":                 "pushups",
    "situps":                  "situps",
    "squats":                  "squats",
    "tricep_extensions":       "tricep_extensions",
    "null":                    "null",
}

# Score dim → PAMAP2 class index (scores and classes share the same order)
PAMAP2_SCORE_DIM_TO_CLASS = list(range(N_PAMAP2_ACTIVITIES))

# Score dim → WEAR class index
_WEAR_PROFILE_KEY_TO_CLS = {
    prof: WEAR_CLASS_TO_IDX[cls]
    for cls, prof in WEAR_LABEL_TO_PROFILE_KEY.items()
    if cls in WEAR_CLASS_TO_IDX
}
WEAR_SCORE_DIM_TO_CLASS = [
    _WEAR_PROFILE_KEY_TO_CLS.get(k, 0) for k in WEAR_ACTIVITIES.keys()
]

# Score dim → MMFIT class index
_MMFIT_PROFILE_KEY_TO_CLS = {
    prof: CLASS_TO_IDX[cls]
    for cls, prof in _MMFIT_DATASET_TO_PROFILE_KEY.items()
    if cls in CLASS_TO_IDX
}
MMFIT_SCORE_DIM_TO_CLASS = [
    _MMFIT_PROFILE_KEY_TO_CLS.get(k, CLASS_TO_IDX["null"])
    for k in MMFIT_ACTIVITIES.keys()
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def windows_to_tensor(windows: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(windows.transpose(0, 2, 1).copy()).float()


def report(name, classes, y_true, y_pred) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"  Accuracy         : {acc:.3f}")
    print(f"  Balanced accuracy: {bacc:.3f}")
    print(f"  Macro F1         : {f1:.3f}")
    print("  Per-class accuracy:")
    for i, cls in enumerate(classes):
        mask = y_true == i
        if not mask.any():
            continue
        print(f"    {cls:35s}: {accuracy_score(y_true[mask], y_pred[mask]):.3f}"
              f"  (n={mask.sum()})")
    return {"acc": float(acc), "bacc": float(bacc), "f1": float(f1)}


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_data(dataset, data_root):
    """Returns (train_w, train_y, val_w, val_y, test_w, test_y, classes, window_frames)."""
    if dataset == "wear":
        train_w, train_y, _ = load_wear_labeled_windows(
            data_root, WEAR_SPLITS["train"], WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES)
        val_w,   val_y,   _ = load_wear_labeled_windows(
            data_root, WEAR_SPLITS["val"],   WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES)
        test_w,  test_y,  _ = load_wear_labeled_windows(
            data_root, WEAR_SPLITS["test"],  WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES)
        return train_w, train_y, val_w, val_y, test_w, test_y, WEAR_CLASSES, WEAR_WINDOW_FRAMES
    elif dataset == "mmfit":
        data_dir = os.path.join(data_root, "mm-fit")
        train_w, train_y, _ = load_labeled_windows(
            data_dir, MMFIT_SPLITS["train"], MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES, SENSOR_NAMES)
        val_w,   val_y,   _ = load_labeled_windows(
            data_dir, MMFIT_SPLITS["val"],   MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES, SENSOR_NAMES)
        test_w,  test_y,  _ = load_labeled_windows(
            data_dir, MMFIT_SPLITS["test"],  MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES, SENSOR_NAMES)
        return train_w, train_y, val_w, val_y, test_w, test_y, MMFIT_CLASSES, MMFIT_WINDOW_FRAMES
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# -----------------------------------------------------------------------
# Encoder loading
# -----------------------------------------------------------------------

def _n_sensors_from_ckpt(ckpt: dict) -> int:
    """Infer n_sensors from checkpoint weight shapes."""
    state = ckpt.get("model_state", {})
    # Encoder1D: stem conv weight
    for key in ("encoder.stem.block.0.weight", "stem.block.0.weight"):
        if key in state:
            return state[key].shape[1] // 3
    return N_SENSORS   # fallback


def _enc_from_ckpt(ckpt: dict, n_sens: int, device) -> tuple:
    """Return a fresh (Encoder1D, embed_dim) sized for the checkpoint."""
    return Encoder1D(in_channels=n_sens * 3, embed_dim=EMBED_DIM).to(device), EMBED_DIM


def load_encoder(method_type, ckpt_path, device, window_frames):
    """Returns (encoder, embed_dim). encoder: (B,C,W) → (B, embed_dim)."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    if method_type == "feat":
        n_sens        = _n_sensors_from_ckpt(ckpt)
        enc, enc_dim  = _enc_from_ckpt(ckpt, n_sens, device)
        model         = build_model(n_sensors=n_sens, embed_dim=enc_dim,
                                    encoder=enc).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model.encoder, enc_dim

    if method_type == "scores":
        n_sens        = _n_sensors_from_ckpt(ckpt)
        n_acts        = ckpt.get("n_activities", N_WEAR_ACTIVITIES)
        enc, enc_dim  = _enc_from_ckpt(ckpt, n_sens, device)
        model         = build_score_model(n_sens, n_acts, embed_dim=enc_dim,
                                          encoder=enc).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model.encoder, enc_dim

    if method_type == "pim":
        dataset = ckpt.get("args", {}).get("dataset", "wear")
        n_sens  = _n_sensors_from_ckpt(ckpt)
        n_bins  = ckpt.get("args", {}).get("n_bins", 11)
        model   = build_pim(dataset, n_sens, EMBED_DIM, n_bins).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model.encoder, EMBED_DIM

    raise ValueError(f"Unknown method type: {method_type!r}")


# -----------------------------------------------------------------------
# Zero-shot (feat only)
# -----------------------------------------------------------------------

def _build_profile_matrix(normalizer, train_profiles, classes, activities,
                           label_to_key, limb_groups):
    """Build per-limb (N_CLASSES, N_FEATURES) normalised profile matrix."""
    LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}
    N         = train_profiles.shape[0]
    n_classes = len(classes)
    result    = {}

    for limb, sensor_indices in limb_groups.items():
        per_sensor = train_profiles.reshape(N, N_SENSORS, N_FEATURES)
        limb_raw   = per_sensor[:, sensor_indices, :].mean(axis=1)

        lo = np.percentile(limb_raw, 33, axis=0)
        hi = np.percentile(limb_raw, 67, axis=0)
        mu  = limb_raw.mean(axis=0)
        std = limb_raw.std(axis=0) + 1e-8

        bin_means = np.zeros((3, N_FEATURES), dtype=np.float32)
        for d in range(N_FEATURES):
            col = limb_raw[:, d]
            l, h = lo[d], hi[d]
            bin_means[0, d] = col[col <  l].mean() if (col <  l).any() else l
            bin_means[1, d] = col[(col >= l) & (col < h)].mean() \
                              if ((col >= l) & (col < h)).any() else (l + h) / 2
            bin_means[2, d] = col[col >= h].mean() if (col >= h).any() else h
        bin_norm = (bin_means - mu) / std

        mat = np.zeros((n_classes, N_FEATURES), dtype=np.float32)
        for cls_i, cls_name in enumerate(classes):
            act_key = label_to_key.get(cls_name, cls_name)
            if act_key not in activities:
                continue
            act = activities[act_key]
            if limb not in act.limbs:
                continue
            for fi, fk in enumerate(PROFILE_KEYS):
                mat[cls_i, fi] = bin_norm[LEVEL_MAP[act.limbs[limb][fk]], fi]
        result[limb] = mat
    return result


@torch.no_grad()
def zero_shot_feat(full_model, test_tensor, normalizer, train_profiles,
                   dataset, data_root, device, batch_size=512):
    """Per-limb L2 profile matching for feat-type models."""
    full_model.eval()
    n_classes  = len(WEAR_CLASSES) if dataset == "wear" else len(MMFIT_CLASSES)
    classes    = WEAR_CLASSES if dataset == "wear" else MMFIT_CLASSES
    activities = WEAR_ACTIVITIES if dataset == "wear" else MMFIT_ACTIVITIES
    limb_groups = WEAR_LIMB_GROUPS if dataset == "wear" else MMFIT_LIMB_GROUPS
    label_to_key = (WEAR_LABEL_TO_PROFILE_KEY if dataset == "wear"
                    else _MMFIT_DATASET_TO_PROFILE_KEY)

    mats    = _build_profile_matrix(normalizer, train_profiles, classes,
                                     activities, label_to_key, limb_groups)
    tensors = {l: torch.from_numpy(m).to(device) for l, m in mats.items()}

    preds = []
    for (x,) in DataLoader(TensorDataset(test_tensor),
                            batch_size=batch_size, shuffle=False):
        x    = x.to(device)
        pred = full_model(x).view(-1, N_SENSORS, N_FEATURES)
        dist = torch.zeros(x.shape[0], n_classes, device=device)
        for limb, sidx in limb_groups.items():
            lp  = pred[:, sidx, :].mean(dim=1)
            mat = tensors[limb]
            dist += (lp.unsqueeze(1) - mat.unsqueeze(0)).pow(2).sum(dim=-1)
        preds.append(dist.argmin(dim=-1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def zero_shot_scores(score_model, test_tensor, dataset, both_mode, device,
                     batch_size=512):
    """Argmax of predicted activity scores → class index."""
    score_model.eval()
    if dataset == "wear":
        n_classes, dim_to_class = len(WEAR_CLASSES),   WEAR_SCORE_DIM_TO_CLASS
    elif dataset == "pamap2":
        n_classes, dim_to_class = len(PAMAP2_CLASSES), PAMAP2_SCORE_DIM_TO_CLASS
        both_mode = False
    else:
        n_classes, dim_to_class = len(MMFIT_CLASSES),  MMFIT_SCORE_DIM_TO_CLASS

    preds = []
    for (x,) in DataLoader(TensorDataset(test_tensor),
                            batch_size=batch_size, shuffle=False):
        scores = score_model(x.to(device))
        if both_mode:
            scores = scores[:, N_MMFIT_ACTIVITIES:] if dataset == "wear" \
                     else scores[:, :N_MMFIT_ACTIVITIES]
        cls_scores = torch.zeros(scores.shape[0], n_classes, device=device)
        for di, ci in enumerate(dim_to_class):
            cls_scores[:, ci] = torch.max(cls_scores[:, ci], scores[:, di])
        preds.append(cls_scores.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# -----------------------------------------------------------------------
# Supervised training
# -----------------------------------------------------------------------

class _Classifier(nn.Module):
    def __init__(self, enc, embed_dim, n_classes, dropout=0.3,
                 head_type="mlp", freeze_encoder=False):
        super().__init__()
        self.enc            = enc
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad_(False)
        if head_type == "linear":
            # True linear probe: a single linear layer on frozen features.
            self.head = nn.Linear(embed_dim, n_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, n_classes),
            )

    def forward(self, x):
        if self.freeze_encoder:
            # Keep the encoder in eval mode so BatchNorm running stats and
            # dropout are frozen — only the head sees gradients.
            self.enc.eval()
            with torch.no_grad():
                z = self.enc(x)
        else:
            z = self.enc(x)
        return self.head(z)


def supervised_train(train_windows, train_labels, val_windows, val_labels,
                     encoder, embed_dim, n_classes, device, label="",
                     epochs=50, batch_size=256, lr=1e-3, encoder_lr_factor=1.0,
                     freeze_encoder=False, head_type="mlp"):
    model = _Classifier(copy.deepcopy(encoder), embed_dim, n_classes,
                        head_type=head_type, freeze_encoder=freeze_encoder).to(device)

    if freeze_encoder:
        # Linear probing / frozen-feature training: optimise the head only.
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr,
                                      weight_decay=1e-4)
    elif encoder_lr_factor < 1.0:
        optimizer = torch.optim.AdamW([
            {"params": model.enc.parameters(),  "lr": lr * encoder_lr_factor},
            {"params": model.head.parameters(), "lr": lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = torch.from_numpy(1.0 / counts).to(device)
    w = w / w.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=w)

    x_train = windows_to_tensor(train_windows).to(device)
    y_train = torch.from_numpy(train_labels).long().to(device)
    x_val   = windows_to_tensor(val_windows).to(device)
    loader  = DataLoader(TensorDataset(x_train, y_train),
                         batch_size=batch_size, shuffle=True, drop_last=True)

    best_bacc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val).argmax(dim=1).cpu().numpy()
        val_bacc = balanced_accuracy_score(val_labels, val_preds)
        if val_bacc > best_bacc:
            best_bacc  = val_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  [{label}] epoch {epoch:3d}/{epochs}  val_bacc={val_bacc:.3f}")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, windows_tensor, device, batch_size=512):
    model.eval()
    preds = []
    for (x,) in DataLoader(TensorDataset(windows_tensor),
                            batch_size=batch_size, shuffle=False):
        preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# -----------------------------------------------------------------------
# PAMAP2 LOSO evaluation
# -----------------------------------------------------------------------

def _evaluate_pamap2_loso(args, method_type, ckpt_path, device):
    """Leave-one-subject-out evaluation on PAMAP2. Averages metrics across 9 folds.

    ckpt_path may contain the literal substring '{fold}', which is replaced with
    the fold index for each fold (enabling fold-specific pre-trained checkpoints).
    """
    n_classes = len(PAMAP2_CLASSES)
    folds     = pamap2_loso_splits()

    fold_specific = "{fold}" in ckpt_path
    if not fold_specific:
        encoder, embed_dim = load_encoder(method_type, ckpt_path, device, PAMAP2_WINDOW_FRAMES)

    scratch_metrics      = []
    finetune_metrics     = []
    linear_probe_metrics = []
    zero_shot_metrics    = []   # populated for scores method only

    for fold_idx, fold in enumerate(folds):
        print(f"\n── Fold {fold_idx+1}/9  "
              f"test={fold['test'][0]}  val={fold['val'][0]} ──")

        fold_ckpt = ckpt_path.replace("{fold}", str(fold_idx)) if fold_specific else ckpt_path
        if fold_specific:
            encoder, embed_dim = load_encoder(method_type, fold_ckpt, device, PAMAP2_WINDOW_FRAMES)

        train_w, train_y, _ = load_pamap2_labeled_windows(
            args.data_root, fold["train"], PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES)
        val_w,   val_y,   _ = load_pamap2_labeled_windows(
            args.data_root, fold["val"],   PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES)
        test_w,  test_y,  _ = load_pamap2_labeled_windows(
            args.data_root, fold["test"],  PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES)
        print(f"  train={len(train_w):,}  val={len(val_w):,}  test={len(test_w):,}")

        test_tensor = windows_to_tensor(test_w)

        # Zero-shot (scores only)
        if method_type == "scores":
            ckpt_s      = torch.load(fold_ckpt, map_location=device, weights_only=False)
            n_sens      = _n_sensors_from_ckpt(ckpt_s)
            n_acts      = ckpt_s.get("n_activities", N_PAMAP2_ACTIVITIES)
            score_model = build_score_model(n_sens, n_acts, EMBED_DIM).to(device)
            score_model.load_state_dict(ckpt_s["model_state"])
            zs_preds    = zero_shot_scores(score_model, test_tensor, "pamap2", False, device)
            m = {"acc":  float(accuracy_score(test_y, zs_preds)),
                 "bacc": float(balanced_accuracy_score(test_y, zs_preds)),
                 "f1":   float(f1_score(test_y, zs_preds, average="macro", zero_division=0))}
            zero_shot_metrics.append(m)
            print(f"  zero-shot F1={m['f1']:.3f}  BAcc={m['bacc']:.3f}")

        # Scratch
        scratch_enc = Encoder1D(in_channels=3 * 3, embed_dim=EMBED_DIM).to(device)
        scratch_model = supervised_train(
            train_w, train_y, val_w, val_y,
            encoder=scratch_enc, embed_dim=EMBED_DIM, n_classes=n_classes,
            device=device, label=f"scratch-fold{fold_idx+1}",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            encoder_lr_factor=1.0)
        preds = predict(scratch_model, test_tensor, device)
        m = {"acc":  float(accuracy_score(test_y, preds)),
             "bacc": float(balanced_accuracy_score(test_y, preds)),
             "f1":   float(f1_score(test_y, preds, average="macro", zero_division=0))}
        scratch_metrics.append(m)
        print(f"  scratch  F1={m['f1']:.3f}  BAcc={m['bacc']:.3f}")

        # Linear probe (frozen encoder + single linear head)
        if args.linear_probe:
            lp_model = supervised_train(
                train_w, train_y, val_w, val_y,
                encoder=encoder, embed_dim=embed_dim, n_classes=n_classes,
                device=device, label=f"linprobe-fold{fold_idx+1}",
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                freeze_encoder=True, head_type="linear")
            preds = predict(lp_model, test_tensor, device)
            m = {"acc":  float(accuracy_score(test_y, preds)),
                 "bacc": float(balanced_accuracy_score(test_y, preds)),
                 "f1":   float(f1_score(test_y, preds, average="macro", zero_division=0))}
            linear_probe_metrics.append(m)
            print(f"  linprobe F1={m['f1']:.3f}  BAcc={m['bacc']:.3f}")

        # Fine-tune
        ft_model = supervised_train(
            train_w, train_y, val_w, val_y,
            encoder=encoder, embed_dim=embed_dim, n_classes=n_classes,
            device=device, label=f"finetune-fold{fold_idx+1}",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            encoder_lr_factor=0.1)
        preds = predict(ft_model, windows_to_tensor(test_w), device)
        m = {"acc":  float(accuracy_score(test_y, preds)),
             "bacc": float(balanced_accuracy_score(test_y, preds)),
             "f1":   float(f1_score(test_y, preds, average="macro", zero_division=0))}
        finetune_metrics.append(m)
        print(f"  finetune F1={m['f1']:.3f}  BAcc={m['bacc']:.3f}")

    def _mean(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    results = {
        "supervised_scratch":  _mean(scratch_metrics),
        "supervised_finetune": _mean(finetune_metrics),
        "per_fold": {
            "supervised_scratch":  scratch_metrics,
            "supervised_finetune": finetune_metrics,
        },
    }
    if linear_probe_metrics:
        results["linear_probe"] = _mean(linear_probe_metrics)
        results["per_fold"]["linear_probe"] = linear_probe_metrics
    if zero_shot_metrics:
        results["zero_shot"] = _mean(zero_shot_metrics)
        results["per_fold"]["zero_shot"] = zero_shot_metrics

    print(f"\n{'═'*52}")
    print(f"  PAMAP2 LOSO — mean across 9 folds")
    print(f"{'─'*52}")
    for cond in ("zero_shot", "linear_probe",
                 "supervised_scratch", "supervised_finetune"):
        if cond not in results:
            continue
        m = results[cond]
        print(f"  {cond:<22s}  F1={m['f1']:.3f}  BAcc={m['bacc']:.3f}")

    tag = args.out_suffix or f"{method_type}_pamap2"
    if args.out_suffix:
        out_name = f"fulldata_{args.out_suffix}.json"
    else:
        out_name = f"fulldata_pamap2_{tag}.json"
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    method_type, ckpt_path = args.method.split(":", 1)

    print(f"Method  : {method_type}")
    print(f"Ckpt    : {ckpt_path}")
    print(f"Dataset : {args.dataset}")

    # ---- PAMAP2: LOSO evaluation -----------------------------------------
    if args.dataset == "pamap2":
        _evaluate_pamap2_loso(args, method_type, ckpt_path, device)
        return

    # ---- Load data -------------------------------------------------------
    print("\nLoading labeled windows...")
    (train_w, train_y, val_w, val_y,
     test_w, test_y, classes, window_frames) = load_data(args.dataset, args.data_root)
    n_classes   = len(classes)
    test_tensor = windows_to_tensor(test_w)
    print(f"  train: {len(train_w):,}  val: {len(val_w):,}  test: {len(test_w):,}")

    # ---- Load pre-trained encoder ----------------------------------------
    encoder, embed_dim = load_encoder(method_type, ckpt_path, device, window_frames)

    results = {}
    step = 1

    # ---- 1. Zero-shot (feat / scores only) --------------------------------
    if method_type == "feat":
        print(f"\n[{step}] Zero-shot (per-limb profile matching)...")
        step += 1
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        norm_path = ckpt_path.replace("_best.pt", "_normalizer.pkl")
        normalizer = ProfileNormalizer.load(norm_path)

        n_sens     = _n_sensors_from_ckpt(ckpt)
        enc, enc_dim = _enc_from_ckpt(ckpt, n_sens, device)
        full_model = build_model(n_sensors=n_sens, embed_dim=enc_dim,
                                 encoder=enc).to(device)
        full_model.load_state_dict(ckpt["model_state"])

        # Collect training profiles for threshold computation
        cache_dir = os.path.join(args.data_root, "cache")
        if args.dataset == "wear":
            ds = WEARDataset(args.data_root, WEAR_SPLITS["train"],
                             WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                             normalizer=None, cache_dir=cache_dir)
        else:
            data_dir = os.path.join(args.data_root, "mm-fit")
            ds = MMFitDataset(data_dir, MMFIT_SPLITS["train"],
                              MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES, SENSOR_NAMES,
                              normalizer=None, cache_dir=cache_dir)
        train_profiles = ds.raw_profiles()

        zs_preds = zero_shot_feat(full_model, test_tensor, normalizer,
                                   train_profiles, args.dataset, args.data_root, device)
        results["zero_shot"] = report("Zero-shot (profile matching)", classes, test_y, zs_preds)

    elif method_type == "scores":
        print(f"\n[{step}] Zero-shot (score argmax)...")
        step += 1
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved     = ckpt.get("args", {})
        both_mode = saved.get("dataset", "") == "both"
        n_acts    = ckpt.get("n_activities", N_WEAR_ACTIVITIES)

        n_sens      = _n_sensors_from_ckpt(ckpt)
        enc, enc_dim = _enc_from_ckpt(ckpt, n_sens, device)
        score_model = build_score_model(n_sens, n_acts, embed_dim=enc_dim,
                                        encoder=enc).to(device)
        score_model.load_state_dict(ckpt["model_state"])

        zs_preds = zero_shot_scores(score_model, test_tensor, args.dataset,
                                     both_mode, device)
        results["zero_shot"] = report("Zero-shot (score argmax)", classes, test_y, zs_preds)

    # ---- 2. Supervised from scratch ---------------------------------------
    print(f"\n[{step}] Supervised from scratch...")
    step += 1
    scratch_enc = Encoder1D(in_channels=N_SENSORS * 3, embed_dim=EMBED_DIM).to(device)
    scratch_model = supervised_train(
        train_w, train_y, val_w, val_y,
        encoder=scratch_enc, embed_dim=EMBED_DIM, n_classes=n_classes,
        device=device, label="scratch",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        encoder_lr_factor=1.0)
    results["supervised_scratch"] = report(
        "Supervised (from scratch)", classes, test_y,
        predict(scratch_model, test_tensor, device))

    # ---- 3. Linear probe (frozen encoder, single linear head) -------------
    if args.linear_probe:
        print(f"\n[{step}] Linear probe (frozen encoder + linear head)...")
        step += 1
        lp_model = supervised_train(
            train_w, train_y, val_w, val_y,
            encoder=encoder, embed_dim=embed_dim, n_classes=n_classes,
            device=device, label="linear_probe",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            freeze_encoder=True, head_type="linear")
        results["linear_probe"] = report(
            "Linear probe (frozen encoder)", classes, test_y,
            predict(lp_model, test_tensor, device))

    # ---- 4. Fine-tuned ----------------------------------------------------
    print(f"\n[{step}] Fine-tuned from pre-trained encoder...")
    ft_model = supervised_train(
        train_w, train_y, val_w, val_y,
        encoder=encoder, embed_dim=embed_dim, n_classes=n_classes,
        device=device, label="finetune",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        encoder_lr_factor=0.1)
    results["supervised_finetune"] = report(
        "Supervised (fine-tuned)", classes, test_y,
        predict(ft_model, test_tensor, device))

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'═'*52}")
    print(f"  {method_type} — {args.dataset} — {os.path.basename(ckpt_path)}")
    print(f"{'─'*52}")
    print(f"  {'Method':<32s} {'Acc':>6}  {'BAcc':>6}  {'F1':>6}")
    for m, v in results.items():
        print(f"  {m:<32s} {v['acc']:>6.3f}  {v['bacc']:>6.3f}  {v['f1']:>6.3f}")

    if args.out_suffix:
        out_name = f"fulldata_{args.out_suffix}.json"
    else:
        tag = os.path.basename(ckpt_path).replace("_best.pt", "")
        out_name = f"fulldata_{args.dataset}_{method_type}_{tag}.json"
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-data evaluation (paper Tables VI and VIII)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--method",     required=True, metavar="TYPE:CKPT",
                   help="Method spec, e.g. feat:runs/seed_0/wear_wear_best.pt")
    p.add_argument("--dataset",    default="wear", choices=["wear", "mmfit", "pamap2"])
    p.add_argument("--data_root",  default="data")
    p.add_argument("--out_dir",    default="runs")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_suffix", default="",
                   help="If set, saves as fulldata_{out_suffix}.json")
    p.add_argument("--linear_probe", action="store_true",
                   help="Also run a frozen-encoder linear probe (not reported in the paper)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
