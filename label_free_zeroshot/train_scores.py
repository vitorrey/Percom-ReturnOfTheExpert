"""
train_scores.py
---------------
Prototype pre-training (paper Section VI, Table VIII).

The pre-training task: given a raw accelerometer window, predict how
closely its per-limb physical features match each LLM-authored class
profile's expected low / medium / high feature levels (one score in [0, 1]
per class profile; see pseudo_labels.py). No class labels are used.

  "Prototypes only"           : train from a random encoder.
  "Features then prototypes"  : --init_from a feature-regression checkpoint
                                (train.py / train_wear.py / train_pamap2.py).

Usage
-----
  python train_scores.py --dataset mmfit
  python train_scores.py --dataset wear
  python train_scores.py --dataset pamap2 --fold 0
  python train_scores.py --dataset mmfit --init_from runs/seed_0/mmfit_best.pt
"""

import os
import argparse
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    MMFIT_SPLITS,
    WINDOW_FRAMES as MMFIT_WINDOW_FRAMES,
    STRIDE_FRAMES as MMFIT_STRIDE_FRAMES,
)
from dataset_wear import (
    WEAR_SPLITS,
    WINDOW_FRAMES as WEAR_WINDOW_FRAMES,
    STRIDE_FRAMES as WEAR_STRIDE_FRAMES,
)
from dataset_pamap2 import (
    pamap2_loso_splits,
    WINDOW_FRAMES as PAMAP2_WINDOW_FRAMES,
    STRIDE_FRAMES as PAMAP2_STRIDE_FRAMES,
)
from dataset_scores import (
    WEARScoreDataset, MMFitScoreDataset, PAMAP2ScoreDataset,
    LimbScoreNormalizer,
    WEAR_ACTIVITIES, MMFIT_ACTIVITIES, PAMAP2_ACTIVITIES,
    N_WEAR_ACTIVITIES, N_MMFIT_ACTIVITIES, N_PAMAP2_ACTIVITIES,
    N_TOTAL_ACTIVITIES, MMFIT_MASK, WEAR_MASK,
)
from model_scores import build_score_model

N_SENSORS = 4   # default for wear/mmfit; pamap2 uses 3


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------

def build_datasets(args):
    """
    Returns (train_loaders, val_loaders, limb_normalizer, n_activities).

    For 'both', each loader list has two entries [MMFIT, WEAR].
    For single datasets, each is a single DataLoader.

    In 'both' mode the model outputs 30 dims; loss is masked per batch
    so that each dataset only contributes loss on its own score dims.

    For 'pamap2', LOSO fold args.fold is used: train on fold["train"][:-1],
    val on fold["train"][-1:].
    """
    mmfit_dir = os.path.join(args.data_root, "mm-fit")
    wear_dir  = args.data_root
    cache_dir = os.path.join(args.data_root, "cache")
    bsz = args.batch_size
    nw  = args.num_workers
    pin = (get_device().type != "mps")

    # ---- PAMAP2 (LOSO) --------------------------------------------------
    if args.dataset == "pamap2":
        fold        = pamap2_loso_splits()[args.fold]
        train_subjs = fold["train"][:-1]   # 6 subjects for pre-training
        val_subjs   = fold["train"][-1:]   # 1 subject for pre-train val

        print(f"  PAMAP2 fold {args.fold}  "
              f"train={train_subjs}  val={val_subjs}")

        p_train = PAMAP2ScoreDataset(args.data_root, train_subjs,
                                     PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES,
                                     cache_dir=cache_dir)
        p_val   = PAMAP2ScoreDataset(args.data_root, val_subjs,
                                     PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES,
                                     cache_dir=cache_dir)

        normalizer = LimbScoreNormalizer().fit(p_train.raw_limb_arrays())
        p_train.set_normalizer_and_activities(normalizer, PAMAP2_ACTIVITIES)
        p_val.set_normalizer_and_activities(normalizer, PAMAP2_ACTIVITIES)

        train_loaders = DataLoader(p_train, batch_size=bsz, shuffle=True,
                                   num_workers=nw, pin_memory=pin, drop_last=True)
        val_loaders   = DataLoader(p_val,   batch_size=bsz * 2, shuffle=False,
                                   num_workers=nw, pin_memory=pin)
        return train_loaders, val_loaders, normalizer, N_PAMAP2_ACTIVITIES

    # ---- build raw datasets (no normalizer yet) -------------------------
    if args.dataset in ("wear", "both"):
        print("  WEAR train subjects...")
        w_train = WEARScoreDataset(wear_dir, WEAR_SPLITS["train"],
                                   WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                                   cache_dir=cache_dir)
        w_val   = WEARScoreDataset(wear_dir, WEAR_SPLITS["val"],
                                   WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                                   cache_dir=cache_dir)

    if args.dataset in ("mmfit", "both"):
        print("  MMFIT train subjects...")
        m_train = MMFitScoreDataset(mmfit_dir, MMFIT_SPLITS["train"],
                                    MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES,
                                    cache_dir=cache_dir)
        m_val   = MMFitScoreDataset(mmfit_dir, MMFIT_SPLITS["val"],
                                    MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES,
                                    cache_dir=cache_dir)

    # ---- fit LimbScoreNormalizer on training limb arrays ----------------
    if args.dataset == "wear":
        combined_limb_arrays = w_train.raw_limb_arrays()
    elif args.dataset == "mmfit":
        combined_limb_arrays = m_train.raw_limb_arrays()
    else:  # both — stack arrays per shared limb name where possible
        mm_la = m_train.raw_limb_arrays()
        w_la  = w_train.raw_limb_arrays()
        # Use each dataset's own limbs; normalizer will have keys for both
        combined_limb_arrays = {**mm_la, **w_la}

    normalizer = LimbScoreNormalizer().fit(combined_limb_arrays)

    # ---- determine activities and score dimensions ----------------------
    if args.dataset == "wear":
        activities  = WEAR_ACTIVITIES
        n_activities = N_WEAR_ACTIVITIES
    elif args.dataset == "mmfit":
        activities  = MMFIT_ACTIVITIES
        n_activities = N_MMFIT_ACTIVITIES
    else:
        activities  = None   # both datasets use their own activity sets
        n_activities = N_TOTAL_ACTIVITIES

    # ---- attach normalizer and compute scores ---------------------------
    if args.dataset in ("wear", "both"):
        w_train.set_normalizer_and_activities(normalizer, WEAR_ACTIVITIES)
        w_val.set_normalizer_and_activities(normalizer, WEAR_ACTIVITIES)

    if args.dataset in ("mmfit", "both"):
        m_train.set_normalizer_and_activities(normalizer, MMFIT_ACTIVITIES)
        m_val.set_normalizer_and_activities(normalizer, MMFIT_ACTIVITIES)

    # ---- build DataLoaders ----------------------------------------------
    if args.dataset == "wear":
        train_loaders = DataLoader(w_train, batch_size=bsz, shuffle=True,
                                   num_workers=nw, pin_memory=pin, drop_last=True)
        val_loaders   = DataLoader(w_val, batch_size=bsz * 2, shuffle=False,
                                   num_workers=nw, pin_memory=pin)
    elif args.dataset == "mmfit":
        train_loaders = DataLoader(m_train, batch_size=bsz, shuffle=True,
                                   num_workers=nw, pin_memory=pin, drop_last=True)
        val_loaders   = DataLoader(m_val, batch_size=bsz * 2, shuffle=False,
                                   num_workers=nw, pin_memory=pin)
    else:
        # Both: separate loaders; model output is 30-dim, loss masked per loader
        train_loaders = [
            DataLoader(m_train, batch_size=bsz, shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=True),
            DataLoader(w_train, batch_size=bsz, shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=True),
        ]
        val_loaders = [
            DataLoader(m_val, batch_size=bsz * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin),
            DataLoader(w_val, batch_size=bsz * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin),
        ]

    return train_loaders, val_loaders, normalizer, n_activities


def _count_samples(loaders) -> int:
    if isinstance(loaders, (list, tuple)):
        return sum(len(l.dataset) for l in loaders)
    return len(loaders.dataset)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def _run_loader(model, loader, optimizer, device, dataset_tag, train: bool):
    """
    One pass over a single loader.

    dataset_tag: "wear" | "mmfit" | None
      Controls which output dims enter the loss in 'both' mode.
    """
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        pred = model(x)   # (B, n_activities)

        if dataset_tag == "mmfit":
            # 30-dim output: only first 11 matter
            loss = F.mse_loss(pred[:, :N_MMFIT_ACTIVITIES], y)
        elif dataset_tag == "wear":
            # 30-dim output: only last 19 matter
            loss = F.mse_loss(pred[:, N_MMFIT_ACTIVITIES:], y)
        else:
            # Single-dataset mode: all dims used
            loss = F.mse_loss(pred, y)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(x)
        n          += len(x)

    return total_loss, n


def train_epoch(model, loaders, optimizer, device, dataset):
    model.train()
    if dataset == "both":
        mm_loader, w_loader = loaders
        tl1, n1 = _run_loader(model, mm_loader, optimizer, device, "mmfit", True)
        tl2, n2 = _run_loader(model, w_loader,  optimizer, device, "wear",  True)
        return (tl1 + tl2) / (n1 + n2)
    else:
        tl, n = _run_loader(model, loaders, optimizer, device, None, True)
        return tl / n


@torch.no_grad()
def eval_epoch(model, loaders, device, dataset):
    model.eval()
    if dataset == "both":
        mm_loader, w_loader = loaders
        tl1, n1 = _run_loader(model, mm_loader, None, device, "mmfit", False)
        tl2, n2 = _run_loader(model, w_loader,  None, device, "wear",  False)
        return (tl1 + tl2) / (n1 + n2)
    else:
        tl, n = _run_loader(model, loaders, None, device, None, False)
        return tl / n


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}  |  Dataset: {args.dataset}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "pamap2":
        tag = f"scores_pamap2_fold{args.fold}" + ("_from_feat" if args.init_from else "")
    else:
        tag = f"scores_{args.dataset}" + ("_from_feat" if args.init_from else "")

    print("\nBuilding datasets...")
    train_loaders, val_loaders, normalizer, n_activities = build_datasets(args)
    print(f"  Train windows : {_count_samples(train_loaders):,}")
    print(f"  Val   windows : {_count_samples(val_loaders):,}")
    print(f"  Activity dims : {n_activities}")

    n_sensors = 3 if args.dataset == "pamap2" else N_SENSORS
    encoder   = None   # default Encoder1D
    embed_dim = args.embed_dim
    model = build_score_model(
        n_sensors    = n_sensors,
        n_activities = n_activities,
        embed_dim    = embed_dim,
        dropout      = args.dropout,
        encoder      = encoder,
    ).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # Accept checkpoints from train.py (model_state has encoder.* + head.*)
        # or from train_scores.py (also has encoder_state key)
        if "encoder_state" in ckpt:
            enc_state = ckpt["encoder_state"]
        else:
            enc_state = {
                k[len("encoder."):]: v
                for k, v in ckpt["model_state"].items()
                if k.startswith("encoder.")
            }
        model.encoder.load_state_dict(enc_state)
        print(f"\nEncoder initialised from: {args.init_from}")

    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total:,} total params  |  {train_:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    ckpt_path    = os.path.join(args.out_dir, f"{tag}_best.pt")
    norm_path    = os.path.join(args.out_dir, f"{tag}_normalizer.pkl")
    history_path = os.path.join(args.out_dir, f"{tag}_history.json")
    normalizer.save(norm_path)

    best_val = float("inf")
    history  = {"train_loss": [], "val_loss": []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loaders, optimizer, device, args.dataset)
        va = eval_epoch(model, val_loaders, device, args.dataset)
        scheduler.step()

        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={tr:.4f}  val={va:.4f} | "
              f"{time.time()-t0:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        if va < best_val:
            best_val = va
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "encoder_state": model.encoder.state_dict(),  # encoder-only extract
                "val_loss":     best_val,
                "n_activities": n_activities,
                "args":         vars(args),
            }, ckpt_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Normalizer  : {norm_path}")
    print(f"  History     : {history_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train score predictor")
    p.add_argument("--dataset",     default="wear",
                   choices=["wear", "mmfit", "pamap2", "both"])
    p.add_argument("--fold",        type=int, default=0,
                   help="LOSO fold index 0-8 (pamap2 only)")
    p.add_argument("--data_root",   default="data")
    p.add_argument("--out_dir",     default="runs")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--init_from",   default=None,
                   help="Path to a stage-1 checkpoint (train.py / train_wear.py) "
                        "whose encoder weights initialise the score predictor")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
