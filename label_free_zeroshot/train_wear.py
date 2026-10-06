"""
train_wear.py
-------------
Pre-train the pseudo-label predictor on WEAR, MMFIT, or both datasets.

Usage (paper Table VI, WEAR "Feature regression"; saves wear_wear_best.pt)
-----
  python train_wear.py --dataset wear
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    MMFitDataset, ProfileNormalizer,
    SENSOR_NAMES as MMFIT_SENSOR_NAMES,
    WINDOW_FRAMES as MMFIT_WINDOW_FRAMES,
    STRIDE_FRAMES as MMFIT_STRIDE_FRAMES,
    MMFIT_SPLITS,
)
from dataset_wear import (
    WEARDataset, WEAR_SPLITS,
    WINDOW_FRAMES as WEAR_WINDOW_FRAMES,
    STRIDE_FRAMES as WEAR_STRIDE_FRAMES,
)
from model import build_model, model_summary

N_SENSORS  = 4   # same for both datasets
N_FEATURES = 8   # PROFILE_KEYS per sensor


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_loader(model, loader, criterion, optimizer, device, train: bool):
    """One pass over a single loader; returns (total_loss, n_samples)."""
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        loss = criterion(model(x), y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss, n


def train_epoch(model, loaders, criterion, optimizer, device):
    """Train on one or two loaders (interleaved for 'both' mode)."""
    model.train()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    total_loss, n = 0.0, 0
    for loader in loaders:
        tl, tn = _run_loader(model, loader, criterion, optimizer, device, train=True)
        total_loss += tl
        n += tn
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loaders, criterion, device):
    """Evaluate on one or two loaders."""
    model.eval()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    total_loss, n = 0.0, 0
    for loader in loaders:
        tl, tn = _run_loader(model, loader, criterion, None, device, train=False)
        total_loss += tl
        n += tn
    return total_loss / n


def build_datasets(args):
    """
    Return (train_loaders, val_loaders, normalizer).
    For 'both', each is a list of two loaders (MMFIT + WEAR).
    For single datasets, each is a single loader.
    The normalizer is always fitted only on training profiles.
    """
    mmfit_dir = os.path.join(args.data_root, "mm-fit")
    wear_dir  = args.data_root
    cache_dir = os.path.join(args.data_root, "cache")
    bsz       = args.batch_size
    nw        = args.num_workers
    pin       = (get_device().type != "mps")

    if args.dataset in ("mmfit", "both"):
        print("  MMFIT train subjects...")
        mm_train = MMFitDataset(mmfit_dir, MMFIT_SPLITS["train"],
                                MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES,
                                MMFIT_SENSOR_NAMES, normalizer=None,
                                cache_dir=cache_dir)
        mm_val   = MMFitDataset(mmfit_dir, MMFIT_SPLITS["val"],
                                MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES,
                                MMFIT_SENSOR_NAMES, normalizer=None,
                                cache_dir=cache_dir)

    if args.dataset in ("wear", "both"):
        print("  WEAR train subjects...")
        w_train = WEARDataset(wear_dir, WEAR_SPLITS["train"],
                              WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                              normalizer=None, cache_dir=cache_dir)
        w_val   = WEARDataset(wear_dir, WEAR_SPLITS["val"],
                              WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                              normalizer=None, cache_dir=cache_dir)

    # Fit normalizer on all training profiles
    if args.dataset == "mmfit":
        raw = mm_train.raw_profiles()
    elif args.dataset == "wear":
        raw = w_train.raw_profiles()
    else:
        raw = np.concatenate([mm_train.raw_profiles(), w_train.raw_profiles()])

    normalizer = ProfileNormalizer().fit(raw)

    # Attach normalizer
    if args.dataset == "mmfit":
        mm_train.normalizer = normalizer
        mm_val.normalizer   = normalizer
        train_loaders = DataLoader(mm_train, batch_size=bsz, shuffle=True,
                                   num_workers=nw, pin_memory=pin, drop_last=True)
        val_loaders   = DataLoader(mm_val, batch_size=bsz * 2, shuffle=False,
                                   num_workers=nw, pin_memory=pin)
    elif args.dataset == "wear":
        w_train.normalizer = normalizer
        w_val.normalizer   = normalizer
        train_loaders = DataLoader(w_train, batch_size=bsz, shuffle=True,
                                   num_workers=nw, pin_memory=pin, drop_last=True)
        val_loaders   = DataLoader(w_val, batch_size=bsz * 2, shuffle=False,
                                   num_workers=nw, pin_memory=pin)
    else:  # both — keep separate loaders (different window lengths)
        mm_train.normalizer = normalizer
        mm_val.normalizer   = normalizer
        w_train.normalizer  = normalizer
        w_val.normalizer    = normalizer
        train_loaders = [
            DataLoader(mm_train, batch_size=bsz, shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=True),
            DataLoader(w_train,  batch_size=bsz, shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=True),
        ]
        val_loaders = [
            DataLoader(mm_val, batch_size=bsz * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin),
            DataLoader(w_val,  batch_size=bsz * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin),
        ]

    return train_loaders, val_loaders, normalizer


def _count_samples(loaders):
    if isinstance(loaders, (list, tuple)):
        return sum(len(l.dataset) for l in loaders)
    return len(loaders.dataset)


def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}  |  Dataset: {args.dataset}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag        = f"wear_{args.dataset}"

    print("\nBuilding datasets...")
    train_loaders, val_loaders, normalizer = build_datasets(args)
    print(f"  Train windows: {_count_samples(train_loaders):,}")
    print(f"  Val   windows: {_count_samples(val_loaders):,}")

    encoder   = None   # default Encoder1D
    embed_dim = args.embed_dim
    model     = build_model(n_sensors=N_SENSORS, n_feature_dims=N_FEATURES,
                            embed_dim=embed_dim, dropout=args.dropout,
                            encoder=encoder).to(device)
    print("\nModel:")
    model_summary(model, in_channels=N_SENSORS * 3,
                  window_frames=WEAR_WINDOW_FRAMES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    ckpt_path    = os.path.join(args.out_dir, f"{tag}_best.pt")
    norm_path    = os.path.join(args.out_dir, f"{tag}_normalizer.pkl")
    history_path = os.path.join(args.out_dir, f"{tag}_history.json")
    normalizer.save(norm_path)

    best_val   = float("inf")
    history    = {"train_loss": [], "val_loss": []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loaders, criterion, optimizer, device)
        va = eval_epoch(model, val_loaders,   criterion, device)
        scheduler.step()
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={tr:.4f}  val={va:.4f} | "
              f"{time.time()-t0:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        if va < best_val:
            best_val = va
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": best_val, "args": vars(args)}, ckpt_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Normalizer : {norm_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="wear",
                   choices=["wear", "mmfit", "both"],
                   help="Dataset(s) to pre-train on")
    p.add_argument("--data_root",   default="data")
    p.add_argument("--out_dir",     default="runs")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
