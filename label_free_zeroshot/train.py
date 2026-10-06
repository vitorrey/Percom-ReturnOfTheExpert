"""
train.py
--------
Pre-training loop: predict pseudo-label feature profiles from raw
accelerometer windows (fully unsupervised — class labels not used).

Usage
-----
  python train.py --hold_out w03 --epochs 50 --batch_size 256

Leave-one-subject-out: the subject passed via --hold_out is kept entirely
out of training and used only for validation.
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
    SENSOR_NAMES, WINDOW_FRAMES, STRIDE_FRAMES, MMFIT_SPLITS,
)
from model import build_model, model_summary


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
# Train / eval loops
# -----------------------------------------------------------------------

def train_epoch(model:     nn.Module,
                loader:    DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device:    torch.device) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model:     nn.Module,
               loader:    DataLoader,
               criterion: nn.Module,
               device:    torch.device) -> float:
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred  = model(x)
        loss  = criterion(pred, y)
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- data ----------------------------------------------------------
    data_dir  = os.path.join(args.data_root, "mm-fit")
    cache_dir = os.path.join(args.data_root, "cache")

    print("\nBuilding datasets (official MM-Fit splits)...")
    print(f"  train={MMFIT_SPLITS['train']})")
    print(f"  val  ={MMFIT_SPLITS['val']}")

    train_ds = MMFitDataset(data_dir, MMFIT_SPLITS["train"],
                            WINDOW_FRAMES, STRIDE_FRAMES, SENSOR_NAMES,
                            normalizer=None, cache_dir=cache_dir)

    normalizer = ProfileNormalizer().fit(train_ds.raw_profiles())
    train_ds.normalizer = normalizer

    val_ds = MMFitDataset(data_dir, MMFIT_SPLITS["val"],
                          WINDOW_FRAMES, STRIDE_FRAMES, SENSOR_NAMES,
                          normalizer=normalizer, cache_dir=cache_dir)

    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val   windows: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type != "mps"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type != "mps"),
    )

    # ---- model ---------------------------------------------------------
    n_sensors  = len(SENSOR_NAMES)
    encoder    = None   # default Encoder1D
    embed_dim  = args.embed_dim
    model      = build_model(
        n_sensors      = n_sensors,
        n_feature_dims = 8,
        embed_dim      = embed_dim,
        dropout        = args.dropout,
        encoder        = encoder,
    ).to(device)

    print("\nModel:")
    model_summary(model, in_channels=n_sensors * 3,
                  window_frames=WINDOW_FRAMES)

    # ---- optimiser & scheduler ----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    # ---- output dir ----------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "mmfit"
    ckpt_path    = os.path.join(args.out_dir, f"{tag}_best.pt")
    norm_path    = os.path.join(args.out_dir, f"{tag}_normalizer.pkl")
    history_path = os.path.join(args.out_dir, f"{tag}_history.json")

    normalizer.save(norm_path)

    # ---- training loop -------------------------------------------------
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f} | "
              f"{elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val,
                "args":        vars(args),
            }, ckpt_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Normalizer : {norm_path}")
    print(f"  History    : {history_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train pseudo-label predictor")
    p.add_argument("--data_root",   default="data",   help="Root data directory")
    p.add_argument("--out_dir",     default="runs",   help="Output directory")
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
