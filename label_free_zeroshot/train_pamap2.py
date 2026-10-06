"""
train_pamap2.py
---------------
Pre-train the pseudo-label predictor on PAMAP2 for one LOSO fold.

Pre-training is fully unsupervised (no class labels used).
For fold i: train on the 7 non-test/non-val subjects, validate on the val subject.

Usage
-----
  python train_pamap2.py --fold 0 --seed 0 --out_dir runs/seed_0
  python train_pamap2.py --fold 3 --seed 2 --out_dir runs/seed_2
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ProfileNormalizer
from dataset_pamap2 import (
    PAMAP2Dataset, pamap2_loso_splits,
    WINDOW_FRAMES, STRIDE_FRAMES,
)
from model import build_model, model_summary

N_SENSORS  = 3   # hand, chest, ankle
N_FEATURES = 8   # PROFILE_KEYS per sensor


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(x)
            n          += len(x)
    return total_loss / n


def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    cache_dir = os.path.join(args.data_root, "cache")

    fold            = pamap2_loso_splits()[args.fold]
    # Pre-training uses only the 7 train subjects (test and eval-val are held out).
    # We carve off the last of those 7 as pre-training validation.
    pretrain_subjects = fold["train"]          # 7 subjects
    train_subjects    = pretrain_subjects[:-1] # 6 for pre-training
    val_subjects      = pretrain_subjects[-1:] # 1 for pre-training val

    print(f"\nFold {args.fold}: test={fold['test'][0]}")
    print(f"  pretrain train subjects: {train_subjects}")
    print(f"  pretrain val  subjects : {val_subjects}")

    train_ds = PAMAP2Dataset(args.data_root, train_subjects,
                             WINDOW_FRAMES, STRIDE_FRAMES,
                             normalizer=None, cache_dir=cache_dir)

    normalizer = ProfileNormalizer().fit(train_ds.raw_profiles())
    train_ds.normalizer = normalizer

    val_ds = PAMAP2Dataset(args.data_root, val_subjects,
                           WINDOW_FRAMES, STRIDE_FRAMES,
                           normalizer=normalizer, cache_dir=cache_dir)

    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val   windows: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type != "mps"), drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type != "mps"))

    encoder    = None   # default Encoder1D
    embed_dim  = args.embed_dim

    model = build_model(n_sensors=N_SENSORS, n_feature_dims=N_FEATURES,
                        embed_dim=embed_dim, dropout=args.dropout,
                        encoder=encoder).to(device)
    print("\nModel:")
    model_summary(model, in_channels=N_SENSORS * 3, window_frames=WINDOW_FRAMES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    tag          = f"pamap2_fold{args.fold}"
    ckpt_path    = os.path.join(args.out_dir, f"{tag}_best.pt")
    norm_path    = os.path.join(args.out_dir, f"{tag}_normalizer.pkl")
    history_path = os.path.join(args.out_dir, f"{tag}_history.json")
    normalizer.save(norm_path)

    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = _run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f} | "
              f"{time.time()-t0:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": best_val, "args": vars(args)}, ckpt_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Normalizer : {norm_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train pseudo-label predictor on PAMAP2")
    p.add_argument("--fold",        type=int,   required=True,
                   help="LOSO fold index (0–8)")
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
