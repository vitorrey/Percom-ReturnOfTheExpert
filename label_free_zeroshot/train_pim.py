"""
train_pim.py
------------
PIM pseudo-label pre-training on WEAR, MMFIT, or PAMAP2.

Follows the PIM training setup described in the PIM paper:
  - SGD optimiser with MultiStepLR scheduler
  - 6× data augmentation (window + 3 transforms, each paired with the window)
  - Discretised pseudo-labels (onehot-dense, 11 uniform bins)
  - BCE loss for angle heads, CrossEntropy for motion/symmetry heads

Usage
-----
  python train_pim.py --dataset wear
  python train_pim.py --dataset mmfit
  python train_pim.py --dataset pamap2 --fold 0
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
    MMFitDataset, MMFIT_SPLITS,
    SENSOR_NAMES as MMFIT_SENSOR_NAMES,
    WINDOW_FRAMES as MMFIT_WINDOW_FRAMES,
    STRIDE_FRAMES as MMFIT_STRIDE_FRAMES,
)
from dataset_wear import (
    WEARDataset, WEAR_SPLITS,
    WINDOW_FRAMES as WEAR_WINDOW_FRAMES,
    STRIDE_FRAMES as WEAR_STRIDE_FRAMES,
)
from dataset_pamap2 import (
    PAMAP2Dataset, pamap2_loso_splits,
    WINDOW_FRAMES as PAMAP2_WINDOW_FRAMES,
    STRIDE_FRAMES as PAMAP2_STRIDE_FRAMES,
)
from features_pim import PIMDiscretizer
from dataset_pim import PIMDataset
from model_pim import build_pim

N_SENSORS = {"wear": 4, "mmfit": 4, "pamap2": 3}
FS        = {"wear": 50.0, "mmfit": 30.0, "pamap2": 100.0 / 3.0}


def get_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------
# Raw window loading (no labels needed for pre-training)
# -----------------------------------------------------------------------

class _WindowOnly(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i): x, _ = self.base[i]; return x


def load_raw_windows(dataset: str, data_root: str, subjects, cache_dir: str):
    """Returns (N, W, C) numpy array of raw windows."""
    if dataset == "wear":
        ds = WEARDataset(data_root, subjects, WEAR_WINDOW_FRAMES, WEAR_STRIDE_FRAMES,
                         normalizer=None, cache_dir=cache_dir)
    elif dataset == "mmfit":
        mmfit_dir = os.path.join(data_root, "mm-fit")
        ds = MMFitDataset(mmfit_dir, subjects, MMFIT_WINDOW_FRAMES, MMFIT_STRIDE_FRAMES,
                          MMFIT_SENSOR_NAMES, normalizer=None, cache_dir=cache_dir)
    else:
        raise ValueError(dataset)

    windows = []
    for i in range(len(ds)):
        x, _ = ds[i]
        windows.append(x.numpy().T)   # (W, C)
    return np.stack(windows)


def load_pamap2_windows(data_root: str, subjects, fold: int):
    """Returns (N, W, C) numpy array from PAMAP2Dataset."""
    ds = PAMAP2Dataset(data_root, subjects, PAMAP2_WINDOW_FRAMES, PAMAP2_STRIDE_FRAMES)
    windows = []
    for i in range(len(ds)):
        x, _ = ds[i]
        windows.append(x.numpy().T)
    return np.stack(windows)


# -----------------------------------------------------------------------
# Collate: stack windows and list-of-ps-tensors
# -----------------------------------------------------------------------

def _collate(batch):
    xs  = torch.stack([b[0] for b in batch])      # (B, C, W)
    n   = len(batch[0][1])
    ps  = [torch.stack([b[1][i] for b in batch]) for i in range(n)]
    return xs, ps


# -----------------------------------------------------------------------
# One epoch
# -----------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    total, n = 0.0, 0
    with ctx:
        for x, ps in loader:
            x  = x.to(device)
            ps = [y.to(device) for y in ps]
            loss = model.get_losses(x, ps)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n     += len(x)
    return total / n


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    fs     = FS[args.dataset]
    n_sen  = N_SENSORS[args.dataset]
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  fs={fs}Hz")

    os.makedirs(args.out_dir, exist_ok=True)
    variant = "pim_noaug" if args.no_augment else "pim"
    tag = (f"{variant}_pamap2_fold{args.fold}" if args.dataset == "pamap2"
           else f"{variant}_{args.dataset}")
    ckpt_path    = os.path.join(args.out_dir, f"{tag}_best.pt")
    disc_path    = os.path.join(args.out_dir, f"{tag}_discretizer.pkl")
    history_path = os.path.join(args.out_dir, f"{tag}_history.json")

    if os.path.exists(ckpt_path):
        print(f"[SKIP] already exists: {ckpt_path}")
        return

    # ---- load windows -----------------------------------------------
    cache_dir = os.path.join(args.data_root, "cache")
    print("\nLoading windows...")
    if args.dataset == "pamap2":
        folds   = pamap2_loso_splits()
        fold    = folds[args.fold]
        train_s = fold["train"][:-1]
        val_s   = fold["train"][-1:]
        print(f"  PAMAP2 fold {args.fold}: train={train_s}  val={val_s}")
        train_w = load_pamap2_windows(args.data_root, train_s, args.fold)
        val_w   = load_pamap2_windows(args.data_root, val_s,   args.fold)
    else:
        splits  = {"wear": WEAR_SPLITS, "mmfit": MMFIT_SPLITS}[args.dataset]
        train_w = load_raw_windows(args.dataset, args.data_root, splits["train"], cache_dir)
        val_w   = load_raw_windows(args.dataset, args.data_root, splits["val"],   cache_dir)

    print(f"  train={len(train_w):,}  val={len(val_w):,}  shape={train_w.shape[1:]}")

    # ---- fit discretizer on training windows ------------------------
    print("\nFitting discretizer...")
    disc = PIMDiscretizer(fs=fs, n_bins=args.n_bins).fit(train_w, args.dataset)
    disc.save(disc_path)

    # ---- build datasets & loaders -----------------------------------
    pin = (device.type != "mps")
    tr_ds = PIMDataset(train_w, args.dataset, disc, augment=not args.no_augment)
    va_ds = PIMDataset(val_w,   args.dataset, disc, augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=pin,
                           drop_last=True, collate_fn=_collate)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin,
                           collate_fn=_collate)
    print(f"  Dataset 6× expanded: {len(tr_ds):,} train  {len(va_ds):,} val")

    # ---- model -------------------------------------------------------
    model = build_pim(args.dataset, n_sen, args.embed_dim, args.n_bins).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total:,} params")

    # SGD with MultiStepLR, as in PIM
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75, 125, 150, 160], gamma=0.1)

    best_val = float("inf")
    history  = {"train_loss": [], "val_loss": []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0  = time.time()
        tr  = _run_epoch(model, tr_loader, optimizer, device, train=True)
        va  = _run_epoch(model, va_loader, optimizer, device, train=False)
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
    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Discretizer : {disc_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="wear",
                   choices=["wear", "mmfit", "pamap2"])
    p.add_argument("--fold",         type=int, default=0)
    p.add_argument("--data_root",    default="data")
    p.add_argument("--out_dir",      default="runs")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--embed_dim",    type=int,   default=256)
    p.add_argument("--n_bins",       type=int,   default=11)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_augment",   action="store_true",
                   help="Disable data augmentation (produces pim_noaug_* checkpoints)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
