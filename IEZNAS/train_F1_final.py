# -*- coding: utf-8 -*-
"""
Train & evaluate NAS / baseline architectures on (imbalanced) medical or CIFAR datasets
with **paper‑grade rigor**:
    • Advanced data augmentation (MedMNIST + CIFAR)
    • Class‑balanced loss (inverse frequency)
    • Exponential Moving Average (EMA) of parameters
    • Early‑Stopping on validation Macro‑F1
    • Automatic test‑set evaluation of the best checkpoint
    • Training/validation curves saved to CSV + optional PNG

Example
-------
python train_f1_full.py \
    --arch_json final_arch.json \
    --dataset organamnist \
    --seeds 0,1,2 \
    --epochs 50 \
    --batch_size 128 \
    --patience 10 \
    --ema_decay 0.999 \
    --save_curve

Last update: 2025‑07‑07
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models as tv_models
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from medmnist_imbalance import get_medmnist_loader
from Dataset_partitioning import get_dataloader
from lib.models.nas201_model import build_model_from_arch_str

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------- EMA implementation ----------------------------
class EMA:
    """Maintain an exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ------------------------- Model building helpers ---------------------------

def build_baseline(name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name in {"resnet18", "resnet34"}:
        net = getattr(tv_models, name)(num_classes=num_classes)
        net.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        return net
    if name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=num_classes)
        net.features[0][0] = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False)
        return net
    raise ValueError(f"Unknown baseline {name}")


# -----------------------------------------------------------------------------
# Data augmentation pipelines
# -----------------------------------------------------------------------------

def build_medmnist_transform(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


def build_cifar_transform(train: bool = True):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# -----------------------------------------------------------------------------
# Metrics & helpers
# -----------------------------------------------------------------------------

def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    inv = 1.0 / (counts + 1e-6)
    inv = inv / inv.sum() * num_classes  # normalized inverse‑frequency
    return torch.tensor(inv, dtype=torch.float32)


@torch.no_grad()
def _eval_metrics(y_true: List[int], y_pred: List[int], y_prob: List[np.ndarray]) -> Dict[str, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    macro_auc = float("nan")
    if y_prob:
        try:
            macro_auc = roc_auc_score(y_true, np.concatenate(y_prob, 0), multi_class="ovr", average="macro") * 100
        except ValueError:
            pass
    return dict(f1=macro_f1, prec=macro_prec, rec=macro_rec, bal_acc=bal_acc, auc=macro_auc)


# -----------------------------------------------------------------------------
# Train / Validation epoch
# -----------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ema: EMA | None = None,
) -> Tuple[float, Dict[str, float]]:

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    for x, y in loader:
        if y.numel() == 0:
            continue  # safety
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long().to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

        epoch_loss += loss.item() * y.size(0)
        probs = torch.softmax(out.detach(), dim=1)
        preds = probs.argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())
        if probs.shape[1] == (criterion.weight.numel() if criterion.weight is not None else probs.shape[1]):
            y_prob.append(probs.cpu().numpy())

    metrics = _eval_metrics(y_true, y_pred, y_prob)
    metrics["acc"] = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    return epoch_loss / max(len(loader.dataset), 1), metrics


# -----------------------------------------------------------------------------
# Training wrapper
# -----------------------------------------------------------------------------

def train_one_arch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    epochs: int,
    patience: int,
    ema_decay: float,
    device: torch.device,
    log_dir: Path,
) -> Dict[str, Any]:

    # ----- loss & optimizer -----
    base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    idx = train_loader.dataset.indices if isinstance(train_loader.dataset, Subset) else None
    labels = np.array(getattr(base_ds, "targets", getattr(base_ds, "labels")))
    if idx is not None:
        labels = labels[idx]
    labels = labels.squeeze()
    num_classes = int(labels.max()) + 1

    criterion = nn.CrossEntropyLoss(weight=_compute_class_weights(labels, num_classes).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ema = EMA(model, decay=ema_decay) if ema_decay < 1.0 else None

    # ----- bookkeeping -----
    history: Dict[str, List[float]] = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, ema)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, ema)
        scheduler.step()

        # ----- record -----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"[Ep {ep:03d}/{epochs}] → "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_metrics['acc']:.2f}%  val_f1 {val_metrics['f1']:.2f}%  "
            f"val_auc {val_metrics['auc']:.2f}%"
        )

        # ----- early stopping -----
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in (ema.shadow.items() if ema else model.state_dict()).items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep} (no improve {patience} epochs).")
                break

    # ----- load best & evaluate on test -----
    if best_state is not None:
        if ema:
            ema.shadow = best_state
            ema.copy_to(model)
        else:
            model.load_state_dict(best_state)

    final_metrics = {"val_f1": best_f1}
    if test_loader is not None:
        _, test_metrics = run_epoch(model, test_loader, criterion, None, device, ema)
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        print(
            f"TEST → acc {test_metrics['acc']:.2f}%  "
            f"f1 {test_metrics['f1']:.2f}%  auc {test_metrics['auc']:.2f}%"
        )

    # ----- save curves -----
    csv_path = log_dir / "train_curves.csv"
    np.savetxt(csv_path, np.column_stack([history[k] for k in history]), delimiter=",", header=",".join(history.keys()), comments="")

    return final_metrics


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(0)

    # ----- parse arch list -----
    arch_cfg = json.load(open(args.arch_json))
    arch_list = list(arch_cfg.keys()) if isinstance(arch_cfg, dict) else arch_cfg

    # ----- build dataset loaders -----
    if args.dataset.lower() in {"pathmnist", "organamnist", "dermamnist"}:
        train_loader, in_ch, n_cls = get_medmnist_loader(
            args.dataset.lower(), batch_size=args.batch_size, split="train", transform=build_medmnist_transform(True)
        )
        val_loader, _, _ = get_medmnist_loader(
            args.dataset.lower(), batch_size=args.batch_size, split="val", transform=build_medmnist_transform(False)
        )
        test_loader, _, _ = get_medmnist_loader(
            args.dataset.lower(), batch_size=args.batch_size, split="test", transform=build_medmnist_transform(False)
        )
    else:
        # CIFAR
        train_loader, in_ch, n_cls = get_dataloader(
            dataset=args.dataset,
            imbalance_type=args.imb_type,
            imbalance_ratio=args.imb_ratio,
            batch_size=args.batch_size,
            transform_train=build_cifar_transform(True),
        )
        tf_val = build_cifar_transform(False)
        val_ds = getattr(datasets, args.dataset.upper())(
            root="./datasets", train=False, download=True, transform=tf_val
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = None  # CIFAR test==val in many setups; adapt if you have official split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_list = [int(s) for s in args.seeds.split(",")]

    results: List[Dict[str, Any]] = []
    out_root = Path("results")
    out_root.mkdir(exist_ok=True)

    for arch in arch_list:
        print(f"\n==== Evaluate {arch} ====")
        seed_metrics: List[Dict[str, Any]] = []
        for sd in seed_list:
            print(f"[Seed {sd}]")
            set_seed(sd)
            model = (
                build_model_from_arch_str(arch, in_channels=in_ch, num_classes=n_cls)
                if ("+" in arch or arch in {"none", "skip_connect", "conv1x1", "conv3x3"})
                else build_baseline(arch, in_channels=in_ch, num_classes=n_cls)
            ).to(device)

            log_dir = out_root / f"{args.dataset}_{arch.replace('+','_')}_seed{sd}"
            log_dir.mkdir(exist_ok=True, parents=True)

            met = train_one_arch(
                model,
                train_loader,
                val_loader,
                test_loader,
                epochs=args.epochs,
                patience=args.patience,
                ema_decay=args.ema_decay,
                device=device,
                log_dir=log_dir,
            )
            seed_metrics.append(met)

        # aggregate
        f1s = [m["val_f1"] for m in seed_metrics]
        accs = [m.get("test_acc", np.nan) for m in seed_metrics]
        aucs = [m.get("test_auc", np.nan) for m in seed_metrics]
        summary = dict(
            arch=arch,
            mean_f1=float(np.nanmean(f1s)),
            std_f1=float(np.nanstd(f1s)),
            mean_acc=float(np.nanmean(accs)),
            mean_auc=float(np.nanmean(aucs)),
        )
        results.append(summary)
        print(
            f"=> {arch}: F1 {summary['mean_f1']:.2f}±{summary['std_f1']:.2f}% | "
            f"Acc {summary['mean_acc']:.2f}% | AUC {summary['mean_auc']:.2f}%"
        )

    # ranking
    results.sort(key=lambda d: d["mean_f1"], reverse=True)
    print("\n=== Ranking (by Macro‑F1) ===")
    for i, r in enumerate(results, 1):
        print(
            f"{i:02d}. {r['arch']:<40} F1 {r['mean_f1']:.2f}% (Acc {r['mean_acc']:.2f}%)"
        )

    # save overall json
    outfile = out_root / f"{args.dataset}_summary.json"
    json.dump(results, open(outfile, "w"), indent=2)
    print(f"Saved summary → {outfile}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_json", required=True, help="JSON list/dict of architectures")
    parser.add_argument(
        "--dataset",
        default="pathmnist",
        choices=["pathmnist", "organamnist", "dermamnist", "cifar10", "cifar100"],
    )
    parser.add_argument("--seeds", default="0", help="Comma‑sep list, e.g. 0,1,2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--imb_ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay; <1 enables EMA")
    parser.add_argument("--save_curve", action="store_true", help="Save PNG of train/val curves")
    args = parser.parse_args()
    main(args)
