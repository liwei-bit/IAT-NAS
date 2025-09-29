# -*- coding: utf-8 -*-
"""
Train & evaluate NAS / baseline architectures on (imbalanced) medical or CIFAR datasets.
Outputs Accuracy plus常用医学指标：Macro‑F1 / Precision / Recall, Balanced‑Accuracy, AUC‑ROC.
保留原 `train_best_arch.py` 的类别权重公式，其他训练流程不变。
Last full sync: 2025‑07‑07
"""

import os
import json
import argparse
import random
from typing import List, Tuple

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

from medmnist_imbalance import get_medmnist_loader
from Dataset_partitioning import get_dataloader
from lib.models.nas201_model import build_model_from_arch_str

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def build_baseline(name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name in {"resnet18", "resnet34"}:
        net = getattr(tv_models, name)(num_classes=num_classes)
        net.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity(); return net
    if name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=num_classes)
        net.features[0][0] = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False); return net
    raise ValueError(f"Unknown baseline {name}")

# -----------------------------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------------------------

def _prepare_labels(y: torch.Tensor) -> torch.Tensor:
    return y.view(-1).to(y.device, non_blocking=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float, np.ndarray, float, float, float, float, float]:

    model.train() if train else model.eval()
    epoch_loss = total = correct = 0

    num_classes = criterion.weight.numel() if criterion.weight is not None else None
    cls_cor = np.zeros(num_classes) if num_classes else None
    cls_tot = np.zeros(num_classes) if num_classes else None

    y_true_all: List[int] = []; y_pred_all: List[int] = []; y_prob_all: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True);
        y = y.view(-1).long().to(device)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x); loss = criterion(out, y)
            if train: loss.backward(); optimizer.step()
        epoch_loss += loss.item() * y.size(0)
        probs = torch.softmax(out.detach(), dim=1); pred = probs.argmax(1)
        total += y.size(0); correct += pred.eq(y).sum().item()
        y_true_all.extend(y.tolist()); y_pred_all.extend(pred.tolist())
        if num_classes and probs.shape[1] == num_classes: y_prob_all.append(probs.cpu().numpy())
        if num_classes:
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cls_tot[t] += 1; cls_cor[t] += (t == p)

    acc = 100. * correct / max(total, 1)
    per_class = 100. * cls_cor / (cls_tot + 1e-9) if num_classes else np.array([])
    macro_f1 = 100. * f1_score(y_true_all, y_pred_all, average='macro')
    macro_prec = 100. * precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    macro_rec = 100. * recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    bal_acc = 100. * balanced_accuracy_score(y_true_all, y_pred_all)

    macro_auc = float('nan')
    if y_prob_all:
        try:
            macro_auc = 100. * roc_auc_score(y_true_all, np.concatenate(y_prob_all, 0), multi_class='ovr', average='macro')
        except ValueError:
            pass

    return epoch_loss / max(total,1), acc, per_class, macro_f1, macro_prec, macro_rec, bal_acc, macro_auc


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    inv = 1.0 / (counts + 1e-6); inv = np.clip(inv, 0.0, 50.0)
    inv = inv / inv.sum() * len(inv)
    return torch.tensor(inv, dtype=torch.float32)


def train_and_eval_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device):
    # class weights
    base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    idx = train_loader.dataset.indices if isinstance(train_loader.dataset, Subset) else None
    labels = np.array(getattr(base_ds, 'targets', getattr(base_ds, 'labels')))
    if idx is not None: labels = labels[idx]
    labels = labels.squeeze(); num_classes = int(labels.max()) + 1
    criterion = nn.CrossEntropyLoss(weight=_compute_class_weights(labels, num_classes).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = {'f1':0, 'acc':0, 'prec':0, 'rec':0, 'auc':0, 'pc':None}

    for ep in range(1, epochs+1):
        run_epoch(model, train_loader, criterion, opt, device, train=True)
        _, acc, pc, f1, prec, rec, _, auc = run_epoch(model, val_loader, criterion, opt, device, train=False)
        sch.step()
        if f1 > best['f1']:
            best.update({'f1':f1,'acc':acc,'prec':prec,'rec':rec,'auc':auc,'pc':pc})
        if ep % max(1, epochs//5) == 0 or ep == epochs:
            print(f"[Ep {ep:03d}/{epochs}] val_acc={acc:.2f} val_f1={f1:.2f} val_prec={prec:.2f} val_rec={rec:.2f} val_auc={auc:.2f}")
    return best

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    arch_cfg = json.load(open(args.arch_json))
    arch_list = list(arch_cfg.keys()) if isinstance(arch_cfg, dict) else arch_cfg

    # dataloaders
    if args.dataset.lower() in {"pathmnist","organamnist","dermamnist"}:
        train_loader, in_ch, n_cls = get_medmnist_loader(args.dataset.lower(), batch_size=args.batch_size, split='train')
        val_loader, _, _ = get_medmnist_loader(args.dataset.lower(), batch_size=args.batch_size, split='val')
    else:
        train_loader, in_ch, n_cls = get_dataloader(dataset=args.dataset, imbalance_type=args.imb_type, imbalance_ratio=args.imb_ratio, batch_size=args.batch_size)
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
        val_ds = getattr(datasets, args.dataset.upper())(root='./datasets', train=False, download=True, transform=tf_val)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed_list = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for arch in arch_list:
        print(f"\n==== Evaluate {arch} ====")
        f1s=[]; accs=[]; precs=[]; recs=[]; aucs=[]; pcls=[]
        for sd in seed_list:
            print(f"[Seed {sd}]"); set_seed(sd)
            model = build_model_from_arch_str(arch, in_channels=in_ch, num_classes=n_cls) if ('+' in arch or arch in {'none','skip_connect','conv1x1','conv3x3'}) else build_baseline(arch, in_channels=in_ch, num_classes=n_cls)
            model.to(device)
            best = train_and_eval_one(model, train_loader, val_loader, epochs=args.epochs, device=device)
            f1s.append(best['f1']); accs.append(best['acc']); precs.append(best['prec']); recs.append(best['rec']); aucs.append(best['auc']); pcls.append(best['pc'])
        summary = {
            'arch':arch,
            'mean_f1':float(np.mean(f1s)),'std_f1':float(np.std(f1s)),
            'mean_acc':float(np.mean(accs)),'std_acc':float(np.std(accs)),
            'mean_prec':float(np.mean(precs)),'std_prec':float(np.std(precs)),
            'mean_rec':float(np.mean(recs)),'std_rec':float(np.std(recs)),
            'mean_auc':float(np.nanmean(aucs)),'std_auc':float(np.nanstd(aucs)),
            'mean_per_class':np.nanmean(pcls,0).tolist(),'std_per_class':np.nanstd(pcls,0).tolist(),
        }
        all_results.append(summary)
        print(f"=> {arch}: F1 {summary['mean_f1']:.2f} ± {summary['std_f1']:.2f}% | Acc {summary['mean_acc']:.2f}% | AUC {summary['mean_auc']:.2f}%")

    all_results.sort(key=lambda d:d['mean_f1'], reverse=True)
    print("\n=== Ranking (by Macro‑F1) ===")
    for i, r in enumerate(all_results, 1):
        print(
            f"{i:02d}. {r['arch']:<40} "
            f"F1 {r['mean_f1']:.2f} ± {r['std_f1']:.2f}%  "
            f"(Acc {r['mean_acc']:.2f}%, AUC {r['mean_auc']:.2f}%)"
        )

    # -------- Save to JSON --------
    os.makedirs("results", exist_ok=True)
    outfile = f"results/{args.dataset}_top{len(all_results)}_{len(seed_list)}seed_metrics.json"
    with open(outfile, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved → {outfile}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_json", required=True)
    parser.add_argument(
        "--dataset",
        default="pathmnist",
        choices=["pathmnist", "organamnist", "dermamnist", "cifar10", "cifar100"],
    )
    parser.add_argument("--seeds", default="0")  # e.g. "0,1,2"
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--imb_ratio", type=float, default=0.05)
    args = parser.parse_args()
    main(args)



# python trian_F1.py --arch_json final_arch.json --dataset dermamnist --seeds 0,1,2 --epochs 10 --batch_size 128