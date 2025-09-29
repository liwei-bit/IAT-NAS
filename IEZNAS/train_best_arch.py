#!/usr/bin/env python
"""
train_best_arch.py – Train NAS-Bench-201 or baseline models on various datasets
---------------------------------------------------------------------------
usage examples
--------------

# CIFAR-10，单种子
python train_best_arch.py \
    --arch_json results/evo_topk_cifar10_top3.json \
    --dataset cifar10 --epochs 100

# PathMNIST，5 个随机种子
python train_best_arch.py \
    --arch_json results/evo_topk_pathmnist_top5.json \
    --dataset pathmnist --seeds 0,1,2,3,4 \
    --epochs 100 --batch_size 64
"""

import os, json, argparse, random, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models as tv_models
from tqdm import tqdm

# ---------- 自己的模块 ----------
from medmnist_imbalance import get_medmnist_loader
from Dataset_partitioning import get_dataloader          # CIFAR 专用
from lib.models.nas201_model import build_model_from_arch_str

# ---------- 复现可控性 ----------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- baseline 构造 ----------
def build_baseline(name: str, in_channels: int, num_classes: int):
    """
    构建未经修改的 ResNet18 / 34 / 50 / MobileNetV2 baseline
    用于公平对比标准架构，不做结构适配
    """
    name = name.lower()
    if name in {"resnet18", "resnet34", "resnet50"}:
        net = getattr(tv_models, name)(num_classes=num_classes)
        # ⚠️ 不修改 conv1，不去掉 maxpool，保持标准结构
        if in_channels != 3:
            # 为了支持灰度图（1通道），补充适配输入通道（但保持其他结构不变）
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return net
    elif name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=num_classes)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return net
    elif name == "densenet121":
        net = tv_models.densenet121(num_classes=num_classes)
        if in_channels != 3:
            net.features.conv0 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return net
    elif name.startswith("tinynet"):
        import timm
        net = timm.create_model(name, pretrained=False, num_classes=num_classes, in_chans=in_channels)
        return net
    else:
        raise ValueError(f"Unknown baseline {name}")


# ---------- 单 epoch 训练 / 验证 ----------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    epoch_loss, total, correct = 0., 0, 0
    num_classes = criterion.weight.numel()
    cls_cor = np.zeros(num_classes); cls_tot = np.zeros(num_classes)

    for x, y in loader:
        x, y = x.to(device), y.squeeze(1).to(device)     # squeeze for MedMNIST
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward(); optimizer.step()
        epoch_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        total += y.size(0); correct += pred.eq(y).sum().item()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cls_tot[t] += 1
            if t == p: cls_cor[t] += 1

    acc = 100. * correct / total
    per_class = 100. * cls_cor / (cls_tot + 1e-9)
    return epoch_loss / total, acc, per_class

# ---------- 主流程 ----------
def train_and_eval_one(model, train_loader, val_loader, epochs, device):
    # 类别加权：1/出现次数 → clip → 归一化
    if isinstance(train_loader.dataset, Subset):
        base_dataset = train_loader.dataset.dataset
        indices = train_loader.dataset.indices
    else:
        base_dataset = train_loader.dataset
        indices = None

    # 兼容 .targets / .labels
    if hasattr(base_dataset, 'targets'):
        labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        labels = np.array(base_dataset.labels)
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'labels'")

    if indices is not None:
        labels = labels[indices]

    labels = labels.squeeze()

    counts = np.bincount(labels, minlength=train_loader.dataset.dataset.info['n_classes']
                         if hasattr(train_loader.dataset, 'dataset') else labels.max()+1).astype(float)
    class_weights = 1. / (counts + 1e-6)
    class_weights = np.clip(class_weights, 0, 50)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_pc = 0., None
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_pc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        if val_acc > best_acc:
            best_acc, best_pc = val_acc, val_pc
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"[{ep:03d}/{epochs}] tr_acc={tr_acc:.2f} val_acc={val_acc:.2f}")
    return best_acc, best_pc

# ---------- 入口 ----------
def main(args):
    # -------- 读取架构列表 --------
    with open(args.arch_json) as f:
        arch_dict = json.load(f)
    arch_list = list(arch_dict.keys()) if isinstance(arch_dict, dict) else arch_dict

    # -------- 数据加载 --------
    if args.dataset in {'pathmnist','organamnist','dermamnist'}:
        train_loader, in_ch, n_cls = get_medmnist_loader(args.dataset,
                                                         batch_size=args.batch_size,
                                                         split='train')
        val_loader, _, _ = get_medmnist_loader(args.dataset,
                                               batch_size=args.batch_size,
                                               split='test')
    else:  # CIFAR
        train_loader, in_ch, n_cls = get_dataloader(dataset=args.dataset,
                                                    imbalance_type=args.imb_type,
                                                    imbalance_ratio=args.imb_ratio,
                                                    batch_size=args.batch_size)
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010)),
        ])
        val_ds = getattr(datasets, args.dataset.upper())(
            root='./datasets', train=False, download=True, transform=tf_val)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # -------- 多随机种子 --------
    seed_list = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

    for arch_str in arch_list:
        accs, percs = [], []
        print(f"\n==== Evaluate {arch_str} ====")
        for sd in seed_list:
            print(f"[Seed {sd}]"); set_seed(sd)
            # 判定是 NAS 架构还是 baseline 名称
            if '+' in arch_str or arch_str in {'none','skip_connect','conv1x1','conv3x3'}:
                model = build_model_from_arch_str(arch_str, in_channels=in_ch, num_classes=n_cls)
            else:
                model = build_baseline(arch_str, in_channels=in_ch, num_classes=n_cls)
            model.to(device)

            best_acc, best_pc = train_and_eval_one(model, train_loader, val_loader,
                                                   epochs=args.epochs, device=device)
            accs.append(best_acc); percs.append(best_pc)

        accs = np.array(accs); percs = np.array(percs)
        entry = {
            "arch": arch_str,
            "acc_list": accs.tolist(),
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_per_class": percs.mean(0).tolist(),
            "std_per_class": percs.std(0).tolist()
        }
        all_results.append(entry)
        print(f"=> {arch_str}: {entry['mean_acc']:.2f} ± {entry['std_acc']:.2f}%")

    # -------- 排序 & 保存 --------
    all_results.sort(key=lambda d: d['mean_acc'], reverse=True)
    print("\n=== Ranking ===")
    for i,r in enumerate(all_results,1):
        print(f"{i:02d}. {r['arch']:<50} {r['mean_acc']:.2f} ± {r['std_acc']:.2f}%")

    os.makedirs("results", exist_ok=True)
    fname = f"results/{args.dataset}_top{len(all_results)}_{len(seed_list)}seed_final_manual.json"
    with open(fname,'w') as f: json.dump(all_results,f,indent=2)
    print(f"\nSaved → {fname}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch_json', required=True, help='JSON with arch strings / baseline names')
    ap.add_argument('--dataset', default='pathmnist',
                    choices=['pathmnist','organamnist','dermamnist','cifar10','cifar100'])
    ap.add_argument('--seeds', default='0', help='e.g. "0" or "0,1,2"')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--imb_type', default='exp')
    ap.add_argument('--imb_ratio', type=float, default=0.05)
    args = ap.parse_args()
    main(args)

   #   python train_best_arch.py --arch_json manual_arch.json
