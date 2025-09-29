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
import timm
import torch.nn.functional as F

# ---------- 自己的模块 ----------
from medmnist_get_imbalance import get_medmnist_dataloaders
from Dataset_partitioning import get_dataloader     # CIFAR 专用
from lib.dataop.ISIC_2019 import get_isic2019_loader
from lib.models.nas201_model import build_model_from_arch_str
import hashlib
from medmnist import INFO  # ← 新增：用来判断是否多标签
from sklearn.metrics import roc_auc_score

def _safe_auc_binary(y_true, y_pos_prob):
    """二分类安全 AUC：y_true ∈ {0,1}，y_pos_prob 为正类概率 (N,)"""
    y_true = np.asarray(y_true)
    y_pos_prob = np.asarray(y_pos_prob)
    # 需要同时有0和1两个类别
    if np.unique(y_true).size < 2:
        return None
    try:
        return roc_auc_score(y_true, y_pos_prob)
    except Exception:
        return None

def _safe_auc_multiclass(y_true, y_prob_2d):
    """多类安全 AUC：y_prob_2d 形状 (N, C)，ovr 宏平均"""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    try:
        return roc_auc_score(y_true, y_prob_2d, multi_class='ovr', average='macro')
    except Exception:
        return None



def short_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]

# ---------- 复现可控性 ----------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class InputPadWrapper(nn.Module):
    def __init__(self, model, min_size=32):
        super().__init__()
        self.model = model
        self.min_size = min_size

    def forward(self, x):
        h, w = x.shape[-2:]
        if h < self.min_size or w < self.min_size:
            pad_h = self.min_size - h
            pad_w = self.min_size - w
            # (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return self.model(x)

# ---------- baseline 构造 ----------
def build_baseline(name: str, in_channels: int, num_classes: int):
    """
    构建未经修改的 ResNet18 / 34 / 50 / MobileNetV2 baseline
    用于公平对比标准架构，不做结构适配
    """
    name = name.lower()
    if name in {"resnet18", "resnet34", "resnet50"}:
        net = getattr(tv_models, name)(num_classes=num_classes)
        # ⚠️ 保持原始结构不变，仅适配灰度图输入
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return InputPadWrapper(net, min_size=64)  # ✅ 自动 pad 到 64×64，避免 BN 报错

    elif name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=num_classes)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return net
    elif name == "densenet121":
        net = tv_models.densenet121(num_classes=num_classes)
        # 为适配灰度图输入
        net.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 替换大卷积和最大池化层，适配小图（如28x28）
        net.features.pool0 = nn.Identity()
        return net
    elif name == "convnext_tiny":
        net = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes, in_chans=in_channels)
        net = InputPadWrapper(net, min_size=32)  # 自动 pad 输入到 32×32
        return net
    elif name in timm.list_models():
        net = timm.create_model(name, pretrained=False, num_classes=num_classes, in_chans=in_channels)
        net = InputPadWrapper(net, min_size=64)
        return net
    else:
        raise ValueError(f"Unknown baseline {name}")

def get_model_output_classes(model):
    """
    自动获取模型输出类别数，兼容 resnet、mobilenet_v2、densenet、convnext、timm 模型等
    """
    # case 1: ResNet (torchvision)

    if isinstance(model, InputPadWrapper):
        model = model.model

    if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
        return model.fc.out_features

    # case 2: DenseNet / MobileNetV2 等 classifier 是 Sequential
    if hasattr(model, 'classifier'):
        clf = model.classifier
        if hasattr(clf, 'out_features'):
            return clf.out_features
        elif isinstance(clf, nn.Sequential):
            for layer in reversed(clf):
                if hasattr(layer, 'out_features'):
                    return layer.out_features

    # case 3: timm 模型统一接口
    if hasattr(model, 'get_classifier') and callable(model.get_classifier):
        try:
            layer = model.get_classifier()
            if hasattr(layer, 'out_features'):
                return layer.out_features
        except:
            pass

    raise ValueError("⚠️ Cannot infer number of output classes from model structure.")



# ---------- 单 epoch 训练 / 验证 ----------
def run_epoch(model, loader, criterion, optimizer, device, train=True, num_classes=None, is_multilabel=False):
    import torch.nn.functional as F

    model.train() if train else model.eval()

    # 推断类别数
    if num_classes is None:
        if criterion is not None and getattr(criterion, 'weight', None) is not None:
            num_classes = criterion.weight.numel()
        else:
            num_classes = get_model_output_classes(model)

    epoch_loss = 0.0
    total_samples = 0
    correct_total = 0.0

    # 按类/按标签
    cls_cor = np.zeros(num_classes, dtype=np.float64)
    cls_tot = 0

    # AUC 收集容器
    all_probs = []     # 二分类: 正类概率 (N,); 多类: (N, C); 多标签: (N, C)
    all_targets = []   # (N,) 或 (N, C)

    for x, y in loader:
        x = x.to(device)

        if is_multilabel:
            y = y.float().to(device)             # [B, C]
        else:
            if y.ndim > 1:
                y = y.squeeze(1)
            y = y.long().to(device)              # [B]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        epoch_loss += float(loss.item()) * bs
        total_samples += bs

        if is_multilabel:
            probs = torch.sigmoid(logits)        # [B, C]
            preds = (probs > 0.5).float()
            correct_total += preds.eq(y).float().mean().item() * bs

            cls_cor += preds.eq(y).float().sum(dim=0).detach().cpu().numpy()
            cls_tot += bs

            if not train:
                all_probs.append(probs.detach().cpu().numpy())   # (B, C)
                all_targets.append(y.detach().cpu().numpy())     # (B, C)
        else:
            # 单标签
            pred = logits.argmax(1)
            correct_total += pred.eq(y).float().sum().item()

            # 粗略 per-class（与你原逻辑保持一致）
            for t, p in zip(y.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                cls_cor[int(t)] += (int(t) == int(p))
            cls_tot += 1

            if not train:
                sm = F.softmax(logits, dim=1).detach().cpu().numpy()  # (B, num_classes)
                if num_classes == 2:
                    pos_prob = sm[:, 1]                                # 正类概率 (B,)
                    all_probs.append(pos_prob)                         # list of (B,)
                else:
                    all_probs.append(sm)                               # list of (B, C)
                all_targets.append(y.detach().cpu().numpy())          # list of (B,)

    # ---- 汇总指标 ----
    acc = 100.0 * (correct_total / max(1, total_samples))

    if is_multilabel:
        per_class = 100.0 * (cls_cor / (cls_tot + 1e-12))
        auc = None
        if len(all_probs) > 0:
            P = np.concatenate(all_probs, axis=0)   # (N, C)
            T = np.concatenate(all_targets, axis=0) # (N, C)
            # 安全 AUC（多标签宏平均）
            try:
                # 需要每个标签既有正也有负，否则 roc_auc 会抛错
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(T, P, average='macro')
            except Exception:
                auc = None
    else:
        per_class = 100.0 * cls_cor / (np.maximum(1.0, cls_tot))  # 与你原逻辑保持一致
        auc = None
        if len(all_probs) > 0:
            T = np.concatenate(all_targets, axis=0)  # (N,)
            # 如果验证/测试里只有单一类，AUC 不定义 -> 返回 None
            if np.unique(T).size >= 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    if num_classes == 2:
                        P = np.concatenate(all_probs, axis=0)    # (N,)
                        auc = roc_auc_score(T, P)                # 二分类用正类概率
                    else:
                        P = np.concatenate(all_probs, axis=0)    # (N, C)
                        auc = roc_auc_score(T, P, multi_class='ovr', average='macro')
                except Exception:
                    auc = None
            else:
                auc = None

        # 把可能的 nan 强制抹掉
        if isinstance(auc, float) and (np.isnan(auc) or np.isinf(auc)):
            auc = None

    avg_loss = epoch_loss / max(1, total_samples)
    return avg_loss, acc, per_class, auc




def train_and_eval_one(model, train_loader, val_loader, test_loader, epochs, device, seed, num_classes, is_multilabel=False):
    import numpy as np

    # ---------- 统计并提示分布 ----------
    if is_multilabel:
        # 多标签：打印每个标签的阳性计数
        pos_counts = np.zeros(num_classes, dtype=np.float64)
        for _, y in train_loader:
            y = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
            if y.ndim == 1:
                y = y[:, None]
            pos_counts += y.sum(axis=0)
        print(f"ℹ️ Multilabel positive counts (train) = {pos_counts}")
    else:
        # 单标签：每类样本数
        counts = np.zeros(num_classes, dtype=np.float64)
        for _, label in train_loader.dataset:
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
            if isinstance(label, np.ndarray) and label.ndim > 0:
                label = int(label[0])
            counts[int(label)] += 1
        if (counts > 0).sum() < num_classes:
            print(f"⚠️ Warning: Some classes are missing in training set! counts = {counts}")

    # ---------- 损失函数 ----------
    if is_multilabel:
        # 可选：计算 pos_weight，缓解不平衡
        # 这里简单用 train_loader 再扫一遍统计
        n_samples = 0
        pos = torch.zeros(num_classes, dtype=torch.float32)
        for _, yb in train_loader:
            yb = yb.float()
            if yb.ndim == 1:
                yb = yb.unsqueeze(1)
            pos += yb.sum(dim=0)
            n_samples += yb.shape[0]
        neg = n_samples - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(0, 20.0).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # 单标签：CrossEntropy（你的 class_weights 逻辑可选接回去）
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_pc, best_auc = 0., None, 0.
    best_test_acc, best_test_auc, best_test_pc = 0., 0., None
    history = []

    import time

    for ep in range(1, epochs + 1):
        ep_start = time.time()

        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                          num_classes=num_classes, is_multilabel=is_multilabel)
        val_loss, val_acc, val_pc, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False,
                                                       num_classes=num_classes, is_multilabel=is_multilabel)
        test_loss, test_acc, test_pc, test_auc = run_epoch(model, test_loader, criterion, None, device, train=False,
                                                           num_classes=num_classes, is_multilabel=is_multilabel)

        scheduler.step()

        if val_acc > best_acc:
            best_acc, best_pc, best_auc = val_acc, val_pc, val_auc
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), f"weights/{short_hash(str(model))}_best_seed{seed}.pth")
            best_test_acc, best_test_auc, best_test_pc = test_acc, test_auc, test_pc

        history.append({
            "epoch": ep,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss,   "val_acc": val_acc,   "val_auc": val_auc,
            "test_loss": test_loss, "test_acc": test_acc, "test_auc": test_auc,
            "test_per_class": test_pc.tolist() if isinstance(test_pc, np.ndarray) else (
                test_pc.tolist() if hasattr(test_pc, 'tolist') else None)
        })

        elapsed = time.time() - ep_start
        print(f"[{ep:03d}/{epochs}] "
              f"tr_acc={tr_acc:.2f} val_acc={val_acc:.2f} val_auc={val_auc if val_auc is None else round(val_auc,3)} "
              f"test_acc={test_acc:.2f} test_auc={test_auc if test_auc is None else round(test_auc,3)} "
              f"(elapsed {elapsed:.2f}s)")

    return best_acc, best_pc, best_auc, history, model, best_test_acc, best_test_auc, best_test_pc






# ---------- 入口 ----------
def main(args):
    # -------- 读取架构列表 --------
    with open(args.arch_json) as f:
        arch_dict = json.load(f)
    arch_list = list(arch_dict.keys()) if isinstance(arch_dict, dict) else arch_dict

    # -------- 数据加载 --------
    if args.dataset in {'pathmnist', 'organamnist', 'dermamnist','organcmnist','organsmnist','chestmnist','pneumoniamnist','breastmnist','retinamnist','bloodmnist','tissuemnist','vesselmnist'}:
        train_loader, val_loader, test_loader, in_ch, n_cls = get_medmnist_dataloaders(
            dataset_name=args.dataset,
            imb_type='exp',
            imb_factor=1.0,
            batch_size=args.batch_size,
            val_from_test = True  # ✅ 新增这一行
        )
    elif args.dataset == 'isic':
        train_loader, val_loader, test_loader,in_ch, n_cls = get_isic2019_loader(
            data_root="./datasets/ISIC_2019",
            batch_size=args.batch_size)


    else:  # CIFAR

        train_loader, val_loader, test_loader, in_ch, n_cls = get_dataloader(
            dataset=args.dataset,
            imbalance_type=args.imb_type,
            imbalance_ratio=args.imb_ratio,
            batch_size=args.batch_size
        )

    from collections import Counter
    import numpy as np
    import torch

    def _to_int(y):
        # 转成 numpy
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = np.asarray(y)

        # 删除多余维度，例如 [k]、[[k]]
        y = np.squeeze(y)

        if y.size == 1:
            # 标量标签
            return int(y.item())
        else:
            # 向量标签（one-hot 或 概率向量）→ 取 argmax
            return int(np.argmax(y))

    # 优先不读图，直接从 dataset 的标签属性拿
    train_labels = getattr(train_loader.dataset, 'targets', None)
    if train_labels is None:
        train_labels = getattr(train_loader.dataset, 'labels', None)

    if train_labels is not None:
        label_count = Counter(_to_int(v) for v in train_labels)
    else:
        # 兜底：遍历 dataset（可能会慢），确保转成 int
        label_count = Counter(_to_int(label) for _, label in train_loader.dataset)

    print(f"类别分布：{label_count}")

    # ---- 是否多标签 ----
    MED_KEYS = {'pathmnist', 'organamnist', 'dermamnist', 'organcmnist', 'organsmnist',
                'chestmnist', 'pneumoniamnist', 'breastmnist', 'retinamnist',
                'bloodmnist', 'tissuemnist', 'vesselmnist'}

    if args.dataset in MED_KEYS:
        task = INFO[args.dataset].get('task', '').lower()
        is_multilabel = ('multi-label' in task)
    else:
        # ISIC-2019 和 CIFAR 都是多类单标签
        is_multilabel = False

    # -------- 多随机种子 --------
    seed_list = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

    for arch_str in arch_list:
        accs, percs, aucs = [], [], []
        print(f"\n==== Evaluate {arch_str} ====")
        for sd in seed_list:
            print(f"[Seed {sd}]"); set_seed(sd)
            # 判定是 NAS 架构还是 baseline 名称
            if '+' in arch_str or arch_str in {'none','skip_connect','conv1x1','conv3x3'}:
                model = build_model_from_arch_str(arch_str, in_channels=in_ch, num_classes=n_cls)
            else:
                model = build_baseline(arch_str, in_channels=in_ch, num_classes=n_cls)
            model.to(device)

            best_acc, best_pc, best_auc, history, trained_model, test_acc, test_auc, test_pc = train_and_eval_one(
                model, train_loader, val_loader, test_loader, epochs=args.epochs, device=device, seed=sd,
                num_classes=n_cls, is_multilabel=is_multilabel)

            os.makedirs("logs", exist_ok=True)
            log_path = f"logs/log_{args.dataset}_{short_hash(arch_str)}_seed{sd}.json"
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=2)

            accs.append(best_acc); percs.append(best_pc);aucs.append(best_auc)

            # ✅ 显存清理
            del trained_model, model, history
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[After {arch_str} seed {sd}] memory_allocated = {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

        accs = np.array(accs); percs = np.array(percs)
        entry = {
            "arch": arch_str,
            "acc_list": accs.tolist(),
            "auc_list": aucs if isinstance(aucs, list) else aucs.tolist(),
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_per_class": percs.mean(0).tolist(),
            "std_per_class": percs.std(0).tolist(),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            # ✅ 新增：测试集记录
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_per_class": test_pc.tolist()
        }

        all_results.append(entry)
        print(f"=> {arch_str}: {entry['mean_acc']:.2f} ± {entry['std_acc']:.2f}%")

    # -------- 排序 & 保存 --------
    all_results.sort(key=lambda d: d['mean_acc'], reverse=True)
    print("\n=== Ranking ===")
    for i,r in enumerate(all_results,1):
        print(f"{i:02d}. {r['arch']:<50} {r['mean_acc']:.2f} ± {r['std_acc']:.2f}%")

    os.makedirs("results", exist_ok=True)
    fname = f"results/{args.dataset}_top{len(all_results)}_{len(seed_list)}seed_manual_ISIC.json"
    with open(fname,'w') as f: json.dump(all_results,f,indent=2)
    print(f"\nSaved → {fname}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch_json', required=True, help='JSON with arch strings / baseline names')
    ap.add_argument('--dataset', default='isic',
                    choices=['pathmnist', 'organamnist', 'dermamnist','organcmnist','organsmnist','chestmnist','pneumoniamnist','breastmnist','retinamnist','bloodmnist','tissuemnist','vesselmnist','cifar10','cifar100','isic'])
    ap.add_argument('--seeds', default='0', help='e.g. "0" or "0,1,2"')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--imb_type', default='exp')
    ap.add_argument('--imb_ratio', type=float, default=0.05)
    args = ap.parse_args()
    main(args)

   #   python train_best_arch_cu.py --arch_json ./archs/medarch.json



   #   python train_best_arch_cu.py --arch_json ./archs/medarch.json --epochs 1
