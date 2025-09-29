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
from sklearn.metrics import roc_auc_score, f1_score   # ← 新增 f1_score

# ---------- 自己的模块 ----------
from medmnist_get_imbalance import get_medmnist_dataloaders
from Dataset_partitioning import get_dataloader     # CIFAR 专用
from lib.dataop.ISIC_2019 import get_isic2019_loader
from lib.models.nas201_model import build_model_from_arch_str
import hashlib
from medmnist import INFO  # ← 新增：用来判断是否多标签
from sklearn.metrics import roc_auc_score

class DownsampleTo32(nn.Module):
    """把任意 [B,3,H,W] 在前向时下采样到 32×32，再喂给 backbone。"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        # 双线性下采样到 32×32，避免 224×224 触发 NB201 显存爆炸
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        return self.backbone(x)


# --- BUSI: Breast Ultrasound Images (classification 3 classes) ---
def get_busi_loader(
    data_root="./datasets/BUSI/Dataset_BUSI_with_GT",
    batch_size=128,
    val_ratio=0.2,
    test_ratio=0.2,
    img_size=224,
    seed=42,
):
    """
    读取目录结构:
      data_root/
        ├─ benign/
        ├─ malignant/
        └─ normal/
    返回: train_loader, val_loader, test_loader, in_channels(=3), num_classes(=3)
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import torch, numpy as np

    # 灰度→3通道 + 统一到 224（通用 CNN）
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # 用 ImageNet 规范化，和 resnet/mobilenet/efficientnet 更配
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(root=data_root, transform=trans)
    n = len(full)
    idx = np.arange(n)

    # 分层随机划分（按类别均衡切分）
    rng = np.random.RandomState(seed)
    labels = np.array([full.samples[i][1] for i in idx])

    def stratified_split(indices, y, test_ratio):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tr, te = next(sss.split(indices, y))
        return indices[tr], indices[te]

    # train+val / test
    idx_trv, idx_te = stratified_split(idx, labels, test_ratio=test_ratio)
    # train / val
    y_trv = labels[idx_trv]
    idx_tr, idx_va = stratified_split(idx_trv, y_trv, test_ratio=val_ratio / (1 - test_ratio))

    def mk(dl_idx, shuffle):
        return DataLoader(Subset(full, dl_idx), batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = mk(idx_tr, True)
    val_loader   = mk(idx_va, False)
    test_loader  = mk(idx_te, False)

    in_ch, n_cls = 3, len(full.classes)  # 3 类：benign/malignant/normal
    return train_loader, val_loader, test_loader, in_ch, n_cls


# --- ILPD: Indian Liver Patient Dataset (tabular) ---
def get_ilpd_loaders(batch_size=128, val_ratio=0.2, test_ratio=0.2, random_state=42):
    import numpy as np, torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=225)     # ILPD
        X = ds.data.features.copy()
        y = ds.data.targets.squeeze().copy()   # 'Selector' 列，取值通常为 {1,2}
    except Exception:
        # 兜底：如果你手里已有 CSV，就改成你的路径
        import pandas as pd
        df = pd.read_csv("ILPD.csv")
        y = df.get('Dataset', df.get('Selector')).copy()
        X = df.drop(columns=[c for c in ['Dataset','Selector'] if c in df.columns])

    # 统一标签到 {0,1}：1=肝病，2=非肝病
    import pandas as pd
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    y = np.asarray(y).astype(float)
    # 若确实是 {1,2}
    if np.all(np.isin(np.unique(y), [1,2])):
        y = (y == 1).astype(np.int64)
    else:
        # 其它异常情况：按中位数二值化（极少见）
        y = (y > np.median(y)).astype(np.int64)

    # 类别不平衡统计（可选）
    # print("Class counts:", {int(k): int((y==k).sum()) for k in np.unique(y)})

    # 性别编码（若存在 Gender 列）
    if 'Gender' in X.columns:
        X['Gender'] = X['Gender'].astype(str).str[0].map({'M':1,'F':0}).fillna(0)

    # 数值化与标准化
    X = X.select_dtypes(include=['number']).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # 分层划分：先 train+val / test，再从 train 切 val
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=val_ratio/(1-test_ratio),
        random_state=random_state, stratify=y_trv)

    # 组装 DataLoader
    def _mk(Xn, yn, bs, shuffle):
        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = _mk(X_tr, y_tr, batch_size, True)
    val_loader   = _mk(X_va, y_va, batch_size, False)
    test_loader  = _mk(X_te, y_te, batch_size, False)

    in_dim  = X.shape[1]     # = 10（标准 ILPD）
    n_cls   = 2              # 二分类
    return train_loader, val_loader, test_loader, in_dim, n_cls


# --- Breast Cancer Wisconsin (Diagnostic) | sklearn built-in ---
def get_breastcancer_loaders(batch_size=128, val_ratio=0.2, test_ratio=0.2, random_state=42):
    import numpy as np, torch, pandas as pd
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = load_breast_cancer(as_frame=True)  # 569×30, target: 0=malignant, 1=benign
    X = ds.data.astype('float32')
    y = pd.Series(ds.target.astype('int64'))

    scaler = StandardScaler()
    X = scaler.fit_transform(X.values).astype('float32')

    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=val_ratio/(1-test_ratio),
                                              random_state=random_state, stratify=y_trv)

    def _mk(Xn, yn, shuffle):
        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn.values if hasattr(yn,'values') else yn))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    in_dim, n_cls = X.shape[1], 2  # 30, 2
    return _mk(X_tr, y_tr, True), _mk(X_va, y_va, False), _mk(X_te, y_te, False), in_dim, n_cls




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
    # --- Tabular MLP for ILPD ---
    if name.startswith("mlp"):
        # 约定命名：mlp、mlp_64、mlp_64x2、mlp_128x3 等
        # 解析隐藏层设置
        hidden = []
        if "_" in name:
            spec = name.split("_", 1)[1]
            for part in spec.split("x"):
                if part.isdigit():
                    hidden.append(int(part))
        if not hidden:
            hidden = [64, 64]  # 默认两层 64

        layers = []
        in_dim = in_channels  # 这里 in_channels 代表输入维度（ILPD=10）
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(0.1)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes)]
        net = nn.Sequential(*layers)
        return net

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

    if num_classes is None:
        if criterion is not None and getattr(criterion, 'weight', None) is not None:
            num_classes = criterion.weight.numel()
        else:
            num_classes = get_model_output_classes(model)

    epoch_loss = 0.0
    total_samples = 0
    correct_total = 0.0

    cls_cor = np.zeros(num_classes, dtype=np.float64)
    cls_tot = 0

    # AUC/F1 收集容器
    all_probs = []      # eval 时：二类 (N,), 多类/多标签 (N,C)
    all_targets = []    # eval 时：二/多类 (N,), 多标签 (N,C)
    all_preds = []      # eval 时：二/多类 (N,), 多标签 (N,C)

    for x, y in loader:
        x = x.to(device)

        if is_multilabel:
            y = y.float().to(device)  # [B, C]
        else:
            if y.ndim > 1:
                y = y.squeeze(1)
            y = y.long().to(device)   # [B]

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
            preds = (probs > 0.5).float()        # [B, C]
            correct_total += preds.eq(y).float().mean().item() * bs

            cls_cor += preds.eq(y).float().sum(dim=0).detach().cpu().numpy()
            cls_tot += bs

            if not train:
                all_probs.append(probs.detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())
        else:
            # 单标签
            pred = logits.argmax(1)
            correct_total += pred.eq(y).float().sum().item()

            for t, p in zip(y.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                cls_cor[int(t)] += (int(t) == int(p))
            cls_tot += 1

            if not train:
                sm = F.softmax(logits, dim=1).detach().cpu().numpy()  # (B, C)
                if num_classes == 2:
                    pos_prob = sm[:, 1]
                    all_probs.append(pos_prob)                         # (B,)
                else:
                    all_probs.append(sm)                               # (B, C)
                all_targets.append(y.detach().cpu().numpy())          # (B,)
                all_preds.append(pred.detach().cpu().numpy())         # (B,)

    acc = 100.0 * (correct_total / max(1, total_samples))

    # per-class 粗略统计与你原逻辑一致
    if is_multilabel:
        per_class = 100.0 * (cls_cor / (cls_tot + 1e-12))
    else:
        per_class = 100.0 * cls_cor / (np.maximum(1.0, cls_tot))

    # ---- 计算 AUC ----
    auc = None
    if not train and len(all_probs) > 0:
        try:
            if is_multilabel:
                P = np.concatenate(all_probs, axis=0)   # (N, C)
                T = np.concatenate(all_targets, axis=0) # (N, C)
                auc = roc_auc_score(T, P, average='macro')
            else:
                T = np.concatenate(all_targets, axis=0)  # (N,)
                if np.unique(T).size >= 2:
                    if num_classes == 2:
                        P = np.concatenate(all_probs, axis=0)    # (N,)
                        auc = roc_auc_score(T, P)
                    else:
                        P = np.concatenate(all_probs, axis=0)    # (N, C)
                        auc = roc_auc_score(T, P, multi_class='ovr', average='macro')
        except Exception:
            auc = None
        if isinstance(auc, float) and (np.isnan(auc) or np.isinf(auc)):
            auc = None

    # ---- 计算 F1（macro）----
    f1 = None
    if not train and len(all_preds) > 0:
        try:
            if is_multilabel:
                Y_true = np.concatenate(all_targets, axis=0)  # (N, C)
                Y_pred = np.concatenate(all_preds, axis=0)    # (N, C)
                # 多标签：macro over labels
                f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
            else:
                Y_true = np.concatenate(all_targets, axis=0)  # (N,)
                Y_pred = np.concatenate(all_preds, axis=0)    # (N,)
                f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
        except Exception:
            f1 = None

    avg_loss = epoch_loss / max(1, total_samples)
    return avg_loss, acc, per_class, auc, f1



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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)

    best_acc, best_pc, best_auc = 0., None, 0.
    best_test_acc, best_test_auc, best_test_pc = 0., 0., None
    history = []

    import time

    for ep in range(1, epochs + 1):
        ep_start = time.time()

        # 训练
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                             num_classes=num_classes, is_multilabel=is_multilabel)

        # 验证
        val_loss, val_acc, val_pc, val_auc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device,
                                                               train=False,
                                                               num_classes=num_classes, is_multilabel=is_multilabel)

        # 测试
        test_loss, test_acc, test_pc, test_auc, test_f1 = run_epoch(model, test_loader, criterion, None, device,
                                                                    train=False,
                                                                    num_classes=num_classes,
                                                                    is_multilabel=is_multilabel)

        best_acc, best_pc, best_auc, best_f1 = 0., None, 0., 0.
        best_test_acc, best_test_auc, best_test_pc, best_test_f1 = 0., 0., None, 0.

        if val_acc > best_acc:
            best_acc, best_pc, best_auc, best_f1 = val_acc, val_pc, val_auc, val_f1
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), f"weights/{short_hash(str(model))}_best_seed{seed}.pth")
            best_test_acc, best_test_auc, best_test_pc, best_test_f1 = test_acc, test_auc, test_pc, test_f1

        history.append({
            "epoch": ep,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss,   "val_acc": val_acc,   "val_auc": val_auc,"val_f1": val_f1,
            "test_loss": test_loss, "test_acc": test_acc, "test_auc": test_auc,"test_f1": test_f1,
            "test_per_class": test_pc.tolist() if isinstance(test_pc, np.ndarray) else (
                test_pc.tolist() if hasattr(test_pc, 'tolist') else None)
        })

        elapsed = time.time() - ep_start
        print(f"[{ep:03d}/{epochs}] "
              f"tr_acc={tr_acc:.2f} "
              f"val_acc={val_acc:.2f} val_auc={val_auc if val_auc is None else round(val_auc, 3)} val_f1={val_f1 if val_f1 is None else round(val_f1, 3)} "
              f"test_acc={test_acc:.2f} test_auc={test_auc if test_auc is None else round(test_auc, 3)} test_f1={test_f1 if test_f1 is None else round(test_f1, 3)} "
              f"(elapsed {elapsed:.2f}s)")

    return best_acc, best_pc, best_auc, history, model, best_test_acc, best_test_auc, best_test_pc, best_f1, best_test_f1







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

    elif args.dataset == 'breastcancer':
        train_loader, val_loader, test_loader, in_ch, n_cls = get_breastcancer_loaders(
            batch_size=args.batch_size)

    elif args.dataset == 'busi':
        train_loader, val_loader, test_loader, in_ch, n_cls = get_busi_loader(
            data_root="./datasets/BUSI/Dataset_BUSI_with_GT",
            batch_size=args.batch_size,
            img_size=224  # 若要跑 NB201 串可改为 32
        )

    elif args.dataset == 'ilpd':

        train_loader, val_loader, test_loader, in_ch, n_cls = get_ilpd_loaders(

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
                if args.dataset not in {'cifar10', 'cifar100'}:
                    model = DownsampleTo32(model)
            else:
                model = build_baseline(arch_str, in_channels=in_ch, num_classes=n_cls)
            model.to(device)

            best_acc, best_pc, best_auc, history, trained_model, test_acc, test_auc, test_pc, best_f1, best_test_f1 = train_and_eval_one(
                model, train_loader, val_loader, test_loader, epochs=args.epochs, device=device, seed=sd,
                num_classes=n_cls, is_multilabel=is_multilabel)

            # 保存单架构多种子统计
            accs.append(best_acc)
            percs.append(best_pc)
            aucs.append(best_auc)
            f1s = locals().get('f1s', [])  # 简便写法，首次循环自动新建
            f1s.append(best_f1)

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
            "f1_list": f1s if isinstance(f1s, list) else list(f1s),
            "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
            "mean_per_class": percs.mean(0).tolist(), "std_per_class": percs.std(0).tolist(),
            "mean_auc": float(np.nanmean(aucs)), "std_auc": float(np.nanstd(aucs)),
            "mean_f1": float(np.nanmean(f1s)), "std_f1": float(np.nanstd(f1s)),
            "test_acc": test_acc, "test_auc": test_auc, "test_f1": best_test_f1,
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
    ap.add_argument('--dataset', default='busi',
                    choices=['pathmnist', 'organamnist', 'dermamnist','organcmnist','organsmnist','chestmnist','pneumoniamnist','breastmnist','retinamnist','bloodmnist','tissuemnist','vesselmnist','cifar10','cifar100','isic','ilpd','breastcancer','busi'])
    ap.add_argument('--seeds', default='0', help='e.g. "0" or "0,1,2"')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--imb_type', default='exp')
    ap.add_argument('--imb_ratio', type=float, default=0.05)
    args = ap.parse_args()
    main(args)

   #   python train_ISIC.py --arch_json isic_arch.json --epochs 1


   #   python train_ISIC.py --arch_json ./archs/medarch.json --epochs 1
