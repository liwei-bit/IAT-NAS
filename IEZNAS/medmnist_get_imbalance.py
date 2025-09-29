from __future__ import annotations
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import INFO
from medmnist import __dict__ as medmnist_dict
from sklearn.model_selection import train_test_split

def get_medmnist_dataloaders(dataset_name: str = 'pathmnist',
                             imb_type: str | None = None,
                             imb_factor: float = 1.0,
                             batch_size: int = 128,
                             num_workers: int = 4,
                             device: str = 'cuda',
                             seed: int = 42,
                             val_split: float = 0.1,
                             val_from_test: bool | None = None,  # 兼容旧参数（忽略）
                             **kwargs):
    """
    通用 MedMNIST 加载器：
      - 多标签：直接使用官方 train/val/test 三分（不切分、不分层）。
      - 单标签：从 train 中按 val_split 切出验证集（尽量分层；不足则回退随机）。
      - 仅对训练集支持不平衡构造（单标签）。

    返回: train_loader, val_loader, test_loader, in_channels, num_classes
    """
    rng = np.random.default_rng(seed)
    info = INFO[dataset_name]
    DataClass = medmnist_dict[info['python_class']]
    in_channels = info['n_channels']
    num_classes = len(info['label'])

    # 判断是否多标签
    task = info.get('task', '')
    is_multilabel = ('multi-label' in task.lower())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*in_channels, std=[.5]*in_channels),
    ])

    if is_multilabel:
        # ==== 多标签：用官方三分 ====
        train_set = DataClass(split='train', root='datasets', download=True, transform=transform)
        val_set   = DataClass(split='val',   root='datasets', download=True, transform=transform)
        test_set  = DataClass(split='test',  root='datasets', download=True, transform=transform)

        # 多标签默认不做不平衡改造（避免破坏标签相关性）
        # 如确需做多标签不平衡，请改为迭代分层方案（另写）
        # 这里保持训练集原样
        # （如果你硬要下采样，至少确保不改变标签矩阵的稀疏结构）
        pass
    else:
        # ==== 单标签：从 train 中切分出 val（尽量分层） ====
        train_set_full = DataClass(split='train', root='datasets', download=True, transform=transform)
        labels = np.array([int(y) for _, y in train_set_full]).reshape(-1)

        # 训练集不平衡（可选）
        if imb_type is not None and imb_factor < 1.0:
            sel_idx_all = _make_imbalance_indices(labels, imb_type=imb_type, imb_factor=imb_factor, random_seed=seed)
        else:
            sel_idx_all = np.arange(len(train_set_full))

        # 分层切分前检查每类样本是否足够
        can_stratify = True
        try:
            # 选中样本的标签
            labels_sel = labels[sel_idx_all]
            # 最小类样本数
            _, counts = np.unique(labels_sel, return_counts=True)
            min_count = counts.min() if len(counts) > 0 else 0
            # 估算分层切分要求：每类至少要能分出 2 个样本（train/val 都 >=1）
            # 更严格点：val 至少 1，train 至少 1
            if min_count < 2:
                can_stratify = False
        except Exception:
            can_stratify = False

        if can_stratify:
            train_idx, val_idx = train_test_split(
                sel_idx_all,
                test_size=val_split,
                random_state=seed,
                stratify=labels_sel
            )
        else:
            # 回退：不分层随机切分（打印一次提示）
            print("[warn] Stratified split skipped (class count too small). Falling back to random split.")
            train_idx, val_idx = train_test_split(
                sel_idx_all,
                test_size=val_split,
                random_state=seed,
                shuffle=True
            )

        train_set = Subset(train_set_full, train_idx)
        val_set   = Subset(train_set_full, val_idx)
        test_set  = DataClass(split='test', root='datasets', download=True, transform=transform)

    pin = (device == 'cuda')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, in_channels, num_classes
