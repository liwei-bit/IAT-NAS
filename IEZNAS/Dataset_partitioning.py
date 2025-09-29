import numpy as np
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_imbalanced_dataset(
    dataset_name='cifar10',
    imb_type='exp',
    imb_factor=0.01,
    batch_size=128,
    shuffle=True,
    seed=42,
    debug=True,
    save_index=True,
    index_dir='./imb_indices',
    val_split=0.1  # 👈 新增参数：验证集划分比例
):
    np.random.seed(seed)

    # ---------- 1. 加载全数据 ----------
    if dataset_name == 'cifar10':
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        full_train_set = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=tf_train)
        test_set = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=tf_test)
        class_num = 10
        in_channels = 3

    elif dataset_name == 'cifar100':
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        full_train_set = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=tf_train)
        test_set = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=tf_test)
        class_num = 100
        in_channels = 3

    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    # ---------- 2. 不平衡划分 ----------
    targets_np = np.array(full_train_set.targets)
    img_num_per_cls = []
    max_num = len(full_train_set) // class_num

    if imb_type == 'exp':
        for cls_idx in range(class_num):
            ratio = cls_idx / (class_num - 1.0)
            num = max_num * (imb_factor ** ratio)
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(class_num):
            if cls_idx < class_num // 2:
                img_num_per_cls.append(max_num)
            else:
                img_num_per_cls.append(int(max_num * imb_factor))
    else:
        raise ValueError("imb_type must be 'exp' or 'step'")

    os.makedirs(index_dir, exist_ok=True)
    index_file = os.path.join(index_dir, f"{dataset_name}_{imb_type}_{imb_factor}.npy")

    if os.path.exists(index_file):
        print(f"[加载已保存索引] {index_file}")
        selected_indices = np.load(index_file)
    else:
        selected_indices = []
        for cls_idx, num in enumerate(img_num_per_cls):
            idx = np.where(targets_np == cls_idx)[0]
            np.random.shuffle(idx)
            selected_indices.extend(idx[:num])
        selected_indices = np.array(selected_indices)
        if save_index:
            np.save(index_file, selected_indices)
            print(f"[已保存索引] {index_file}")

    # ---------- 3. Train/Val 划分 ----------
    train_idx, val_idx = train_test_split(
        selected_indices, test_size=val_split, random_state=seed, stratify=targets_np[selected_indices])

    train_subset = Subset(full_train_set, train_idx)
    val_subset = Subset(full_train_set, val_idx)

    if debug:
        from collections import Counter
        val_targets = [full_train_set.targets[i] for i in val_idx]
        counter = Counter(val_targets)
        print("Validation Sample Distribution:")
        for cls_idx in range(class_num):
            print(f"  Class {cls_idx}: {counter.get(cls_idx, 0)} images")

    # ---------- 4. 构造 Loader ----------
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, in_channels, class_num


def get_dataloader(dataset, imbalance_type, imbalance_ratio, batch_size=128):
    return get_imbalanced_dataset(
        dataset_name=dataset,
        imb_type=imbalance_type,
        imb_factor=imbalance_ratio,
        batch_size=batch_size,
        val_split=0.1  # 可选改为 0.2 等
    )

