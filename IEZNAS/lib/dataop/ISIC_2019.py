#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ISIC 2019 DataLoader（兼容 NB201：3×S×S，默认 S=32）
- 自动解压 zip（若目标目录不存在）
- 读取官方 CSV（第一列 image，其余为 one-hot；若存在 UNK 列将自动忽略）
- 过滤缺失图像、大小写/后缀不一致的文件（.jpg/.JPG/.png/.jpeg/...）
- 分层划分：默认 70%/15%/15%
- 返回：train_loader, val_loader, test_loader, in_ch(=3), n_cls(=类别数)
"""

import os, csv, random, zipfile
from typing import List, Tuple, Optional
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 支持的图像后缀
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def unzip_images(zip_path: str, extract_to: str):
    """若 extract_to 不存在，则解压 zip_path 到该目录。"""
    if os.path.isdir(extract_to):
        print(f"[✓] 图像目录已存在: {extract_to}")
        return
    if not os.path.isfile(zip_path):
        print(f"[!] 未找到压缩包：{zip_path}（若已解压可以忽略）")
        return
    print(f"[...] 解压中: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"[✓] 解压完成: {extract_to}")

def _resolve_img_path(img_dir: str, stem: str) -> Optional[str]:
    """根据文件名 stem（无扩展名）在 img_dir 中匹配常见扩展名。"""
    for ext in VALID_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def _stratified_split(y: List[int], seed: int, val_ratio: float, test_ratio: float):
    """分层划分 train/val/test；优先用 sklearn；失败则使用 fallback。"""
    idx = np.arange(len(y))
    y = np.array(y)

    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trainval_idx, test_idx = next(sss1.split(idx, y))
        y_trainval = y[trainval_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(1.0 - test_ratio), random_state=seed)
        train_idx, val_idx = next(sss2.split(trainval_idx, y_trainval))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    except Exception:
        # 简易 fallback：每类内按比例切分
        rng = random.Random(seed)
        train_idx, val_idx, test_idx = [], [], []
        y_list = y.tolist()
        classes = sorted(set(y_list))
        for c in classes:
            cls_idx = [i for i, yy in enumerate(y_list) if yy == c]
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            cur_test = cls_idx[:n_test]
            cur_val  = cls_idx[n_test:n_test+n_val]
            cur_train = cls_idx[n_test+n_val:]
            test_idx.extend(cur_test)
            val_idx.extend(cur_val)
            train_idx.extend(cur_train)
        return train_idx, val_idx, test_idx

class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir: str, items: List[Tuple[str, int]], transform):
        self.img_dir = img_dir
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, y = self.items[i]
        p = _resolve_img_path(self.img_dir, stem)
        if p is None:
            raise FileNotFoundError(f"Image not found for '{stem}' in {self.img_dir}")
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y

def get_isic2019_loader(
    data_root: str = "datasets/ISIC_2019",
    batch_size: int = 128,
    image_size: int = 32,          # 建议 32 以契合 NB201（更快更省显存）
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment: bool = False,
    seed: int = 0,
    num_workers: int = 4,
):
    """
    期望目录结构：
      data_root/
        ISIC_2019_Training_Input.zip   (可选)
        ISIC_2019_Training_Input/      (已解压图像目录)
        ISIC_2019_Training_GroundTruth.csv
    """
    zip_path = os.path.join(data_root, "ISIC_2019_Training_Input.zip")
    img_dir  = os.path.join(data_root, "ISIC_2019_Training_Input")
    csv_path = os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv")

    # 自动解压（若目录不存在）
    unzip_images(zip_path, img_dir)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if len(cols) < 2 or not cols[0].lower().startswith('image'):
        raise ValueError(f"Unexpected CSV header: {cols[:5]} ...")

    # 处理 UNK：若存在则忽略；否则使用所有类别列
    if 'UNK' in cols:
        cls_start = 1
        cls_end = cols.index('UNK')  # 不含 UNK
        class_cols = cols[cls_start:cls_end]
    else:
        class_cols = cols[1:]

    # 先过滤缺图样本
    keep_mask = df['image'].apply(lambda s: _resolve_img_path(img_dir, s) is not None)
    miss_cnt = (~keep_mask).sum()
    if miss_cnt > 0:
        print(f"[ISIC2019] 跳过缺图样本 {miss_cnt} 条。")
    df = df[keep_mask].reset_index(drop=True)

    # one-hot -> index
    label_indices = df[class_cols].values.argmax(axis=1)
    stems = df['image'].tolist()
    n_cls = len(class_cols)

    # 分层划分
    train_idx, val_idx, test_idx = _stratified_split(label_indices, seed, val_ratio, test_ratio)

    def _build_items(indices):
        return [(stems[i], int(label_indices[i])) for i in indices]

    train_items = _build_items(train_idx)
    val_items   = _build_items(val_idx)
    test_items  = _build_items(test_idx)

    # 变换
    aug_ops = [T.RandomHorizontalFlip()] if augment else []
    tf_train = T.Compose([T.Resize((image_size, image_size)), *aug_ops, T.ToTensor(),
                          T.Normalize([0.5]*3, [0.5]*3)])
    tf_eval  = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(),
                          T.Normalize([0.5]*3, [0.5]*3)])

    # 数据集与加载器
    train_set = ISIC2019Dataset(img_dir, train_items, tf_train)
    val_set   = ISIC2019Dataset(img_dir, val_items,   tf_eval)
    test_set  = ISIC2019Dataset(img_dir, test_items,  tf_eval)

    # Windows 下 persistent_workers 可能不稳定；做个保守判定
    persistent_ok = (num_workers > 0) and (os.name != 'nt')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent_ok)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent_ok)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent_ok)

    in_ch = 3
    return train_loader, val_loader, test_loader, in_ch, n_cls
