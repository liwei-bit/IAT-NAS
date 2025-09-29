from collections import Counter
import numpy as np
import os
import pandas as pd

# 路径为 ISIC_2019 官方标注文件（包含图像标签信息）
csv_path = "./datasets/ISIC_2019/ISIC_2019_Training_GroundTruth.csv"

# 加载标签
df = pd.read_csv(csv_path)
df = df.drop(columns=["image"])  # 去除图像名列，只保留标签列（多标签 one-hot）

# 将 one-hot 转为单标签索引（若为多标签可做特殊处理）
labels = df.values.argmax(axis=1)  # 单标签模式（只取 one-hot 中最大的）

counts = Counter(labels)
max_c = max(counts.values())
min_c = min(counts.values())
imbalance = max_c / min_c

print("ISIC-2019 类别统计:")
for cls, count in counts.items():
    print(f"Class {cls}: {count} samples")

print(f"\nMax count: {max_c}, Min count: {min_c}, Imbalance ratio: {imbalance:.2f}")


# python ISIC_imbalance.py