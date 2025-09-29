import os
from medmnist import INFO
from medmnist import __dict__ as medmnist_dict
from collections import Counter
import numpy as np

def compute_imbalance(labels):
    counts = Counter(labels)
    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count > 0 else float('inf')
    return max_count, min_count, ratio

print(f"{'Subset':<20}{'Classes':<10}{'Max':<10}{'Min':<10}{'Imbalance':<12}")
print("="*60)

for subset_key, info in INFO.items():
    DataClass = medmnist_dict.get(info['python_class'])
    if DataClass is None:
        continue

    try:
        # 👇👇👇 修改这里：显式传入 root 路径
        dataset = DataClass(split='train', download=True, root='datasets')
        labels = [int(label[0]) for _, label in dataset]
        num_classes = len(set(labels))
        max_c, min_c, ratio = compute_imbalance(labels)

        print(f"{subset_key:<20}{num_classes:<10}{max_c:<10}{min_c:<10}{ratio:<12.2f}")
    except Exception as e:
        print(f"{subset_key:<20} ERROR: {e}")
