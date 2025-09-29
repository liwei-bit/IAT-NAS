#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dump 500 NAS-Bench-201 architectures and their CIFAR-10 / CIFAR-100 accuracies.

- Randomly sample N unique architectures from NB201
- Read test accuracies for CIFAR-10 and CIFAR-100 from the API (no training)
- Save to CSV: [idx, arch_str, acc_cifar10, acc_cifar100]

Requirements:
    pip install nas-bench-201    # or ensure nas_201_api is importable
Data:
    NAS-Bench-201-v1_1-096897.pth

Example:
    python dump_nb201_acc.py \
      --api_path ./searchspace/NAS-Bench-201-v1_1-096897.pth \
      --n_arch 500 --hp 200 \
      --out_csv results/nb201_500_acc_c10_c100.csv \
      --seed 0
"""

import os, sys
sys.path.insert(0, os.path.abspath("./lib"))
from nas_201_api import NASBench201API as API
import csv
import json
import time
import random
import argparse
from typing import Optional, Any, Dict

try:
    # Official name in pip
    from nas_201_api import NASBench201API as API
except Exception:
    # Some repos expose the same class via nas_201_api.* namespace; if you have another path, modify here.
    from nas_201_api import NASBench201API as API  # type: ignore

from tqdm import tqdm


def get_arch_str_safe(api: API, idx: int) -> str:
    """Try several NB201 API access patterns to obtain arch string."""
    # Path 1: query_by_index → object with .arch_str
    try:
        obj = api.query_by_index(idx)
        if hasattr(obj, "arch_str"):
            return obj.arch_str if isinstance(obj.arch_str, str) else str(obj.arch_str)
    except Exception:
        pass
    # Path 2: get_arch_str (exists in some versions)
    try:
        return api.get_arch_str(idx)  # type: ignore
    except Exception:
        pass
    # Path 3: index2arch / arch / tostr fallbacks
    try:
        arch = api.arch(idx)  # type: ignore
        if hasattr(arch, "tostr"):
            return arch.tostr()  # type: ignore
    except Exception:
        pass
    # Last resort: stringify whatever
    return f"arch_idx_{idx}"


def fetch_acc(api: API, idx: int, dataset: str, hp: str = "200") -> Optional[float]:
    """
    Fetch accuracy from NB201 API.
    For CIFAR-10 / CIFAR-100 we prefer 'test-accuracy'. If missing, try 'valid-accuracy' or 'accuracy'.
    """
    try:
        # is_random=False → averaged over the 3 repeated runs in NB201
        info: Dict[str, Any] = api.get_more_info(idx, dataset, hp=hp, is_random=False)  # type: ignore
    except Exception:
        return None

    # Common keys in NB201 dict:
    # 'test-accuracy', 'train-accuracy', 'valid-accuracy', 'accuracy', etc.
    for key in ("test-accuracy", "valid-accuracy", "accuracy"):
        if key in info and info[key] is not None:
            try:
                return float(info[key])
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_path", type=str, required=True, help="Path to NAS-Bench-201-v1_1-096897.pth")
    parser.add_argument("--n_arch", type=int, default=500, help="Number of unique architectures to sample")
    parser.add_argument("--hp", type=str, default="200", choices=["12", "200"], help="Which training budget to query")
    parser.add_argument("--out_csv", type=str, default="nb201_500_acc_c10_c100.csv", help="Output CSV path")
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON dump path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    print(f"[INFO] Loading NB201 API from: {args.api_path}")
    t0 = time.time()
    api = API(args.api_path)
    total = len(api)
    print(f"[OK] Loaded. Total architectures = {total}. Time: {time.time()-t0:.2f}s")

    n = min(args.n_arch, total)
    random.seed(args.seed)
    indices = random.sample(range(total), n)

    rows = []
    print(f"[INFO] Sampling {n} architectures (hp={args.hp})...")
    for idx in tqdm(indices, ncols=100):
        arch_str = get_arch_str_safe(api, idx)
        # CIFAR-10: use dataset key 'cifar10' (NB201 provides test accuracies)
        acc_c10 = fetch_acc(api, idx, "cifar10", hp=args.hp)
        # CIFAR-100: dataset key 'cifar100'
        acc_c100 = fetch_acc(api, idx, "cifar100", hp=args.hp)

        rows.append({
            "idx": idx,
            "arch_str": arch_str,
            "acc_cifar10_test": acc_c10,
            "acc_cifar100_test": acc_c100,
        })

    # Save CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "arch_str", "acc_cifar10_test", "acc_cifar100_test"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Saved CSV: {args.out_csv}")

    # Optional JSON
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved JSON: {args.out_json}")

    # Preview first 5
    print("\n[PREVIEW: first 5 rows]")
    for r in rows[:5]:
        print(r)

    print("\n[NOTE]")
    print("- acc_cifar10_test / acc_cifar100_test 单位是百分数（如 93.25 表示 93.25%）。")
    print("- 如果某些条目是 None，通常是你的 API 版本不匹配或 hp 设置缺少该条目，可尝试 --hp 12 或检查 API 文件。")


if __name__ == "__main__":
    main()
