# select_high_mid_low.py
import json
from pathlib import Path


def select_subsets(score_path, k_each=30, out_path="arch_selected.json"):
    with open(score_path) as f:
        scores = json.load(f)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:k_each]
    mid = sorted_items[len(sorted_items) // 2 - k_each // 2: len(sorted_items) // 2 + k_each // 2]
    low = sorted_items[-k_each:]

    selected = dict(top + mid + low)
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"✅ 选择完成: {len(selected)} 架构 → {out_path}")
    print(f"Top-1: {top[0][1]:.6f}, Mid: {mid[k_each // 2][1]:.6f}, Bottom: {low[-1][1]:.6f}")


if __name__ == "__main__":
    select_subsets("arch_fisher_patch_scores.json", k_each=30)


#    python select_arch.py

