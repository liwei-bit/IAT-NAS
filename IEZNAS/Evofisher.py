import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from lib.nas_201_api import NASBench201API
import torch

# --- 依赖自定义模块（路径根据你的项目结构修改）---
from lib.procedures.fisher_proxy import tail_aware_fisher_proxy
from Dataset_partitioning import get_imbalanced_dataset
from lib.models.nas201_model import build_model_from_arch_str


def load_arch_list(path: str, dataset='cifar10'):
    if path.lower() == 'nasbench201':
        api = NASBench201API('searchspace/NAS-Bench-201-v1_1-096897.pth')
        arch_list = []
        for i in range(len(api)):
            arch_dict = api.get_net_config(i, dataset)  # 返回dict
            arch = arch_dict['arch_str']               # 从dict取出arch_str
            arch_list.append(arch)
        return arch_list
    else:
        p = Path(path)
        if p.suffix == '.json':
            data = json.loads(p.read_text())
            return list(data.keys()) if isinstance(data, dict) else data
        else:
            return [line.strip() for line in p.open() if line.strip()]


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 构造不平衡数据集 DataLoader
    loader = get_imbalanced_dataset(
        dataset_name=args.dataset,
        imb_type=args.imb_type,
        imb_factor=args.imb_factor,
        batch_size=args.batch_size,
        shuffle=False,
        debug=False,
    )

    # 2. 读取架构列表，传入 dataset 参数
    arch_list = load_arch_list(args.arch_list, dataset=args.dataset)
    print(f"Loaded {len(arch_list)} architectures.")

    scores = {}
    t0 = time.time()

    # 使用 tqdm 显示进度条
    for idx, arch_str in enumerate(tqdm(arch_list, desc="Scoring architectures"), 1):
        model = build_model_from_arch_str(
            arch_str,
            num_classes=100 if args.dataset == 'cifar100' else 10
        )
        model.to(device)

        # 3. 计算 Tail-Aware Fisher 评分
        score = tail_aware_fisher_proxy(model, loader)
        scores[arch_str] = score

        elapsed = time.time() - t0
        tqdm.write(f"[{idx}/{len(arch_list)}] score={score:.4f}  time={elapsed / idx:.2f}s/arch")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 4. 排序并保存结果
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    Path(args.out).write_text(json.dumps(sorted_scores, indent=2))
    print(f"Saved sorted scores to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail-Aware Fisher Proxy scoring")
    parser.add_argument('--arch-list', required=True,
                        help="arch list file path, or 'nasbench201' to load NAS-Bench-201 full search space")
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--imb-type', default='exp', choices=['exp', 'step'])
    parser.add_argument('--imb-factor', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--out', default='tail_fisher_scores.json')
    args = parser.parse_args()
    main(args)


# python Evofisher.py --arch-list nasbench201 --dataset cifar10