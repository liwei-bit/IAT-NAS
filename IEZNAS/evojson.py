# filename: score_archs_with_fisher.py

import os, json, argparse
import torch
from tqdm import tqdm

from lib.models.nas201_model import build_model_from_arch_str
from Dataset_partitioning import get_dataloader

# ✅ 从单个 proxy 文件中导入五个函数（假设你文件名是 lib/procedures/proxy.py）
from lib.procedures.otherproxies import (
    snip_proxy, jaccov_proxy, synflow_proxy, gradnorm_proxy, fisher_proxy
)

# ✅ 所有支持的代理函数字典
PROXIES = {
    'snip': snip_proxy,
    'jaccov': jaccov_proxy,
    'synflow': synflow_proxy,
    'gradnorm': gradnorm_proxy,
    'fisher': fisher_proxy
}

def set_seed(seed=0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(0)

    # === 载入架构列表 ===
    with open(args.arch_json, 'r') as f:
        arch_acc_dict = json.load(f)
    arch_list = list(arch_acc_dict.keys())
    print(f"共载入 {len(arch_list)} 个架构")

    # === 获取数据集加载器 ===
    data_loader = get_dataloader(
        dataset=args.dataset,
        imbalance_type=args.imb_type,
        imbalance_ratio=args.imb_ratio,
        batch_size=args.batch_size
    )

    # === 选择代理指标函数 ===
    if args.proxy not in PROXIES:
        raise ValueError(f"未知 proxy 指标: {args.proxy}, 可选项: {list(PROXIES.keys())}")
    proxy_fn = PROXIES[args.proxy]

    # === 对所有架构进行打分 ===
    scores = {}
    for arch_str in tqdm(arch_list, desc=f"计算 {args.proxy} 分数"):
        model = build_model_from_arch_str(arch_str, num_classes=args.num_classes)
        score = proxy_fn(model, data_loader, args=None)
        scores[arch_str] = float(score)

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"[√] {args.proxy} 评分完成，结果已保存至: {args.out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_json', type=str, required=True, help='输入架构 JSON 路径')
    parser.add_argument('--out_json', type=str, default='arch_scores_fisher.json', help='输出路径')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--imb_type', type=str, default='exp')
    parser.add_argument('--imb_ratio', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--proxy', type=str, default='fisher', choices=list(PROXIES.keys()))
    args = parser.parse_args()
    main(args)


# 用 SNIP 打分
# python evojson.py --arch_json trained_acc.json --proxy snip

# 用 SynFlow 打分
# python evojson.py --arch_json trained_acc.json --proxy synflow

# 用 GradNorm 打分
# python evojson.py --arch_json trained_acc.json --proxy gradnorm

# 用 fisher 打分
# python evojson.py --arch_json trained_acc.json --proxy fisher
