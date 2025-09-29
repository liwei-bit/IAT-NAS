#!/usr/bin/env python
import argparse, json, random, time, gc, os
from typing import List, Tuple
from collections import OrderedDict

import torch
from tqdm import tqdm

from Dataset_partitioning import get_dataloader
from medmnist_imbalance import get_medmnist_loader
from lib.models.nas201_model import build_model_from_arch_str
from lib.procedures.fisher_proxy_patch import get_proxy_metric_fn

# ----------------------------  Search space utils  ---------------------------- #
_OPS = ["none", "skip_connect", "conv1x1", "conv3x3", "avg_pool_3x3"]
_CELL_LEN = 6


def random_arch() -> str:
    return "+".join(random.choice(_OPS) for _ in range(_CELL_LEN))


def mutate_arch(arch: str, m_rate: float = 1.0) -> str:
    ops = arch.split("+")
    for i in range(_CELL_LEN):
        if random.random() < m_rate / _CELL_LEN:
            ops[i] = random.choice([o for o in _OPS if o != ops[i]])
    return "+".join(ops)


# ----------------------------  Architecture Evaluation  ---------------------------- #

def evaluate_arch(arch: str, proxy_fn, loader, device, args) -> float:
    model = build_model_from_arch_str(
        arch,
        in_channels=args.in_channels,
        num_classes=args.num_classes
    )
    model.to(device)
    score = proxy_fn(model, loader, args)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return score


# ----------------------------  Evolutionary Loop  ---------------------------- #

def evolutionary_search(args):
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------- 数据加载器 ----------------------
    if args.dataset in ['pathmnist', 'organamnist', 'dermamnist']:
        loader, in_ch, n_cls = get_medmnist_loader(
            dataset_name=args.dataset,
            imb_factor=1.0,
            batch_size=args.batch_size
        )
    else:
        loader, in_ch, n_cls = get_dataloader(
            dataset=args.dataset,
            imbalance_type=args.imb_type,
            imbalance_ratio=args.imb_ratio,
            batch_size=args.batch_size
        )

    args.in_channels = in_ch
    args.num_classes = n_cls
    proxy_fn = get_proxy_metric_fn(args.proxy)

    # 初始化种群
    population: List[str] = [random_arch() for _ in range(args.population)]
    scores = OrderedDict()

    print("[EA] Initial evaluation …")
    for arch in tqdm(population):
        scores[arch] = evaluate_arch(arch, proxy_fn, loader, device, args)

    μ = max(2, int(args.population * args.parent_frac))
    λ = args.population - μ

    best_arch, best_score = max(scores.items(), key=lambda kv: kv[1])
    history = [(0, best_score, best_arch)]

    for gen in range(1, args.generations + 1):
        parents = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:μ]
        parent_archs = [a for a, _ in parents]

        children: List[str] = []
        while len(children) < λ:
            p_arch = random.choice(parent_archs)
            c_arch = mutate_arch(p_arch, m_rate=args.mutation_rate)
            if c_arch not in scores:
                children.append(c_arch)

        for arch in tqdm(children, desc=f"Gen {gen}"):
            scores[arch] = evaluate_arch(arch, proxy_fn, loader, device, args)

        population = [a for a, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.population]]
        scores = OrderedDict((a, scores[a]) for a in population)

        best_arch, best_score = next(iter(scores.items()))
        history.append((gen, best_score, best_arch))
        print(f"[Gen {gen}] best_score={best_score:.4f}  arch={best_arch}")

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved final top-{len(scores)} scores to {args.save_json}")

    if args.history_json:
        with open(args.history_json, 'w') as f:
            json.dump(history, f, indent=2)


# ----------------------------  CLI Arguments  ---------------------------- #

def build_arg_parser():
    p = argparse.ArgumentParser("Evolutionary NAS with Tail-Aware Proxy")
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--parent_frac', type=float, default=0.3)
    p.add_argument('--mutation_rate', type=float, default=1.0)

    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'pathmnist', 'organamnist', 'dermamnist'])
    p.add_argument('--imb_type', default='exp')
    p.add_argument('--imb_ratio', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=128)

    p.add_argument('--proxy', default='tail_fisher')
    p.add_argument('--class_weight_exp', type=float, default=0.3)
    p.add_argument('--max_batches', type=int, default=300)

    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_json', default='results/evo_topk_pathmnist.json')
    p.add_argument('--history_json', default='results/evo_history_pathmnist.json')
    return p


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    evolutionary_search(args)



# 运行医学数据集搜索（保留真实不平衡）
# python Evo_fisher_patch.py --dataset cifar10 --proxy tail_fisher --population 10 --generations 2
