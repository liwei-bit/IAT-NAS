#!/usr/bin/env python
import argparse, json, random, time, gc, os
from typing import List, Tuple
from collections import OrderedDict
import numpy as np

import torch
from tqdm import tqdm

from Dataset_partitioning import get_dataloader
from medmnist_get_imbalance import get_medmnist_dataloaders
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

def normalize_score(s, all_scores):
    s_arr = np.array(list(all_scores.values()))
    return (s - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-8)

def softmax_tail_selection(score_dict, temperature=3.0, top_k=8):
    scores = np.array(list(score_dict.values()))
    archs = list(score_dict.keys())
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
    weights = np.exp(scores / temperature)
    probs = weights / weights.sum()
    selected = np.random.choice(archs, size=top_k, replace=False, p=probs)
    return selected

# ----------------------------  Dataset name normalization  ---------------------------- #
_MEDMNIST_ALIASES = {
    # path
    "path": "pathmnist", "pathmnist": "pathmnist",
    # organ A/C/S
    "organamnist": "organamnist", "organ_a": "organamnist", "organa": "organamnist",
    "organcmnist": "organcmnist", "organ_c": "organcmnist", "organc": "organcmnist",
    "organsmnist": "organsmnist", "organ_s": "organsmnist", "organs": "organsmnist",
    # chest
    "chest": "chestmnist", "chestmnist": "chestmnist",
    # pneumonia
    "pneumonia": "pneumoniamnist", "pneumoniamnist": "pneumoniamnist",
    # breast
    "breast": "breastmnist", "breastmnist": "breastmnist",
    # retina
    "retina": "retinamnist", "retinamnist": "retinamnist",
    # tissue
    "tissue": "tissuemnist", "tissuemnist": "tissuemnist",
    # derma
    "derma": "dermamnist", "dermamnist": "dermamnist",
    # blood
    "blood": "bloodmnist", "bloodmnist": "bloodmnist",
}

def normalize_dataset_name(name: str) -> str:
    n = name.lower().strip()
    if n in ("cifar10", "cifar-10"):
        return "cifar10"
    if n in ("cifar100", "cifar-100"):
        return "cifar100"
    return _MEDMNIST_ALIASES.get(n, n)  # 默认为原样（便于扩展）

def is_medmnist(name: str) -> bool:
    n = normalize_dataset_name(name)
    return n in _MEDMNIST_ALIASES.values()

# ----------------------------  Architecture Evaluation  ---------------------------- #
def evaluate_arch(arch: str, proxy_fn, loader, device, args) -> float:
    model = build_model_from_arch_str(
        arch,
        in_channels=args.in_channels,
        num_classes=args.num_classes
    )
    model.to(device)
    # 注意：proxy_fn(model, device, data_loader)
    score = proxy_fn(model, device, loader)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return score

# ----------------------------  Evolutionary Loop  ---------------------------- #
def evolutionary_search(args):
    device = torch.device(args.device)
    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 数据加载（支持 CIFAR 与 MedMNIST 多别名）
    ds = normalize_dataset_name(args.dataset)

    if is_medmnist(ds):
        train_loader, val_loader, test_loader, in_ch, n_cls = get_medmnist_dataloaders(
            dataset_name=ds,
            imb_type=args.imb_type if args.imb_ratio < 1.0 else None,  # 只有 <1.0 才做不平衡
            imb_factor=args.imb_ratio if args.imb_ratio < 1.0 else 1.0,
            batch_size=args.batch_size,
            seed=args.seed
        )
        loader = train_loader  # 代理指标只用训练集即可
    elif ds in ("cifar10", "cifar100"):
        loader, in_ch, n_cls = get_dataloader(
            dataset=ds,
            imbalance_type=args.imb_type,
            imbalance_ratio=args.imb_ratio,
            batch_size=args.batch_size
        )
        val_loader = None
        test_loader = None
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset} (normalized: {ds})")

    args.in_channels = in_ch
    args.num_classes = n_cls
    proxy_fn = get_proxy_metric_fn(args.proxy)

    # 初始化种群
    population: List[str] = [random_arch() for _ in range(args.population)]
    scores = OrderedDict()

    print("[IEZNAS] Initial evaluation …")
    for arch in tqdm(population):
        scores[arch] = evaluate_arch(arch, proxy_fn, loader, device, args)

    μ = max(2, int(args.population * args.parent_frac))
    μ = min(μ, len(scores))  # 兜底
    λ = args.population - μ

    base_mutation_rate = args.mutation_rate
    best_arch, best_score = max(scores.items(), key=lambda kv: kv[1])
    history = [(0, best_score, best_arch)]

    for gen in range(1, args.generations + 1):
        # === Variance-based global mutation control (机制3)
        score_values = np.array(list(scores.values()))
        std_score = np.std(score_values)

        if std_score < 1e-6:
            base_mutation_rate *= 1.2
        elif std_score > 1e-4:
            base_mutation_rate *= 0.9
        base_mutation_rate = np.clip(base_mutation_rate, 0.1, 2.0)  # 控制上下限

        # === Softmax Parent Sampling (机制1)
        parent_archs = softmax_tail_selection(scores, temperature=args.tau, top_k=μ)

        # === Evolution + Adaptive Mutation (机制2)
        children: List[str] = []
        while len(children) < λ:
            p_arch = random.choice(parent_archs)
            norm_score = normalize_score(scores[p_arch], scores)
            mut_rate = np.clip(base_mutation_rate * (1.0 - norm_score), 0.1, 0.5)  # 机制2 + clip
            c_arch = mutate_arch(p_arch, m_rate=mut_rate)
            if c_arch not in scores:
                score = evaluate_arch(c_arch, proxy_fn, loader, device, args)
                scores[c_arch] = score
                children.append(c_arch)

        # 更新种群
        population = [a for a, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.population]]
        scores = OrderedDict((a, scores[a]) for a in population)

        best_arch, best_score = next(iter(scores.items()))
        history.append((gen, best_score, best_arch))
        print(f"[Gen {gen}] std={std_score:.2e}  base_mut={base_mutation_rate:.3f}  best_score={best_score:.4f}  arch={best_arch}")

    save_dir = os.path.dirname(args.save_json) or "."
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved final top-{len(scores)} scores to {args.save_json}")

    if args.history_json:
        with open(args.history_json, 'w') as f:
            json.dump(history, f, indent=2)

# ----------------------------  CLI Arguments  ---------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser("IEZNAS: Imbalance-Evolution NAS")

    # Evolution params
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--parent_frac', type=float, default=0.3)
    p.add_argument('--mutation_rate', type=float, default=1.0)
    p.add_argument('--tau', type=float, default=3.0, help='Softmax temperature for parent selection')

    # Dataset (允许自由字符串，内部做别名映射)
    p.add_argument('--dataset', default=' Chestmnist', type=str,
                   help="支持: cifar10/cifar100，或 MedMNIST 别名：Path Organa organc organs Chest pneumonia breast retina tissue Derma Blood 等")

    # Imbalance
    p.add_argument('--imb_type', default='exp', help='仅对单标签训练集生效：exp/step/None')
    p.add_argument('--imb_ratio', type=float, default=1.0, help='<1.0 才会启用不平衡构造；MedMNIST 多标签不生效')

    # Loader
    p.add_argument('--batch_size', type=int, default=128)

    # Proxy
    p.add_argument('--proxy', default='tail_fisher')
    p.add_argument('--class_weight_exp', type=float, default=0.3)
    p.add_argument('--max_batches', type=int, default=300)

    # Misc
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_json', default='results/path_scores.json')
    p.add_argument('--history_json', default='results/path_history.json')
    return p

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    evolutionary_search(args)
