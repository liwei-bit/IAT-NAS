#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEZNAS – NB201 evolutionary search with three switchable mechanisms
------------------------------------------------------------------
*Training-free* evolutionary search on NAS-Bench-201 using a zero-cost proxy
(e.g., Tail-Aware Fisher) as the objective. Three mechanisms (M1/M2/M3) can be
toggled via CLI flags for clean ablations in a single script.

Mechanisms
----------
M1 (parent selection): softmax-based parent sampling (temperature τ), with
    alternatives 'uniform' and 'topk'.
M2 (adaptive mutation): per-parent mutation rate scales inversely to its
    normalized score (lower score → more exploration).
M3 (variance control): adjust the *global* mutation rate based on population
    score variance (low variance → increase exploration; high variance → decrease).

Fairness knobs
--------------
- `--sample_budget` fixes the number of samples the proxy sees; changing
  `--batch_size` will not change proxy load (eff_batches = ceil(budget/bs)).
- Keep μ, λ, generations, seeds, and data the same across runs for fair ablations.

Project deps (from your repo)
-----------------------------
- lib.models.nas201_model.build_model_from_arch_str
- lib.procedures.fisher_proxy_patch.get_proxy_metric_fn
- medmnist_get_imbalance.get_medmnist_dataloaders
- isic2019_loader.get_isic2019_loader  ← 本文件新增依赖

Examples
--------
# Full (M1+M2+M3 on; softmax parents) on ISIC2019
python ieznas_switchable.py \
  --dataset isic2019 --isic_root ./datasets/ISIC_2019 --isic_size 32 \
  --proxy tail_fisher --class_weight_exp 0.3 \
  --batch_size 128 --sample_budget 38400 \
  --mu 64 --lmbda 64 --generations 20 --seeds 0,1,2 \
  --m1_softmax_parent 1 --parent_sel softmax --tau 3.0 \
  --m2_adaptive_mut 1 --mutation_rate 0.3 \
  --m3_var_ctrl 1 --var_low 1e-6 --var_high 1e-4

# Baseline (all OFF; uniform parents; fixed mutation)
python ieznas_switchable.py \
  --dataset isic2019 --isic_root ./datasets/ISIC_2019 --isic_size 32 \
  --proxy tail_fisher --class_weight_exp 0.3 \
  --batch_size 128 --sample_budget 38400 \
  --mu 64 --lmbda 32 --generations 10 --seeds 0,1,2 \
  --m1_softmax_parent 0 --parent_sel uniform \
  --m2_adaptive_mut 0 --mutation_rate 0.3 \
  --m3_var_ctrl 0
"""

import os, json, argparse, random, math, gc
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- your repo modules ----
from lib.models.nas201_model import build_model_from_arch_str
from lib.procedures.fisher_proxy_patch import get_proxy_metric_fn
from medmnist_get_imbalance import get_medmnist_dataloaders

# ---- ISIC2019 loader (新增) ----
from lib.dataop.ISIC_2019 import get_isic2019_loader


# ---------------------------- Search space (NB201) ---------------------------- #
_OPS = ["none", "skip_connect", "conv1x1", "conv3x3", "avg_pool_3x3"]
_CELL_LEN = 6  # NAS-Bench-201 cell length

def random_arch() -> str:
    return "+".join(random.choice(_OPS) for _ in range(_CELL_LEN))

def mutate_arch(arch: str, m_rate: float = 0.3) -> str:
    """Mutate each cell position with probability m_rate; pick a different op."""
    ops = arch.split("+")
    for i in range(_CELL_LEN):
        if random.random() < m_rate:
            cur = ops[i]
            cand = [o for o in _OPS if o != cur]
            ops[i] = random.choice(cand)
    return "+".join(ops)


# --------------------------- Utils: metrics & logging -------------------------- #
def shannon_entropy(pop_archs: List[str]) -> float:
    """Entropy of op distribution across 6 positions × 5 ops."""
    if not pop_archs:
        return 0.0
    counts = np.zeros((_CELL_LEN, len(_OPS)), dtype=np.float64)
    op2i = {o: i for i, o in enumerate(_OPS)}
    for a in pop_archs:
        toks = a.split("+")
        for pos, op in enumerate(toks):
            counts[pos, op2i[op]] += 1
    # per-position probabilities
    denom = np.clip(counts.sum(axis=1, keepdims=True), 1e-12, None)
    probs = counts / denom
    p = probs.reshape(-1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def normalize_score(s: float, all_scores: Dict[str, float]) -> float:
    vals = list(all_scores.values())
    if not vals:
        return 0.5
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return 0.5
    return float((s - vmin) / (vmax - vmin))


# ----------------------------- Parent selection ------------------------------ #
def softmax_parent_selection(scores: Dict[str, float], mu: int, tau: float = 3.0,
                             ensure_unique: bool = True) -> List[str]:
    """Sample μ parents with softmax over z-scored scores / τ."""
    if mu <= 0 or len(scores) == 0:
        return []
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=np.float64)
    vmean, vstd = float(vals.mean()), float(vals.std() + 1e-12)
    logits = (vals - vmean) / vstd
    logits = logits / max(tau, 1e-6)
    # stable softmax
    mx = float(np.max(logits))
    exps = np.exp(logits - mx)
    probs = exps / np.clip(exps.sum(), 1e-12, None)

    if not ensure_unique:
        idx = np.random.choice(len(keys), size=mu, replace=True, p=probs)
        return [keys[i] for i in idx]

    chosen = []
    avail = list(range(len(keys)))
    for _ in range(min(mu, len(keys))):
        cur_probs = probs[avail]
        cur_probs = cur_probs / np.clip(cur_probs.sum(), 1e-12, None)
        i_rel = int(np.random.choice(len(avail), p=cur_probs))
        i = avail.pop(i_rel)
        chosen.append(keys[i])
    return chosen


# -------------------------------- Proxy scoring -------------------------------- #
# ---- 替换整个函数定义 ----
def proxy_score_of_arch(arch: str, proxy_fn, loader: DataLoader, device: torch.device,
                        in_ch: int, n_cls: int) -> float:
    model = build_model_from_arch_str(arch, in_channels=in_ch, num_classes=n_cls).to(device)
    # 确保训练态 + 参数需要梯度
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    try:
        # 关键：显式开启梯度，避免外层 no_grad 影响
        with torch.enable_grad():
            score = float(proxy_fn(model, device, loader))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            del model; torch.cuda.empty_cache(); gc.collect()
            cpu = torch.device("cpu")
            model = build_model_from_arch_str(arch, in_channels=in_ch, num_classes=n_cls).to(cpu)
            model.train()
            for p in model.parameters():
                p.requires_grad_(True)
            with torch.enable_grad():
                score = float(proxy_fn(model, cpu, loader))
        else:
            raise
    finally:
        del model
        torch.cuda.empty_cache(); gc.collect()
    return score



# --------------------------------- Main search -------------------------------- #
def search_once(args, seed: int) -> Dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device(args.device)
    dataset = args.dataset.strip().lower()

    # dataloaders
    if dataset == 'isic2019':
        train_loader, val_loader, test_loader, in_ch, n_cls = get_isic2019_loader(
            data_root=args.isic_root,
            batch_size=args.batch_size,
            image_size=args.isic_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            augment=bool(args.isic_augment),
            seed=seed,
            num_workers=args.num_workers
        )
    else:
        train_loader, val_loader, test_loader, in_ch, n_cls = get_medmnist_dataloaders(
            dataset_name=dataset,
            imb_type=None,
            imb_factor=1.0,
            batch_size=args.batch_size,
            seed=seed
        )

    # proxy function with effective max_batches derived from sample_budget
    if args.sample_budget is not None and args.sample_budget > 0:
        eff_batches = max(1, math.ceil(args.sample_budget / args.batch_size))
    else:
        eff_batches = args.max_batches

    proxy_fn = get_proxy_metric_fn(
        args.proxy,
        max_batches=eff_batches,
        class_weight_exp=args.class_weight_exp,
    )

    mu, lmbda, G = args.mu, args.lmbda, args.generations
    pop: List[str] = []
    scores: Dict[str, float] = {}

    # warm start: random μ architectures
    while len(pop) < mu:
        a = random_arch()
        if a in scores:
            continue
        scores[a] = proxy_score_of_arch(a, proxy_fn, train_loader, device, in_ch, n_cls)
        pop.append(a)

    history = []
    base_mut = float(args.mutation_rate)
    base_mut = float(np.clip(base_mut, 0.0, 1.0))

    for gen in range(1, G + 1):
        # === M3: variance-based global mutation control ===
        if args.m3_var_ctrl:
            vals = np.array([scores[a] for a in pop], dtype=np.float64)
            std_score = float(vals.std())
            if std_score < args.var_low:
                base_mut *= args.var_up
            elif std_score > args.var_high:
                base_mut *= args.var_down
            base_mut = float(np.clip(base_mut, args.mut_min, args.mut_max))
        else:
            std_score = float('nan')

        # === M1: parent selection ===
        if args.parent_sel == 'softmax' and args.m1_softmax_parent:
            parents = softmax_parent_selection({a: scores[a] for a in pop}, mu=args.mu, tau=args.tau, ensure_unique=True)
        elif args.parent_sel == 'topk':
            k = args.topk_mu if args.topk_mu > 0 else args.mu
            parents = [a for a, _ in sorted(((a, scores[a]) for a in pop), key=lambda kv: kv[1], reverse=True)[:k]]
        else:
            # uniform
            parents = random.sample(pop, k=min(args.mu, len(pop)))

        # === produce offspring (λ) ===
        children: List[str] = []
        tries = 0
        while len(children) < lmbda and tries < lmbda * 20:
            tries += 1
            p_arch = random.choice(parents)
            if args.m2_adaptive_mut:
                norm = normalize_score(scores[p_arch], {a: scores[a] for a in pop})
                mut_rate = float(np.clip(base_mut * (1.0 - norm), args.mut_min, args.mut_max))
            else:
                mut_rate = base_mut
            c_arch = mutate_arch(p_arch, m_rate=mut_rate)
            if c_arch in scores:
                continue
            sc = proxy_score_of_arch(c_arch, proxy_fn, train_loader, device, in_ch, n_cls)
            scores[c_arch] = sc
            children.append(c_arch)

        # === environmental selection: (μ+λ) → μ (elitist) ===
        pool = pop + children
        pool = list({a: None for a in pool}.keys())  # deduplicate keep order
        pool_sorted = sorted(pool, key=lambda a: scores[a], reverse=True)
        pop = pool_sorted[:mu]

        # record metrics
        best = float(scores[pop[0]])
        vals = np.array([scores[a] for a in pop], dtype=np.float64)
        mean, std = float(vals.mean()), float(vals.std())
        uniq = len(set(pop))
        ent = shannon_entropy(pop)
        history.append({
            'gen': gen,
            'best': best,
            'mean': mean,
            'std': std,
            'unique': uniq,
            'entropy': ent,
            'base_mut': float(base_mut),
        })

    topk = [(a, float(scores[a])) for a in sorted(pop, key=lambda a: scores[a], reverse=True)[:args.report_topk]]
    return {
        'seed': seed,
        'dataset': dataset,
        'proxy': args.proxy,
        'batch_size': args.batch_size,
        'sample_budget': args.sample_budget,
        'mu': mu,
        'lmbda': lmbda,
        'generations': G,
        'm1_softmax_parent': int(args.m1_softmax_parent),
        'parent_sel': args.parent_sel,
        'tau': args.tau,
        'm2_adaptive_mut': int(args.m2_adaptive_mut),
        'm3_var_ctrl': int(args.m3_var_ctrl),
        'var_low': args.var_low,
        'var_high': args.var_high,
        'var_up': args.var_up,
        'var_down': args.var_down,
        'mut_min': args.mut_min,
        'mut_max': args.mut_max,
        'mutation_rate_init': args.mutation_rate,
        'history': history,
        'topk': topk,
    }


# ----------------------------------- I/O -------------------------------------- #
def save_run(out_dir: str, tag: str, result: Dict):
    os.makedirs(out_dir, exist_ok=True)
    # per-seed json
    js_path = os.path.join(out_dir, f"{tag}.json")
    with open(js_path, 'w') as f:
        json.dump(result, f, indent=2)
    # per-seed csv (history)
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    with open(csv_path, 'w') as f:
        f.write("gen,best,mean,std,unique,entropy,base_mut\n")
        for h in result['history']:
            f.write(f"{h['gen']},{h['best']:.6f},{h['mean']:.6f},{h['std']:.6f},{h['unique']},{h['entropy']:.6f},{h['base_mut']:.4f}\n")


# ----------------------------------- CLI -------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser("IEZNAS – NB201 evolutionary (switchable mechanisms)")

    # dataset / loader / proxy
    p.add_argument('--dataset', type=str, default='organcmnist',
                   help='MedMNIST subset or "isic2019"')
    p.add_argument('--proxy', type=str, default='tail_fisher')
    p.add_argument('--class_weight_exp', type=float, default=0.3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--max_batches', type=int, default=300, help='used only if sample_budget is None')
    p.add_argument('--sample_budget', type=int, default=38400, help='fix proxy sample count; eff_batches=ceil(budget/bs)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--num_workers', type=int, default=4)

    # ISIC2019 专用
    p.add_argument('--isic_root', type=str, default='datasets/ISIC_2019')
    p.add_argument('--isic_size', type=int, default=32, help='resize to (S,S) for NB201')
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--isic_augment', type=int, default=0, help='1=RandHF on')

    # evolution
    p.add_argument('--mu', type=int, default=64)
    p.add_argument('--lmbda', type=int, default=64)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--seeds', type=str, default='0', help='comma-separated, e.g. 0,1,2')

    # mechanisms (switches)
    p.add_argument('--m1_softmax_parent', type=int, default=1, help='1=ON, 0=OFF (OFF+softmax => fallback to uniform)')
    p.add_argument('--parent_sel', type=str, default='softmax', choices=['softmax','uniform','topk'])
    p.add_argument('--tau', type=float, default=3.0, help='temperature for softmax parent selection')
    p.add_argument('--topk_mu', type=int, default=0, help='when parent_sel=topk, use top-k (=μ if 0)')

    p.add_argument('--m2_adaptive_mut', type=int, default=1, help='1=ON, 0=OFF')
    p.add_argument('--mutation_rate', type=float, default=0.3, help='initial/base mutation rate (0..1)')
    p.add_argument('--mut_min', type=float, default=0.1)
    p.add_argument('--mut_max', type=float, default=0.5)

    p.add_argument('--m3_var_ctrl', type=int, default=1, help='1=ON, 0=OFF')
    p.add_argument('--var_low', type=float, default=1e-6)
    p.add_argument('--var_high', type=float, default=1e-4)
    p.add_argument('--var_up', type=float, default=1.2, help='multiplier when std < var_low')
    p.add_argument('--var_down', type=float, default=0.9, help='multiplier when std > var_high')

    # outputs
    p.add_argument('--out_dir', type=str, default='results/ieznas')
    p.add_argument('--report_topk', type=int, default=10)

    return p


def main(args):
    seeds = [int(s.strip()) for s in args.seeds.split(',') if str(s).strip() != '']

    tag_base = (
        f"{args.dataset}_proxy-{args.proxy}_bs{args.batch_size}_sb{args.sample_budget}_"
        f"mu{args.mu}_la{args.lmbda}_G{args.generations}_"
        f"M1{int(args.m1_softmax_parent)}-{args.parent_sel}_tau{args.tau}_"
        f"M2{int(args.m2_adaptive_mut)}_mut{args.mutation_rate}_"
        f"M3{int(args.m3_var_ctrl)}_vL{args.var_low}_vH{args.var_high}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    for seed in seeds:
        run = search_once(args, seed=seed)
        tag = tag_base + f"_seed{seed}"
        save_run(args.out_dir, tag, run)
        # brief stdout
        h = run['history']
        best_seq = [x['best'] for x in h]
        print(f"[SEED {seed}] best-by-gen (first 5): {np.array(best_seq)[:5]} ... -> final {best_seq[-1]:.6f}")
        print(f"[SEED {seed}] TOP{args.report_topk}: {run['topk'][:3]} ...")


if __name__ == '__main__':
    main(build_arg_parser().parse_args())
