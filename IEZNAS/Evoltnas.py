import os, argparse, json, sample_random
import torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from lib.procedures.proxies import get_proxy_metric_fn
from Dataset_partitioning import get_dataloader
from lib.nas_201_api import NASBench201API
from lib.models.nas201_model import build_model_from_arch_str, convert_arch_str


# ──────────────────  进化算子  ──────────────────
def selection(population, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return [population[i] for i in idx]

def crossover(parents, n_child):
    offspring = []
    for _ in range(n_child):
        p1, p2 = random.sample(parents, 2)
        ops1, ops2 = p1.split('+'), p2.split('+')
        cp = random.randint(1, len(ops1) - 1)
        child = ops1[:cp] + ops2[cp:]
        offspring.append('+'.join(child))
    return offspring

def mutation(offspring, mu):
    mutated = []
    for arch in offspring:
        ops = arch.split('+')
        if random.random() < mu:
            idx = random.randint(0, len(ops) - 1)
            ops[idx] = random.choice(
                ['none', 'skip_connect', 'conv1x1', 'conv3x3', 'avg_pool_3x3']
            )
        mutated.append('+'.join(ops))
    return mutated


# ──────────────────  主流程  ──────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # 数据集 & API
    api = NASBench201API(args.api_path)
    train_loader = get_dataloader(
        args.dataset, args.imb_type, args.imb_ratio, batch_size=args.batch_size
    )
    proxy_fn = get_proxy_metric_fn(args.score)

    # 初始种群（简化格式）
    pop_size, num_gen, num_parents = 30, 20, 10
    raw_pop = [api.query_by_index(i).arch_str
               for i in random.sample(range(len(api)), pop_size)]
    population = [convert_arch_str(a) for a in raw_pop]

    best_score, best_arch = -float('inf'), None
    best_curve, avg_curve = [], []
    os.makedirs('Experimental_figure1', exist_ok=True)

    for g in range(num_gen):
        print(f'[INFO] Generation {g+1}/{num_gen}')
        scores = []

        # ── 评估：带进度条 ──
        for arch in tqdm(population, desc=f'Scoring Gen {g+1}', ncols=80):
            nc = 10 if args.dataset == 'cifar10' else 100
            net = build_model_from_arch_str(arch, num_classes=nc).to(device)
            net.eval()
            with torch.no_grad():
                s = proxy_fn(net, train_loader, args)
            scores.append(s)
            if s > best_score:
                best_score, best_arch = s, arch

        best_curve.append(best_score)
        avg_curve.append(np.mean(scores))

        # 直方图
        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=10, edgecolor='black', color='skyblue')
        plt.xlabel('Score'); plt.ylabel('Count')
        plt.title(f'Generation {g+1} Score Distribution')
        plt.tight_layout()
        plt.savefig(f'Experimental_figure1/gen_{g+1}_hist.png'); plt.close()

        # ── 进化 ──
        parents = selection(population, scores, num_parents)
        offspring = crossover(parents, pop_size - num_parents)

        # 动态变异率：0.4→0.05
        cur_mu = max(0.05, 0.4 * (1 - g / (num_gen - 1)))
        print(f'    └─ mutation rate = {cur_mu:.3f}')
        offspring = mutation(offspring, cur_mu)

        population = parents + offspring

    # 趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_gen+1), best_curve, 'r-', label='Best Score')
    plt.plot(range(1, num_gen+1), avg_curve,  'b-', label='Average Score')
    plt.xlabel('Generation'); plt.ylabel('Score')
    plt.title('Best & Average Scores per Generation')
    plt.legend(); plt.tight_layout()
    plt.savefig('Experimental_figure1/score_trend.png'); plt.close()

    # 保存最佳
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'{args.score}_best_arch.json'), 'w') as fp:
        json.dump({'best_arch': best_arch, 'best_score': float(best_score)}, fp, indent=2)
    print(f'[INFO] Best architecture saved to {args.output_dir}')

    # 最佳文本图
    plt.figure(figsize=(10, 6)); plt.axis('off')
    plt.text(0.5, 0.65, f'Best Architecture:\n{best_arch}', ha='center', va='center')
    plt.text(0.5, 0.25, f'Best Score: {best_score:.4f}', ha='center', va='center')
    plt.tight_layout(); plt.savefig('Experimental_figure1/best_arch.png'); plt.close()


# ──────────────────  CLI  ──────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser('LTNAS Evolution with Dynamic Mutation')
    p.add_argument('--score', default='fisher_score')
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--imb_type', default='exp')
    p.add_argument('--imb_ratio', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--api_path', default='searchspace/NAS-Bench-201-v1_1-096897.pth')
    p.add_argument('--output_dir', default='output')
    args = p.parse_args()
    main(args)
