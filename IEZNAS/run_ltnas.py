import os
import argparse
import json
import torch
from lib.procedures.proxies import get_proxy_metric_fn
from Dataset_partitioning import get_dataloader  # 你写的划分逻辑
from lib.nas_201_api import NASBench201API
from lib.models.nas201_model import build_model_from_arch_str# 确保你引入的是 NAS201 接口
from tqdm import tqdm

def load_search_space(api_path, space_type='nasbench201'):
    if space_type == 'nasbench201':
        if not os.path.exists(api_path):
            raise FileNotFoundError(f"API path {api_path} not found.")
        api = NASBench201API(api_path)
        return api
    else:
        raise NotImplementedError(f"Search space {space_type} not supported yet.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Step 1: Load NAS search space
    api = load_search_space(args.api_path)

    # Step 2: Load dataset (不平衡 CIFAR10/100)
    train_loader = get_dataloader(args.dataset, imbalance_type=args.imb_type, imbalance_ratio=args.imb_ratio, batch_size=args.batch_size)

    # Step 3: Get your proxy score function
    proxy_score_fn = get_proxy_metric_fn(args.score)
    assert proxy_score_fn is not None, f"Proxy function for {args.score} not found."

    # Step 4: Score architectures
    results = {}
    print("[INFO] Starting architecture scoring...")
    for idx in tqdm(range(len(api))):
        arch_obj = api.query_by_index(idx)
        arch_str = arch_obj.arch_str  # 提取干净格式

        # ✅ 加入调试打印，检查是否为标准 6-op 格式
        print(f"idx: {idx}, arch_str: {arch_str}, ops count: {len(arch_str.split('+'))}")

        net = build_model_from_arch_str(arch_str, num_classes=100).to(device)
        net.eval()

        with torch.no_grad():
            score = proxy_score_fn(net, train_loader, args)
            results[arch_str] = float(score)

    # Step 5: Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.score}_scores.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Scores saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LTNAS Zero-Shot NAS with custom proxy score")
    parser.add_argument('--score', type=str, default='fisher_score', help='Proxy score name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--imb_type', type=str, default='longtail', help='Imbalance type')
    parser.add_argument('--imb_ratio', type=float, default=0.01, help='Imbalance ratio')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--api_path', type=str, default='searchspace/NAS-Bench-201-v1_1-096897.pth', help='Path to NAS-Bench-201 API')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    args = parser.parse_args()
    main(args)
