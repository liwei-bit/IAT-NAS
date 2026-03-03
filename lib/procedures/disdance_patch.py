import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# --------- util：统一获取倒数第二层特征 ---------
def extract_penultimate(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    在不同模型上尽量取到 logits 之前的全局特征向量。
    1. 若模型实现 forward_features，则直接调用；
    2. 若有属性 .features，则走该模块后再 GAP；
    3. 否则退化为 logits （最后再做 flatten）。
    返回形状：(B, D) 的 2‑D Tensor。
    """
    if hasattr(model, "forward_features"):
        feat = model.forward_features(x)               # e.g. ResNet/NAS201
    elif hasattr(model, "features"):
        feat = model.features(x)                       # e.g. MobileNetV2
        if feat.dim() == 4:                            # (B, C, H, W)
            feat = nn.functional.adaptive_avg_pool2d(feat, 1)
    else:                                              # 最差情况直接 logits
        feat = model(x)

    return feat.flatten(1)  # -> (B, D)
# --------------------------------------------------

class FisherScoreEvaluator:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.eval().to(device)
        self.device = device

    # 1. 抽特征：倒数第二层而非 logits
    def extract_features(self, data_loader):
        features_by_class = defaultdict(list)
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                feats = extract_penultimate(self.model, imgs)  # (B, D)
                for f, l in zip(feats, labels):
                    features_by_class[int(l)].append(f.cpu().numpy())
        return features_by_class

    # 2. 统计类均值、类内方差
    @staticmethod
    def compute_class_statistics(features_by_class):
        class_means, intra_vars, class_counts = {}, {}, {}
        for cls, feats in features_by_class.items():
            feats_arr = np.stack(feats)                # (Ni, D)
            mean_vec = feats_arr.mean(axis=0)
            var = np.mean(np.sum((feats_arr - mean_vec) ** 2, axis=1))  # 类内平方距离
            class_means[cls] = mean_vec
            intra_vars[cls] = var
            class_counts[cls] = len(feats)
        return class_means, intra_vars, class_counts

    # 3. 计算类间距离
    @staticmethod
    def compute_inter_class_distances(class_means):
        inter_dists = {}
        classes = list(class_means.keys())
        for cls in classes:
            dists = [np.linalg.norm(class_means[cls] - class_means[other])
                     for other in classes if other != cls]
            inter_dists[cls] = np.mean(dists)
        return inter_dists

    # 4. 整体 Fisher‑Score（带 √ 权重 & 方向翻转）
    def compute_weighted_fisher_score(self, data_loader):
        feats = self.extract_features(data_loader)
        class_means, intra_vars, class_counts = self.compute_class_statistics(feats)
        inter_dists = self.compute_inter_class_distances(class_means)

        if not class_counts:           # 避免空 dataloader
            return 0.0, {}

        max_n = max(class_counts.values())
        # √(N_max / N_i)   —— 既关注尾部，又避免权重过大
        class_weights = {cls: np.sqrt(max_n / (n + 1e-6)) for cls, n in class_counts.items()}
        total_w = sum(class_weights.values())
        class_weights = {cls: w / total_w for cls, w in class_weights.items()}

        fisher_scores = {}
        for cls in class_means:
            if intra_vars[cls] > 0:
                fisher_scores[cls] = np.log1p(inter_dists[cls] / (intra_vars[cls] + 1e-6))
            else:
                fisher_scores[cls] = 0.0

        # 加权求和后“乘 −1”翻转方向，使分数越大 → 准确率越高
        weighted_sum = sum(class_weights[cls] * fisher_scores[cls] for cls in fisher_scores)
        final_score = weighted_sum
        return final_score, fisher_scores


# === proxy 函数保持接口不变 ========================
def fisher_score_proxy(model, dataloader, args):
    evaluator = FisherScoreEvaluator(model, device='cuda')
    score, _ = evaluator.compute_weighted_fisher_score(dataloader)
    return score


def get_proxy_metric_fn(score_name):
    if score_name == 'fisher_score':
        return fisher_score_proxy
    raise ValueError(f"Unknown score: {score_name}")

# === Patch: 引入可调 α 和变换类型 ===
alpha = getattr(args, 'alpha', 0.5)
transform = getattr(args, 'transform', 'sqrt')

# === Patch: 计算 class_weights ===
class_weights = {c: (max_n / (n+1e-6)) ** alpha for c, n in class_counts.items()}

# === Patch: 替换 fisher_scores 计算 ===
# 原代码中某处:
# fisher_scores[cls] = np.log(inter_dists[cls] / (intra_vars[cls] + 1e-6))
# 改为:
ratio = inter_dists[cls] / (intra_vars[cls] + 1e-6)
if transform == "log":
    fisher_scores[cls] = np.log1p(ratio)
elif transform == "sqrt":
    fisher_scores[cls] = np.sqrt(ratio)
else:
    fisher_scores[cls] = ratio
