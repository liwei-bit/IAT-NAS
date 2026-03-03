import torch
import numpy as np
from collections import defaultdict

class FisherScoreEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.eval().to(device)
        self.device = device

    def extract_features(self, data_loader):
        features_by_class = defaultdict(list)
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                feats = self.model(imgs)  # shape: (B, D)
                for f, l in zip(feats, labels):
                    features_by_class[int(l.item())].append(f.cpu().numpy())
        return features_by_class

    def compute_class_statistics(self, features_by_class):
        class_means = {}
        intra_class_vars = {}
        class_counts = {}
        for cls, feats in features_by_class.items():
            feats_arr = np.stack(feats)
            mean_vec = np.mean(feats_arr, axis=0)
            var = np.mean(np.sum((feats_arr - mean_vec) ** 2, axis=1))
            class_means[cls] = mean_vec
            intra_class_vars[cls] = var
            class_counts[cls] = len(feats)
        return class_means, intra_class_vars, class_counts

    def compute_inter_class_distances(self, class_means):
        inter_class_dists = {}
        classes = list(class_means.keys())
        for cls in classes:
            dists = [np.linalg.norm(class_means[cls] - class_means[other])
                     for other in classes if other != cls]
            inter_class_dists[cls] = np.mean(dists)
        return inter_class_dists

    def compute_weighted_fisher_score(self, data_loader):
        features_by_class = self.extract_features(data_loader)
        class_means, intra_vars, class_counts = self.compute_class_statistics(features_by_class)
        inter_dists = self.compute_inter_class_distances(class_means)

        fisher_scores = {}
        total_samples = sum(class_counts.values())

        for cls in class_means:
            if intra_vars[cls] > 0:
                fisher_scores[cls] = np.log1p(inter_dists[cls] / (intra_vars[cls] + (1e-6)))
            else:
                fisher_scores[cls] = 0.0

        weighted_sum = sum(class_counts[cls] * fisher_scores[cls] for cls in fisher_scores)
        weighted_score = weighted_sum / total_samples
        return weighted_score, fisher_scores

def fisher_score_proxy(model, dataloader, args):
    evaluator = FisherScoreEvaluator(model, device='cuda')
    score, _ = evaluator.compute_weighted_fisher_score(dataloader)
    return score

def get_proxy_metric_fn(score_name):
    if score_name == 'fisher_score':
        return fisher_score_proxy
    else:
        raise ValueError(f"Unknown score: {score_name}")
