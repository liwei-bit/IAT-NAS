from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, Tuple


def _attach_activation_hooks(model: nn.Module):
    cache: Dict[str, torch.Tensor] = {}
    handles = []

    def forward_hook(name):
        def _hook(module, inp, out):
            if not isinstance(out, torch.Tensor):
                return out
            out.requires_grad_(True)
            out.retain_grad()
            cache[name] = out
            return out

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.Identity)):
            handles.append(module.register_forward_hook(forward_hook(name)))
    return handles, cache


class TailAwareFisherEvaluator:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        device: str | torch.device = "cuda",
        scale: float = 1e6,
    ) -> None:
        self.device = device
        self.model = model.eval().to(device)
        # 原逻辑：默认 CE（单标签）。同时准备 BCE 以便多标签时使用。
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.scale = scale  # 放大系数（保留占位）

    def _accumulate_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        fisher_sum: Dict[int, Dict[str, float]],
        fisher_cnt: Dict[int, Dict[str, int]],
    ):
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)  # [B, C] 或 [B, ...]
        if out.dim() != 2:
            out = out.view(out.size(0), -1)
        B, C = out.shape

        # —— 多标签判定：y 形状为 [B, C] 且与输出通道一致 —— #
        is_multilabel = (y.dim() == 2 and y.size(1) == C)

        # === 保持“整批一次反传”的原逻辑；仅按任务切换损失 ===
        if is_multilabel:
            # 多标签：y ∈ {0,1}^{B×C}，用 BCEWithLogitsLoss（reduction='mean'）
            y_float = y.float()
            loss = self.bce(out, y_float)
            y_index_like = None  # 仅为接口兼容
        else:
            # 单标签：完全保持原逻辑（CE + squeeze）
            y_index_like = y.squeeze() if y.dim() > 1 else y
            loss = self.ce(out, y_index_like)

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # === 与原逻辑一致：使用 z.grad 构造逐样本的 inner product ===
        for layer_name, z in cache.items():
            if z.grad is None:
                continue

            # ① 激活标准化，层间尺度对齐
            z_norm = (z - z.mean()) / (z.std() + 1e-6)

            # ② 计算每个样本的 inner-product（shape=[B]）
            inner_prod = (z_norm * z.grad).flatten(start_dim=1).sum(dim=1) / z_norm[0].numel()

            # ③ 取绝对值 + log1p，避免极值 & 负数
            fisher_val = torch.log1p(inner_prod.abs()).detach()  # [B]

            # === 将样本贡献累计到类别 ===
            if is_multilabel:
                # 多标签：一个样本可能同时属于多个正类
                for idx in range(B):
                    pos = (y[idx] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
                    if not pos:
                        continue
                    v = fisher_val[idx].item()
                    for cls in pos:
                        fisher_sum[cls][layer_name] += v
                        fisher_cnt[cls][layer_name] += 1
            else:
                # 单标签：完全保持原版做法
                for idx in range(B):
                    cls = int(y_index_like[idx].item())
                    fisher_sum[cls][layer_name] += fisher_val[idx].item()
                    fisher_cnt[cls][layer_name] += 1

        # 清理缓存
        for z in cache.values():
            z.grad = None
        cache.clear()

    def compute_tail_fisher(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        max_batches: int | None = None,
        reduce: str = "mean",
    ) -> Tuple[float, Dict[int, Dict[str, float]]]:
        hooks, cache = _attach_activation_hooks(self.model)

        fisher_sum: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        fisher_cnt: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        class_count: Dict[int, int] = defaultdict(int)

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            # 统计类别频次（用于尾部加权）：多标签/单标签分别处理
            with torch.no_grad():
                x_dev = x.to(self.device)
                out_dummy = self.model(x_dev)
                if out_dummy.dim() != 2:
                    out_dummy = out_dummy.view(out_dummy.size(0), -1)
                C = out_dummy.size(1)

                if y.dim() == 2 and y.size(1) == C:
                    # 多标签：按列统计阳性数
                    y_dev = y.to(self.device)
                    pos_per_class = y_dev.sum(dim=0)  # [C]
                    for cls in range(C):
                        class_count[cls] += int(pos_per_class[cls].item())
                else:
                    # 单标签：与原逻辑一致
                    y_dev = (y.to(self.device).squeeze() if y.dim() > 1 else y.to(self.device))
                    for c in y_dev.unique().tolist():
                        class_count[int(c)] += (y_dev == c).sum().item()

            # 真正累积 Fisher（带梯度）
            self._accumulate_batch(x, y, cache, fisher_sum, fisher_cnt)

        for h in hooks:
            h.remove()

        # 层内规约
        fisher_map: Dict[int, Dict[str, float]] = {}
        for cls, layer_dict in fisher_sum.items():
            fisher_map[cls] = {}
            for layer_name, total in layer_dict.items():
                cnt = fisher_cnt[cls][layer_name]
                fisher_map[cls][layer_name] = total / max(cnt, 1) if reduce == "mean" else total

        # ---- 类别权重（尾部加权） p=0.2（与原思路一致，可按需改回 0.3）----
        max_n = max(class_count.values()) if class_count else 1
        p = 0.2
        class_weights = {cls: (max_n / (n + 1e-6)) ** p for cls, n in class_count.items()}
        total_w = sum(class_weights.values()) or 1.0
        class_weights = {cls: w / total_w for cls, w in class_weights.items()}

        # ---- 加权汇总 & 翻转方向（越大越好 -> 取负做越小越好）----
        weighted_score = 0.0
        for cls, layers in fisher_map.items():
            cls_mean = sum(layers.values()) / max(len(layers), 1)
            weighted_score += class_weights.get(cls, 0.0) * cls_mean

        final_score = -weighted_score
        if abs(final_score) < 1e-8:
            final_score = 0.0

        print(f"Tail-aware Fisher Score: {final_score:.6e}")
        return final_score, fisher_map


# === Proxy 接口保持一致 ===
def tail_aware_fisher_proxy(
    model: nn.Module,
    device: str | torch.device,
    dataloader: torch.utils.data.DataLoader,
    *,
    max_batches: int | None = None,
) -> float:
    evaluator = TailAwareFisherEvaluator(model, device=device)
    score, _ = evaluator.compute_tail_fisher(dataloader, max_batches=max_batches, reduce="mean")
    return float(score)


def get_proxy_metric_fn(score_name: str = "tail_fisher", **kwargs):
    name = (score_name or "tail_fisher").lower()
    if name != "tail_fisher":
        raise ValueError(f"Unknown score: {score_name}")

    max_batches = kwargs.get("max_batches", None)

    def metric_fn(model, device, data_loader):
        if data_loader is None:
            raise ValueError("tail_fisher 需要 --use_real_data 才能工作（需要一个 DataLoader）")
        return tail_aware_fisher_proxy(model, device, data_loader, max_batches=max_batches)

    return metric_fn
