from __future__ import annotations
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, Tuple, Iterable, Optional

# -----------------------
# Hooks
# -----------------------
def _attach_activation_hooks(
    model: nn.Module,
    include_modules: Optional[Iterable[type]] = None,
):
    """
    Attach forward hooks to capture layer outputs (activations) and their grads.
    By default, only Conv2d & Linear are hooked to reduce noise/memory.
    """
    if include_modules is None:
        include_modules = (nn.Conv2d, nn.Linear)

    cache: Dict[str, torch.Tensor] = {}
    handles = []

    def forward_hook(name):
        def _hook(module, inp, out):
            if not isinstance(out, torch.Tensor):
                return out
            # 使非叶子张量也能在 backward 后拿到 grad
            out.retain_grad()
            cache[name] = out
            return out
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, include_modules):
            handles.append(module.register_forward_hook(forward_hook(name)))
    return handles, cache


# -----------------------
# Evaluator
# -----------------------
class TailAwareFisherEvaluator:
    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cuda",
        ce: Optional[nn.Module] = None,
        bce: Optional[nn.Module] = None,
        class_weight_exp: float = 0.3,   # p：尾部加权指数
        eps: float = 1e-6,               # Z-score 与权重中的数值稳定项
        use_abs_log: bool = True,        # 是否对 (inner product) 做 abs+log1p
        hook_bn: bool = False,           # 是否也钩 BatchNorm2d（默认否）
        minimize: bool = False,          # 如需给“最小化”优化器用，则取负
    ) -> None:
        self.device = device
        self.model = model.to(device).eval()
        self.ce = ce or nn.CrossEntropyLoss()
        self.bce = bce or nn.BCEWithLogitsLoss()
        self.p = float(class_weight_exp)
        self.eps = float(eps)
        self.use_abs_log = bool(use_abs_log)
        self.hook_bn = bool(hook_bn)
        self.minimize = bool(minimize)

    @torch.no_grad()
    def _tally_class_count(self, y: torch.Tensor, C: int | None = None) -> Dict[int, int]:
        """
        仅用标签统计批内各类出现次数。
        - 单标签: y shape [B] 或 [B,1]
        - 多标签: y shape [B, C]，统计每列阳性数
        """
        class_count: Dict[int, int] = defaultdict(int)

        if y.dim() == 2 and (C is None or y.size(1) == C):
            # 多标签：逐列统计阳性
            pos_per_class = y.sum(dim=0)  # [C]
            for cls in range(y.size(1)):
                class_count[cls] += int(pos_per_class[cls].item())
        else:
            # 单标签
            y1d = y.squeeze()
            vals, cnts = torch.unique(y1d, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                class_count[int(v)] += int(c)
        return class_count

    def _accumulate_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        fisher_sum: Dict[int, Dict[str, float]],
        fisher_cnt: Dict[int, Dict[str, int]],
    ):
        """
        在一个 batch 上：
        - 前向 + 计算任务损失（CE 或 BCE）
        - 反向得到每层激活的梯度
        - 计算样本级 Fisher 内积，并累计到 (类, 层)
        """
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        out = self.model(x)  # [B, C]（分类头）
        if out.dim() != 2:
            out = out.view(out.size(0), -1)
        B, C = out.shape

        # 判定多标签
        is_multilabel = (y.dim() == 2 and y.size(1) == C)

        # 损失（reduction='mean'）
        if is_multilabel:
            loss = self.bce(out, y.float())
            y_idx = None
        else:
            y_idx = (y.squeeze() if y.dim() > 1 else y).long()
            loss = self.ce(out, y_idx)

        # 反传以拿 z.grad
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # 遍历已缓存的各层激活（z）及其梯度（z.grad）
        for layer_name, z in cache.items():
            if z.grad is None:
                continue
            # -------- Z-score 按层标准化（沿 B×N_l）--------
            # z: [B, C, H, W] 或 [B, F]
            Bz = z.shape[0]
            z_flat = z.view(Bz, -1)
            mu = z_flat.mean()  # 标量（沿 B×N_l）
            sigma = z_flat.std()
            z_norm = (z - mu) / (sigma + self.eps)

            # -------- Fisher 内积（每样本）--------
            inner = (z_norm * z.grad).flatten(start_dim=1).mean(dim=1)  # [B], 1/N_l * sum_n
            if self.use_abs_log:
                val = torch.log1p(inner.abs())  # 稳定且非负
            else:
                val = inner  # 保留符号（一般不建议）

            # -------- 累计到“类-层”--------
            if is_multilabel:
                for idx in range(B):
                    # 一个样本可能有多个正类
                    pos = (y[idx] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
                    if not pos:
                        continue
                    v = float(val[idx].item())
                    for cls in pos:
                        fisher_sum[cls][layer_name] += v
                        fisher_cnt[cls][layer_name] += 1
            else:
                for idx in range(B):
                    cls = int(y_idx[idx].item())
                    fisher_sum[cls][layer_name] += float(val[idx].item())
                    fisher_cnt[cls][layer_name] += 1

        # 清理本批缓存中的梯度引用
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
        # 选择要挂钩的层
        include = (nn.Conv2d, nn.Linear) if not self.hook_bn else (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
        hooks, cache = _attach_activation_hooks(self.model, include_modules=include)

        fisher_sum: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        fisher_cnt: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        class_count: Dict[int, int] = defaultdict(int)

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            # 仅用 y 统计批内类频（单/多标签兼容）
            with torch.no_grad():
                C_guess = None
                if y.dim() == 2:
                    C_guess = y.size(1)
                cc = self._tally_class_count(y.to(self.device), C=C_guess)
                for k, v in cc.items():
                    class_count[k] += v

            # 真正累计 Fisher
            self._accumulate_batch(x, y, cache, fisher_sum, fisher_cnt)

        for h in hooks:
            h.remove()

        # 层内规约（mean 或 sum）
        fisher_map: Dict[int, Dict[str, float]] = {}
        for cls, layer_dict in fisher_sum.items():
            fisher_map[cls] = {}
            for layer_name, total in layer_dict.items():
                cnt = fisher_cnt[cls][layer_name]
                fisher_map[cls][layer_name] = (total / max(cnt, 1)) if reduce == "mean" else total

        # 类别权重（尾部加权）
        # 使用本次评估中观测到的类频 class_count 近似全数据计数
        if len(class_count) == 0:
            # 极端情况：数据集空
            return 0.0, fisher_map

        max_n = max(class_count.values())
        weights = {cls: (max_n / (n + self.eps)) ** self.p for cls, n in class_count.items()}
        Z = sum(weights.values()) or 1.0
        weights = {cls: w / Z for cls, w in weights.items()}

        # 加权汇总（每类对其所有被钩的层先取均值，再做类加权）
        weighted_score = 0.0
        for cls, layers in fisher_map.items():
            if len(layers) == 0:
                continue
            cls_mean = sum(layers.values()) / len(layers)
            weighted_score += weights.get(cls, 0.0) * cls_mean

        final_score = -weighted_score if self.minimize else weighted_score
        # 输出便于对齐日志（注意：若 minimize=True，这里会打印负值）
        print(f"Tail-aware Fisher Score: {final_score:.6e}")
        return float(final_score), fisher_map


# -----------------------
# Proxy entry
# -----------------------
def tail_aware_fisher_proxy(
    model: nn.Module,
    device: str | torch.device,
    dataloader: torch.utils.data.DataLoader,
    *,
    max_batches: int | None = None,
    class_weight_exp: float = 0.3,
    eps: float = 1e-6,
    use_abs_log: bool = True,
    hook_bn: bool = False,
    minimize: bool = False,
) -> float:
    evaluator = TailAwareFisherEvaluator(
        model,
        device=device,
        class_weight_exp=class_weight_exp,
        eps=eps,
        use_abs_log=use_abs_log,
        hook_bn=hook_bn,
        minimize=minimize,
    )
    score, _ = evaluator.compute_tail_fisher(dataloader, max_batches=max_batches, reduce="mean")
    return float(score)


def get_proxy_metric_fn(score_name: str = "tail_fisher", **kwargs):
    name = (score_name or "tail_fisher").lower()
    if name != "tail_fisher":
        raise ValueError(f"Unknown score: {score_name}")

    def metric_fn(model, device, data_loader):
        if data_loader is None:
            raise ValueError("tail_fisher 需要 --use_real_data 才能工作（需要一个 DataLoader）")
        return tail_aware_fisher_proxy(
            model,
            device,
            data_loader,
            max_batches=kwargs.get("max_batches", None),
            class_weight_exp=kwargs.get("class_weight_exp", 0.3),
            eps=kwargs.get("eps", 1e-6),
            use_abs_log=kwargs.get("use_abs_log", True),
            hook_bn=kwargs.get("hook_bn", False),
            minimize=kwargs.get("minimize", False),
        )

    return metric_fn
