from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from contextlib import nullcontext


# ---------------- Hooks: 只挂 Conv/Linear，降低噪声 ----------------
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

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(forward_hook(name)))
    return handles, cache


# ---------------- 工具：鲁棒聚合、标准化、权重 ----------------
def _trimmed_mean(v: torch.Tensor, trim_ratio: float) -> float:
    """去掉两端各 r 比例的截断均值；n<10 时退化为中位数。"""
    v = v.detach().float()
    n = v.numel()
    if n == 0:
        return 0.0
    if n < 10 or trim_ratio <= 0:
        return float(v.median())
    v, _ = torch.sort(v)
    k = int(trim_ratio * n)
    if k > 0:
        v = v[k:-k]
    return float(v.mean())


def _per_class_normalize(vals: torch.Tensor, mode: str) -> torch.Tensor:
    """
    对“该类的各层聚合值”做标准化：
      - 'minmax':  (x-min)/(max-min)
      - 'robust':  (x-med)/IQR  -> clip到[-3,3] -> 映射到[0,1]
      - 'none'  :  不做
    """
    vals = vals.detach().float()
    if vals.numel() == 0:
        return vals

    if mode == "minmax":
        lo, hi = vals.min(), vals.max()
        scale = (hi - lo).clamp_min(1e-12)
        return (vals - lo) / scale

    if mode == "robust":
        med = vals.median()
        q1 = vals.kthvalue(max(1, int(0.25 * (len(vals)))))[0]
        q3 = vals.kthvalue(max(1, int(0.75 * (len(vals)))))[0]
        iqr = (q3 - q1).abs().clamp_min(1e-12)
        z = (vals - med) / iqr
        z = torch.clamp(z, -3.0, 3.0)
        return (z + 3.0) / 6.0

    return vals  # 'none'


def _class_weights(counts: Dict[int, int], power: float) -> Dict[int, float]:
    """
    w_c ∝ (n_max / n_c)^power ；归一化到和为1。
    """
    if not counts:
        return {}
    n_max = max(counts.values())
    raw = {c: (n_max / (n + 1e-6)) ** power for c, n in counts.items()}
    s = sum(raw.values()) or 1.0
    return {c: w / s for c, w in raw.items()}


# ---------------- Tail-aware Fisher：激活×梯度 + 多级稳健化 ----------------
class TailAwareFisherOptimized:
    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cuda",
        amp: bool = True,
        eps: float = 1e-6,
        # Fisher 值后变换
        use_abs: bool = True,
        use_log1p: bool = True,
        # 每类内层聚合
        layer_agg: str = "trim_mean",          # {'trim_mean','median','mean'}
        trim_ratio: float = 0.10,              # 截断比例
        # 每类标准化
        per_class_norm: str = "robust",        # {'robust','minmax','none'}
        # 类别权重
        tail_power: float = 0.5,               # p：0.3~0.7 较稳
    ):
        self.use_amp = bool(amp)               # ← 统一 AMP 开关
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.criterion = nn.CrossEntropyLoss()
        self.eps = eps
        self.use_abs = use_abs
        self.use_log1p = use_log1p
        self.layer_agg = layer_agg
        self.trim_ratio = trim_ratio
        self.per_class_norm = per_class_norm
        self.tail_power = tail_power

    # ⚠️ 不要在这里使用任何装饰器（如 @torch.no_grad / @autocast）
    def _forward_backward(self, x: torch.Tensor, y: torch.Tensor,
                          cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp)
            if (self.use_amp and self.device.type == "cuda")
            else nullcontext()
        )

        with amp_ctx:
            logits = self.model(x)
            y = y.view(-1).long()
            loss = self.criterion(logits, y)

        self.model.zero_grad(set_to_none=True)
        loss.backward()  # 需要激活的梯度

        # 计算本批次每层的 per-sample Fisher 值（激活×梯度 内积）
        per_layer_values: Dict[str, torch.Tensor] = {}
        for name, z in cache.items():
            if z.grad is None:
                continue

            # 按样本做层内 Z-score（对齐通道/空间维）
            dims = tuple(range(1, z.dim()))
            mu = z.mean(dim=dims, keepdim=True)
            std = z.std(dim=dims, unbiased=False, keepdim=True)
            z_norm = (z - mu) / (std + self.eps)

            inner = (z_norm * z.grad).flatten(start_dim=1).sum(dim=1) / float(z_norm[0].numel())
            v = inner
            if self.use_abs:
                v = v.abs()
            if self.use_log1p:
                v = torch.log1p(v)

            per_layer_values[name] = v.detach()

        # 清理激活缓存里的 grad，避免累加
        for t in cache.values():
            t.grad = None
        cache.clear()

        return per_layer_values

    def compute(self,
                loader: torch.utils.data.DataLoader,
                *,
                max_batches: Optional[int] = None) -> float:
        handles, cache = _attach_activation_hooks(self.model)

        # 收集：每类、每层、一个值列表（来自多个样本/批次）
        cls_layer_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        class_count: Dict[int, int] = defaultdict(int)

        try:
            for b_idx, (xb, yb) in enumerate(loader):
                if (max_batches is not None) and (b_idx >= max_batches):
                    break
                xb, yb = xb.to(self.device), yb.to(self.device)

                per_layer = self._forward_backward(xb, yb, cache)  # {layer: [B]}

                # 记数，用于类别权重
                for c in yb.view(-1).tolist():
                    class_count[int(c)] += 1

                # 按样本分类追加
                for layer, vec in per_layer.items():
                    for i, c in enumerate(yb.view(-1).tolist()):
                        cls_layer_values[int(c)][layer].append(float(vec[i].item()))
        finally:
            for h in handles:
                h.remove()

        # 每类-每层：鲁棒聚合 -> 得到“该类的各层分数”
        cls_layer_agg: Dict[int, Dict[str, float]] = {}
        for c, layer_dict in cls_layer_values.items():
            cls_layer_agg[c] = {}
            for layer, lst in layer_dict.items():
                v = torch.tensor(lst, dtype=torch.float32)
                if self.layer_agg == "trim_mean":
                    agg = _trimmed_mean(v, self.trim_ratio)
                elif self.layer_agg == "median":
                    agg = float(v.median())
                else:
                    agg = float(v.mean())
                cls_layer_agg[c][layer] = float(agg)

        # 每类内做一次标准化，再对层取平均
        per_class_scores: Dict[int, float] = {}
        for c, layer_scores in cls_layer_agg.items():
            if not layer_scores:
                per_class_scores[c] = 0.0
                continue
            vals = torch.tensor(list(layer_scores.values()), dtype=torch.float32)
            vals = _per_class_normalize(vals, mode=self.per_class_norm)
            per_class_scores[c] = float(vals.mean())

        # 类别权重（长尾放大）
        ws = _class_weights(class_count, power=self.tail_power)

        # 汇总：负号使“更平滑(小Fisher)”得到更高分，便于与准确率正相关
        weighted = 0.0
        for c, s in per_class_scores.items():
            weighted += ws.get(c, 0.0) * s
        final_score = -float(weighted)

        # 极小值归零
        if abs(final_score) < 1e-12:
            final_score = 0.0
        return final_score


# ---------------- 免数据兜底（保持接口完整） ----------------
def _fisher_nodata(model: nn.Module,
                   device: torch.device,
                   amp: bool = True) -> float:
    """
    随机噪声 + KL 自一致，做个 fallback；不建议替代主方案，仅用于无数据时不中断。
    """
    model = model.to(device).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    denom = sum(p.numel() for p in params) or 1.0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp))
        if (amp and device.type == "cuda")
        else nullcontext()
    )

    xb = torch.randn(8, 3, 32, 32, device=device)
    with amp_ctx:
        logits = model(xb)
        probs = logits.softmax(dim=1).detach()
        loss = (probs * (probs.clamp_min(1e-8).log() - logits.log_softmax(dim=1))).sum(dim=1).mean()

    model.zero_grad(set_to_none=True)
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)
    score = 0.0
    for g in grads:
        if g is not None:
            score += (g.detach() ** 2).sum().item()
    return float(score / denom)


# ---------------- 对外工厂：给 nb201_rank_corr.py 动态加载 ----------------
def get_proxy_metric_fn(
    name: str = "tail_fisher_opt",
    *,
    # 可从 --proxy_cfg JSON 里覆写
    max_batches: Optional[int] = 3,
    amp: bool = True,
    tail_power: float = 0.5,        # 长尾放大力度
    trim_ratio: float = 0.10,       # 截断均值去两端10%
    per_class_norm: str = "robust", # {'robust','minmax','none'}
    use_log1p: bool = True,
    use_abs: bool = True,
    layer_agg: str = "trim_mean",   # {'trim_mean','median','mean'}
    eps: float = 1e-6,
    **kwargs
):
    """
    返回 metric_fn(model, device, data_loader) -> float
    - name="tail_fisher_opt": 需要 --use_real_data
    - name="fisher_nodata"   : 无数据兜底
    以上超参可在 JSON 配置里调整。
    """
    name = (name or "tail_fisher_opt").lower()

    if name == "tail_fisher_opt":
        def metric_fn(model, device, data_loader):
            device = torch.device(device)
            if data_loader is None:
                # 优雅回退：没给数据就用免数据，不建议长期使用
                return _fisher_nodata(model, device=device, amp=amp)

            eva = TailAwareFisherOptimized(
                model,
                device=device,
                amp=amp,
                eps=eps,
                use_abs=use_abs,
                use_log1p=use_log1p,
                layer_agg=layer_agg,
                trim_ratio=trim_ratio,
                per_class_norm=per_class_norm,
                tail_power=tail_power,
            )
            return float(eva.compute(data_loader, max_batches=max_batches))
        return metric_fn

    elif name == "fisher_nodata":
        def metric_fn(model, device, data_loader):
            return _fisher_nodata(model, device=torch.device(device), amp=amp)
        return metric_fn

    else:
        raise ValueError(f"Unknown proxy name: {name}")
