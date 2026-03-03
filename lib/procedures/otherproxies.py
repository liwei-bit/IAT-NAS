"""lib/procedures/{snip,jaccov,synflow,gradnorm,fisher}_proxy.py

轻量级 zero‑shot NAS 代理实现集合。
每个函数接口保持一致：
    score = proxy_fn(model, dataloader, args=None)
返回值越大代表架构越优。

•  snip_proxy      – SNIP saliency
•  jaccov_proxy    – Jacobian‐Covariance log‑det
•  synflow_proxy   – SynFlow signal flow
•  gradnorm_proxy  – 负梯度范数 (grad‑norm)
•  fisher_proxy    – 原版 Fisher proxy（无尾部加权）

所有计算均在 torch.no_grad() 或一次 backward 后完成，默认只看前 1‑2 个 batch，适合快速排名。
"""
from __future__ import annotations
import torch, math
from torch import nn
from typing import Any
import numpy as np

__all__ = [
    'snip_proxy', 'jaccov_proxy', 'synflow_proxy', 'gradnorm_proxy', 'fisher_proxy'
]

# ---------------- 通用：取前 N batch ---------------- #

def _take_batches(loader, n=1):
    for i, (x, y) in enumerate(loader):
        yield x, y
        if i + 1 >= n:
            break

# ---------------- 1. SNIP --------------------------- #

def snip_proxy(model: nn.Module, dataloader, args: Any = None):
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    crit = nn.CrossEntropyLoss()
    x, y = next(_take_batches(dataloader, 1))
    x = x.to(device)
    y = y.squeeze() if y.dim() > 1 else y
    y = y.to(device)

    out = model(x); loss = crit(out, y); loss.backward()
    score = sum((p.grad * p).abs().sum().item() for p in model.parameters() if p.grad is not None)
    return score / 1e6  # 数值缩放

# ---------------- 2. JacCov ------------------------- #

def jaccov_proxy(model: nn.Module, dataloader, args: Any = None):
    device = next(model.parameters()).device
    x, _ = next(_take_batches(dataloader, 1)); x = x.to(device).requires_grad_(True)
    out = model(x)
    C = out.size(1)
    jac_feats = []
    for c in range(C):
        model.zero_grad(set_to_none=True)
        out[:, c].sum().backward(retain_graph=True)
        jac_feats.append(x.grad.detach().clone().flatten(start_dim=1))
        x.grad.zero_()
    J = torch.stack(jac_feats, 0)      # [C, B, HW*C]
    cov = torch.cov(J.reshape(C, -1))  # [C, C]
    eig = torch.linalg.eigvals(cov).real.clamp_min(1e-8)
    return eig.log().sum().item() / C

# ---------------- 3. SynFlow ------------------------ #

def synflow_proxy(model: nn.Module, dataloader, args: Any = None):
    device = next(model.parameters()).device
    with torch.no_grad():
        for p in model.parameters():
            p.data.abs_()
    model.zero_grad(set_to_none=True)
    dummy = torch.ones([1] + list(dataloader.dataset[0][0].shape)).to(device)
    model(dummy).sum().backward()
    score = sum((p * p.grad).abs().sum().item() for p in model.parameters() if p.grad is not None)
    return score / 1e6

# ---------------- 4. GradNorm ----------------------- #

def gradnorm_proxy(model: nn.Module, dataloader, args: Any = None):
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    crit = nn.CrossEntropyLoss()
    total_gn = 0.
    for x, y in _take_batches(dataloader, n=2):
        x = x.to(device)
        y = y.squeeze() if y.dim() > 1 else y
        y = y.to(device)

        crit(model(x), y).backward()
        total_gn += sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    return -total_gn / 1e6  # 取负使"越大越好"

# ---------------- 5. Fisher (原版) ------------------ #

def fisher_proxy(model: nn.Module, dataloader, args: Any = None):
    """近似 trace(Fisher) 的快速实现：对前 1 batch 做一次 backward，累积(grad*activ)^2"""
    model.eval(); model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    crit = nn.CrossEntropyLoss()

    cache = {}
    hooks = []
    def reg_hook(m):
        def _fwd(_, __, out):
            if torch.is_tensor(out):
                out.retain_grad(); cache[id(out)] = out
        return _fwd
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            hooks.append(m.register_forward_hook(reg_hook(m)))

    x, y = next(_take_batches(dataloader,1))
    x = x.to(device)
    y = y.squeeze() if y.dim() > 1 else y
    y = y.to(device)

    loss = crit(model(x), y); loss.backward()

    score = 0.
    with torch.no_grad():
        for out in cache.values():
            if out.grad is None: continue
            g = out.grad
            score += (g * g).sum().item()
    for h in hooks: h.remove()
    return score / 1e6
