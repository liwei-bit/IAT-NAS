import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- OPS 定义 ----------------------
OPS = {
    'none': lambda C_in, C_out, stride=1: ZeroOp(),
    'skip_connect': lambda C_in, C_out, stride=1:
        Identity() if stride == 1 and C_in == C_out else Conv1x1(C_in, C_out, stride),
    'conv1x1': lambda C_in, C_out, stride=1: Conv1x1(C_in, C_out, stride),
    'conv3x3': lambda C_in, C_out, stride=1: Conv3x3(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride=1: nn.AvgPool2d(3, stride=stride, padding=1),
}

# ---------------------- 基础模块 ----------------------
class Identity(nn.Module):
    def forward(self, x): return x

class ZeroOp(nn.Module):
    def forward(self, x): return x.mul(0.)

class Conv1x1(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )
    def forward(self, x): return self.op(x)

class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )
    def forward(self, x): return self.op(x)

# ---------------------- 辅助函数：字符串转换 ----------------------
def convert_arch_str(arch_str):
    """将原始 NAS-Bench-201 格式转换成简洁的 6 个 op 名"""
    parts = arch_str.split('+')
    ops = []
    for part in parts:
        inner_ops = part.strip('|').split('|')
        for op in inner_ops:
            op_name = op.split('~')[0]
            if op_name.startswith('nor_conv'):
                op_name = 'conv1x1' if '1x1' in op_name else 'conv3x3'
            elif op_name.startswith('skip_con'):
                op_name = 'skip_connect'
            elif op_name.startswith('avg_pool'):
                op_name = 'avg_pool_3x3'
            elif op_name == 'none':
                op_name = 'none'
            ops.append(op_name)
    return '+'.join(ops[:6])  # 取前 6 条边

# ---------------------- Cell 定义 ----------------------
class Cell(nn.Module):
    def __init__(self, arch_cell_str, C_in, C_out):
        super().__init__()
        arch_cell_str = convert_arch_str(arch_cell_str)
        ops = arch_cell_str.split('+')
        assert len(ops) == 6, f"arch_cell_str must have 6 ops, got {len(ops)}"

        self.edges = nn.ModuleList()
        channel_map = [C_in, C_out, C_out, C_out]
        edges_src = [0, 0, 1, 0, 1, 2]
        edges_dst = [1, 2, 2, 3, 3, 3]

        for i, op_name in enumerate(ops):
            src, dst = edges_src[i], edges_dst[i]
            op = OPS[op_name](channel_map[src], channel_map[dst])
            self.edges.append(op)

    def forward(self, x):
        node = {0: x}
        node[1] = self.edges[0](node[0]) + self.edges[1](node[0])
        node[2] = self.edges[2](node[1]) + self.edges[3](node[0])
        node[3] = self.edges[4](node[1]) + self.edges[5](node[2])
        return node[3]

# ---------------------- 主网络 ----------------------
class NASBench201Network(nn.Module):
    """
    三个相同 Cell 级联，stem 输出通道固定 16，保持与原 NAS-Bench-201 一致。
    """
    def __init__(self, arch_str, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )

        self.cell1 = Cell(arch_str, 16, 16)
        self.cell2 = Cell(arch_str, 16, 16)
        self.cell3 = Cell(arch_str, 16, 16)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.cell1(out)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.global_pooling(out).view(out.size(0), -1)
        return self.classifier(out)

# ---------------------- 构建接口 ----------------------
def build_model_from_arch_str(arch_str,
                              in_channels: int = 3,
                              num_classes: int = 10):
    """
    Args
        arch_str   : 原始 NAS-Bench-201 格式
        in_channels: 1 (灰度) 或 3 (彩色)
        num_classes: 分类类别数
    """
    return NASBench201Network(arch_str,
                              in_channels=in_channels,
                              num_classes=num_classes)
