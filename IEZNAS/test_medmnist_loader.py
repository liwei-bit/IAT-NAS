import sys, torch
sys.path.append('.')                      # 保证能 import lib.*
from medmnist_imbalance import get_medmnist_loader

# === 1. PathMNIST, 保持原分布 ===
loader, c_in, n_cls = get_medmnist_loader(
    dataset_name='pathmnist',
    imb_factor=1.0,        # 保留原始失衡
    batch_size=32
)
x, y = next(iter(loader))
print('[PathMNIST]')
print(' input :', x.shape)              # (32, 3, 28, 28)
print(' labels:', y.shape, ', min=', y.min().item(), ', max=', y.max().item())
print(' c_in  :', c_in, ', n_cls:', n_cls)

# === 2. OrganAMNIST, 灰度图 ===
loader2, c_in2, n_cls2 = get_medmnist_loader(
    dataset_name='organamnist',
    imb_factor=1.0,
    batch_size=32
)
x2, y2 = next(iter(loader2))
print('[OrganAMNIST]')
print(' input :', x2.shape)             # (32, 1, 28, 28)
print(' labels:', y2.shape, ', min=', y2.min().item(), ', max=', y2.max().item())
print(' c_in  :', c_in2, ', n_cls:', n_cls2)
