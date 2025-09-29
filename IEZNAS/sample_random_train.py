import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models as tv_models
from Dataset_partitioning import get_dataloader
from lib.models.nas201_model import build_model_from_arch_str


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_from_name(name: str, num_classes=10):
    name = name.lower()
    if name == "resnet18":
        net = tv_models.resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        return net
    elif name == "resnet34":
        net = tv_models.resnet34(num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        return net
    elif name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=num_classes)
        net.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        return net
    else:
        raise ValueError(f"Unknown baseline model name: {name}")


def train_and_evaluate(model, train_loader, val_loader, epochs=10, device='cuda'):
    model = model.to(device)

    # 提取标签用于类别权重计算
    if isinstance(train_loader.dataset, Subset):
        targets_all = np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices]
    else:
        targets_all = np.array(train_loader.dataset.targets)

    class_counts = np.bincount(targets_all, minlength=10).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = np.clip(class_weights, 0, 50)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    num_classes = 10

    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()

        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc


def main(args):
    # 加载 [arch_str, proxy_score] 格式的 JSON
    with open(args.arch_json, 'r') as f:
        score_list = json.load(f)
        arch_list = [x[0] for x in score_list]

    seed_list = [int(s) for s in args.seeds.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    val_ds = datasets.CIFAR10('./datasets', train=False, download=True, transform=tf_test)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    mean_acc_dict = {}

    for idx, arch_str in enumerate(arch_list):
        print(f"\n==== [{idx+1}/{len(arch_list)}] Training arch: {arch_str} ====")
        acc_list = []

        for seed in seed_list:
            print(f"[Seed {seed}]")
            set_seed(seed)

            train_loader = get_dataloader(
                dataset='cifar10',
                imbalance_type=args.imb_type,
                imbalance_ratio=args.imb_ratio,
                batch_size=args.batch_size
            )

            try:
                model = build_model_from_arch_str(arch_str, num_classes=10)
            except Exception:
                model = build_model_from_name(arch_str, num_classes=10)

            acc = train_and_evaluate(
                model, train_loader, val_loader,
                epochs=args.epochs, device=device
            )
            acc_list.append(acc)

        mean_acc = np.mean(acc_list)
        print(f"=> {arch_str}: {mean_acc:.2f}%")

        mean_acc_dict[arch_str] = round(mean_acc, 4)

        # 每10个保存一次
        if (idx + 1) % 10 == 0:
            with open('results/trained_acc.json', 'w') as f:
                json.dump(mean_acc_dict, f, indent=2)

    os.makedirs("results", exist_ok=True)
    with open('results/sample_fisher_500_trained_acc.json', 'w') as f:
        json.dump(mean_acc_dict, f, indent=2)
    print("\n✅ 已保存到 sample_fisher_500_trained_acc.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_json', type=str, required=True,
                        help='JSON file of [[arch_str, proxy_score], ...]')
    parser.add_argument('--seeds', type=str, default='0',
                        help='comma-separated random seed(s), e.g. "0,1,2"')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--imb_type', type=str, default='exp')
    parser.add_argument('--imb_ratio', type=float, default=0.05)
    args = parser.parse_args()
    main(args)


# python sample_random_train.py --arch_json sampled_500_with_scores.json --seeds 0
