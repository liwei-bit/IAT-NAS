import os
import json
import sample_random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from lib.models.nas201_model import build_model_from_arch_str
from Dataset_partitioning import get_dataloader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn


def get_val_loader_cifar10(batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    val_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader


def train_and_evaluate(model, train_loader, val_loader, epochs=10, device='cuda'):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_per_class_acc = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs}, Step {batch_idx+1}/{len(train_loader)}, "
                      f"Avg Loss: {running_loss / (batch_idx % 50 + 1):.4f}")
                running_loss = 0.0

        scheduler.step()

        # 训练集准确率
        model.eval()
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()
        train_acc = 100. * correct_train / total_train
        print(f"Training Accuracy after epoch {epoch+1}: {train_acc:.2f}%")

        # 验证集准确率及每类准确率
        correct_val = 0
        total_val = 0
        num_classes = 10  # CIFAR-10类别数
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

                for i in range(len(targets)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        val_acc = 100. * correct_val / total_val
        print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.2f}%\n")

        if val_acc > best_acc:
            best_acc = val_acc
            best_per_class_acc = 100. * class_correct / (class_total + 1e-10)

    return best_acc, best_per_class_acc


def train_and_record(arch_list, label_prefix, train_loader, val_loader, epochs, device):
    results = []
    for i, arch_str in enumerate(arch_list):
        print(f"\n[{label_prefix} {i+1}/{len(arch_list)}] Training and evaluating architecture:\n{arch_str}")
        model = build_model_from_arch_str(arch_str, num_classes=10)
        acc, per_class_acc = train_and_evaluate(model, train_loader, val_loader, epochs=epochs, device=device)
        print(f"Accuracy of {label_prefix} architecture {i+1}: {acc:.2f}%")

        results.append({
            "arch_str": arch_str,
            "accuracy": acc,
            "per_class_accuracy": per_class_acc.tolist()
        })

        # 画尾部类别准确率柱状图（类别5-9）
        tail_acc = per_class_acc[5:10]
        plt.figure(figsize=(8, 4))
        plt.bar(range(5, 10), tail_acc, color='coral')
        plt.xlabel("Tail Classes (5-9)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Tail Classes Accuracy for {label_prefix} arch {i+1}")
        plt.tight_layout()
        plt.savefig(f"results/tail_acc_{label_prefix}_arch_{i+1}.png")
        plt.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['topk', 'random'], default='topk',
                        help="选择运行模式：topk（默认）或 random")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 参数配置
    topk = 5
    dataset = 'cifar10'
    imb_type = 'exp'
    imb_factor = 0.01
    batch_size = 128
    epochs = 10

    # 载入评分结果
    score_path = "./output/fisher_score_scores.json"
    with open(score_path, "r") as f:
        arch_scores = json.load(f)

    sorted_archs = sorted(arch_scores.items(), key=lambda x: x[1])
    topk_archs = [arch for arch, score in sorted_archs[:topk]]
    remaining_archs = [arch for arch, score in sorted_archs[topk:]]

    train_loader = get_dataloader(dataset=dataset, imbalance_type=imb_type, imbalance_ratio=imb_factor, batch_size=batch_size)
    val_loader = get_val_loader_cifar10(batch_size=batch_size)

    os.makedirs("results", exist_ok=True)
    os.makedirs("Experimental_figure", exist_ok=True)

    if args.mode == 'topk':
        print(f"Running training and evaluation on Top-{topk} architectures.")
        results = train_and_record(topk_archs, "topk", train_loader, val_loader, epochs, device)
        json_path = "results/topk_eval_results.json"
        plot_title = f"Top-{topk} Architectures on CIFAR-10-LT"
        plot_filename = "Experimental_figure/topk_eval_performance.png"
    else:
        print(f"Running training and evaluation on Random-{topk} architectures.")
        random5_archs = random.sample(remaining_archs, topk)
        results = train_and_record(random5_archs, "random", train_loader, val_loader, epochs, device)
        json_path = "results/random_eval_results.json"
        plot_title = f"Random-{topk} Architectures on CIFAR-10-LT"
        plot_filename = "Experimental_figure/random_eval_performance.png"

    # 保存json结果
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 画整体准确率柱状图
    accs = [r["accuracy"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(accs)), accs, color='skyblue')
    plt.xticks(range(len(accs)), [f"arch{i+1}" for i in range(len(accs))], rotation=30)
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()


if __name__ == "__main__":
    main()


#python run_topk_eval.py --mode random
