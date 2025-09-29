import json
import random
from lib.nas_201_api import NASBench201API

def convert_nasbench201_arch(full_str):
    import re
    ops = re.findall(r'([^|+]+)~\d', full_str)
    if len(ops) != 6:
        raise ValueError(f"非法架构格式: {full_str}")
    return '+'.join(ops)


def sample_random_archs(api_path, num_samples=5, save_path='arch_list.json'):
    # 1. 加载 NAS-Bench-201 API
    api = NASBench201API(api_path)
    total = len(api)
    print(f"NAS-Bench-201 共 {total} 个架构")

    # 2. 随机采样索引
    sampled_indices = random.sample(range(total), num_samples)
    arch_list = []

    # 3. 获取对应架构字符串
    for idx in sampled_indices:
        full_str = api.arch(idx)  # 原始表示
        simple_str = convert_nasbench201_arch(full_str)
        arch_list.append(simple_str)
        print(f"[idx={idx}] {simple_str}")

    # 4. 保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(arch_list, f, indent=2)
    print(f"✅ 已保存到: {save_path}")


if __name__ == "__main__":
    # 示例用法
    sample_random_archs(
        api_path='searchspace/NAS-Bench-201-v1_1-096897.pth',
        num_samples=400,
        save_path='arch_random_400.json'
    )

# python sample_random.py