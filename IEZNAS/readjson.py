import json

def get_top_archs(json_path, top_k=5, save_path=None):
    # 1. 读json文件
    with open(json_path, 'r') as f:
        arch_dict = json.load(f)

    # 2. 按value排序（降序）
    sorted_archs = sorted(arch_dict.items(), key=lambda x: x[1], reverse=True)

    # 3. 取top_k条
    top_archs = sorted_archs[:top_k]

    # 4. 打印结果
    for i, (arch, score) in enumerate(top_archs, 1):
        print(f"Top {i}: Score={score:.6f}, Arch={arch}")

    # 5. 如果指定了保存路径，则保存为json
    if save_path:
        top_dict = {arch: score for arch, score in top_archs}
        with open(save_path, 'w') as f:
            json.dump(top_dict, f, indent=2)
        print(f"Top {top_k} architectures saved to {save_path}")

    return top_archs

if __name__ == "__main__":
    top_k = 3
    results = get_top_archs("output/fisher_score_scores.json", top_k, save_path="output/top_archs_from.json")
