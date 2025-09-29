import json
import random

# 1. 加载原始评分数据（格式：[ [arch_str, score], ... ]）
with open('tail_fisher_scores.json', 'r') as f:
    score_list = json.load(f)

# 2. 随机抽取 500 个架构 + 分数
random.seed(0)  # 固定种子以保证复现
sampled_data = random.sample(score_list, 500)

# 3. 保存为 JSON 文件（保留 arch_str 和 score）
with open('sampled_500_with_scores.json', 'w') as f:
    json.dump(sampled_data, f, indent=2)

print(f"已保存 500 个架构及其分数到 sampled_500_with_scores.json")


# python sample_random_proxy.py