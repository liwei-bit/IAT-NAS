import json
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, rankdata

# === 第一步：统一架构字符串格式 ===
def normalize_arch_str(s):
    return s.replace(" ", "").replace("\n", "")  # 可扩展：.lower() 或正则等

# === 第二步：读取两个 JSON 文件 ===

with open("trained_acc.json", "r") as f:
    raw_acc_dict = json.load(f)

with open("arch_scores_fisher.json", "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    proxy_dict = data
elif isinstance(data, list):
    proxy_dict = {item[0]: item[1] for item in data}
else:
    raise ValueError("未知的 proxy JSON 格式")

# 清洗 acc_dict 和 proxy_dict 的 key
acc_dict = {normalize_arch_str(k): v for k, v in raw_acc_dict.items()}
proxy_dict = {normalize_arch_str(k): v for k, v in proxy_dict.items()}  # ← FIXED

# === 第三步：对齐两个文件中的架构 ===

common_archs = sorted(list(set(acc_dict.keys()) & set(proxy_dict.keys())))
if len(common_archs) == 0:
    raise ValueError("两个 JSON 中没有重合的架构字符串！")

acc_list = [acc_dict[arch] for arch in common_archs]
proxy_list = [proxy_dict[arch] for arch in common_archs]

# === 第四步：计算 Kendall's Tau ===

tau, p_value = kendalltau(proxy_list, acc_list)
print(f"Kendall's Tau: {tau:.4f}, p-value: {p_value:.4e}")

# === 第五步：将分数转换为排名，准备绘图 ===

rank_acc = rankdata(acc_list)
rank_proxy = rankdata(proxy_list)

# === 第六步：绘制排名散点图 ===

plt.figure(figsize=(6, 6))
plt.scatter(rank_proxy, rank_acc, color='#5086C4', label='Architectures')
plt.plot([0, len(rank_acc)], [0, len(rank_acc)], 'r--', label='Perfect Ranking')
plt.plot([], [], ' ', label=f"Kendall's Tau = {tau:.4f}")

plt.xlabel("Proxy Score Rank(Fisher)", fontsize=12)
plt.ylabel("Accuracy Rank", fontsize=12)
plt.legend()


plt.grid(True)
plt.tight_layout()
plt.savefig("kendall_rank_fisher.png")
plt.show()

# python kendallTau.py