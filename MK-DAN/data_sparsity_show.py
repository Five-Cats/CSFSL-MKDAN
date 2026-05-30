import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False    # 负号显示正常

sparsity = [1.0, 0.5, 0.3, 0.1]
labels = ['100%', '50%', '30%', '10%']
cities = ["LosAngeles", "Chicago", "Washington", "NewYork"]

# ======================
# 数据
# ======================
models = {
    "Ours": {
        "MAE": {
            "LosAngeles": [0.48, 0.59, 0.68, 0.79],
            "Chicago": [2.28, 4.37, 5.08, 6.12],
            "Washington": [2.34, 2.85, 2.93, 3.14],
            "NewYork": [5.11, 9.72, 15.11, 19.08],
        }
    },
    "MTPB": {
        "MAE": {
            "LosAngeles": [0.36, 0.43, 0.69, 0.97],
            "Chicago": [2.78, 5.75, 7.24, 9.05],
            "Washington": [2.49, 3.11, 3.28, 4.05],
            "NewYork": [7.50, 13.20, 18.90, 29.30],
        }
    },
    "DASTNet": {
        "MAE": {
            "LosAngeles": [1.71, 2.35, 3.01, 3.99],
            "Chicago": [4.09, 6.84, 8.02, 11.20],
            "Washington": [3.89, 4.27, 5.55, 7.95],
            "NewYork": [7.24, 10.60, 15.20, 34.50],
        }
    },
    "MetaST": {
        "MAE": {
            "LosAngeles": [0.76, 0.98, 1.28, 1.72],
            "Chicago": [6.53, 9.84, 14.58, 20.50],
            "Washington": [5.42, 6.75, 8.55, 11.20],
            "NewYork": [8.38, 13.90, 22.60, 42.80],
        }
    }
}

# ======================
# 工具函数
# ======================
def normalize(data):
    """归一化相对第一个数据"""
    return np.array(data) / data[0]

# ======================
# 绘图
# ======================
for city in cities:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mae_deg = []

    for name, data in models.items():
        mae = data["MAE"][city]

        # ===== 折线图：归一化 MAE =====
        mae_norm = normalize(mae)
        style = {'linewidth': 3} if name == "Ours" else {'linestyle': '--'}
        axes[0].plot(sparsity, mae_norm, marker='o', label=name, **style)

        # ===== 退化率计算 =====
        deg = (mae[-1] - mae[0]) / mae[0]
        mae_deg.append(deg)

    # ===== 折线图设置（中文） =====
    axes[0].set_title(f"{city} - 归一化 MAE", fontsize=18)
    axes[0].set_xticks(sparsity)
    axes[0].set_xticklabels(labels, fontsize=16)
    axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=16)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # y轴保留一位小数
    axes[0].set_xlabel("训练数据比例", fontsize=16)
    axes[0].set_ylabel("相对 MAE", fontsize=16)
    axes[0].grid()
    axes[0].legend(fontsize=14)

    # ===== 退化率柱状图设置（中文） =====
    x = np.arange(len(models))
    width = 0.6
    axes[1].bar(x, mae_deg, width, color='skyblue')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models.keys(), fontsize=16)
    axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=16)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # y轴保留一位小数
    axes[1].set_xlabel("模型名称", fontsize=16)
    axes[1].set_ylabel("退化率", fontsize=16)
    axes[1].set_title(f"{city} - 性能退化 (100% → 10%)", fontsize=18)
    axes[1].grid(axis='y')

    plt.tight_layout()
    plt.show()