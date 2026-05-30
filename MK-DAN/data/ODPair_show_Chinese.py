import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties

# ================== 中文字体（仅用于中文文本） ==================
ch_font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=30)
ch_font_title = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=18)
# =================================================================

# 读取 OD Pairs 数据
with open('E:/trans/DAN/MK-DAN/data/Washington/Washington_OD', 'r') as f:
    od_dict = ast.literal_eval(f.read())

# 强制生成 1–36 的节点
all_nodes = list(range(1, 37))
n_nodes = len(all_nodes)

# 创建流量矩阵
flow_matrix = np.zeros((n_nodes, n_nodes))
for (o, d), count in od_dict.items():
    i = o
    j = d
    flow_matrix[i, j] = count

# 绘制热力图
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(
    flow_matrix,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    norm=LogNorm(),
    xticklabels=all_nodes,
    yticklabels=all_nodes,
    vmin=1,
    cbar_kws={'label': '交通流量'}
)

# 颜色条设置（中文标签 + 数字刻度保持默认字体）
cbar = heatmap.collections[0].colorbar
cbar.set_label('交通流量', fontproperties=ch_font)
cbar.ax.tick_params(labelsize=18)

# 标题（中文，放在图下方）
# plt.title(
#     '洛杉矶 OD 交通流量热力图',
#     fontproperties=ch_font_title,
#     y=-0.15,
#     pad=40
# )

# 坐标轴标签（仅标签文字使用中文字体）
plt.xlabel('目的地区域编号', fontproperties=ch_font)
plt.ylabel('起始地区域编号', fontproperties=ch_font)

# 坐标刻度数字大小（不指定 fontproperties，数字字体不变）
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()
