import ast
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 对数色标（适用于流量差异大的情况）


# 读取OD Pairs数据
with open('E:/trans/DAN/MK-DAN/data/NYCBike/NYCBike_OD', 'r') as f:
    od_dict = ast.literal_eval(f.read())  # 将字符串转为字典

# 提取所有OD对编号
origins = set(k[0] for k in od_dict.keys())
destinations = set(k[1] for k in od_dict.keys())
# 强制生成1-36的节点（即使某些节点无流量）
all_nodes = list(range(1, 37))  # 1-36
n_nodes = len(all_nodes)
# 创建流量矩阵（初始值为0，而非NaN）
flow_matrix = np.zeros((n_nodes, n_nodes))
for (o, d), count in od_dict.items():
    i = o  # 原始数据中的0对应1，1对应2，以此类推
    j = d
    flow_matrix[i, j] = count

# 绘制热力图
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(
    flow_matrix,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    norm=LogNorm(),  # 对数色标（可选）
    xticklabels=all_nodes,
    yticklabels=all_nodes,
    vmin=1,
    cbar_kws={'label': 'Traffic Flow'}
)
cbar = heatmap.collections[0].colorbar
cbar.set_label('Traffic Flow', size=16)
# 将标题放在图下方，调整位置和边距
plt.title('LosAngeles OD Flow Heatmap',
          fontsize=18,
          y=-0.15,      # 控制标题的垂直位置（负值表示在图下方）
          pad=40)       # 调整标题与图的间距
plt.xlabel('Destination Area ID', fontsize=16)
plt.ylabel('Origin Area ID', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # 增加底部边距，确保标题可见
plt.show()