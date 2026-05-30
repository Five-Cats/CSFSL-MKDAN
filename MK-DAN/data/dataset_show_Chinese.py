import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# ================== 指定中文字体（仅用于中文文本） ==================
ch_font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=30)
ch_font_title = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=18)
# ===================================================================

data = np.load('E:/trans/DAN/MK-DAN/data/Washington/dataset_day.npy')

# 计算平均车流量
average_flow = data.mean(axis=2)

plt.figure(figsize=(12, 8))

heatmap = sns.heatmap(
    average_flow.T,
    cmap='YlOrRd',
    annot=False,
    cbar_kws={'label': '交通流量'}
)

# 颜色条中文标签（单独设置字体）
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.set_label('交通流量', fontproperties=ch_font)

# 横坐标刻度
xticks = np.arange(0, average_flow.shape[0], 2)
plt.xticks(xticks, xticks, fontsize=18)

# 纵坐标刻度
yticks = np.arange(0.5, average_flow.shape[1] + 0.5, 1)
plt.yticks(yticks, np.arange(1, 37, 1), fontsize=18)

# 坐标轴标签（仅标签使用中文字体）
plt.xlabel('天数', fontproperties=ch_font)
plt.ylabel('小区编号', fontproperties=ch_font)

# # 标题（中文，位于图下方）
# plt.title(
#     '洛杉矶自行车交通流量分布',
#     fontproperties=ch_font_title,
#     y=-0.15,
#     pad=20
# )

plt.tight_layout()
plt.show()
