import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# ================== 中文字体（仅用于中文文本） ==================
ch_font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=24)
ch_font_title = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=18)
ch_font_legend = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=14)
# =================================================================

data = np.load('E:/trans/DAN/MK-DAN/data/LABike/LABike_poi.npy')

# 如果数据是浮点数，转换为整数
if data.dtype == float:
    data = data.astype(int)

n_neighborhoods, n_poi_types = data.shape

# POI 类型中文命名
poi_types = [f'POI_{i}' for i in range(n_poi_types)]
neighborhood_ids = [i + 1 for i in range(n_neighborhoods)]

# 颜色配置
colors = plt.cm.tab20(np.linspace(0, 1, n_poi_types))

# 绘制堆叠柱状图
plt.figure(figsize=(13, 6))
bottom = np.zeros(n_neighborhoods)

for i in range(n_poi_types):
    plt.bar(
        neighborhood_ids,
        data[:, i],
        bottom=bottom,
        label=poi_types[i],
        color=colors[i],
        width=0.8
    )
    bottom += data[:, i]

# 标题（中文，位于图下方）
# plt.title(
#     '洛杉矶兴趣点（POI）空间分布',
#     fontproperties=ch_font_title,
#     y=-0.15,
#     pad=-5
# )

# 坐标轴标签（仅文字使用中文字体）
plt.xlabel('小区编号', fontproperties=ch_font)
plt.ylabel('兴趣点数量', fontproperties=ch_font)

# 刻度（不指定 fontproperties，数字字体保持默认）
plt.xticks(
    neighborhood_ids,
    rotation=45,      # 或 60
    fontsize=16
)

# plt.xticks(neighborhood_ids, rotation=0, fontsize=16)
plt.yticks(fontsize=16)

# 图例（中文，单独指定字体）
plt.legend(
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    prop=ch_font_legend
)

plt.tight_layout()
plt.show()
