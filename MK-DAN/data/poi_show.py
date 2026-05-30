import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('E:/trans/DAN/MK-DAN/data/LABike/LABike_poi.npy')  # 替换为你的npy文件路径
# 如果数据是浮点数，转换为整数
if data.dtype == float:
    data = data.astype(int)  # 强制转换为整数
n_neighborhoods, n_poi_types = data.shape
poi_types = [f'POI_{i}' for i in range(n_poi_types)]
neighborhood_ids = [i+1 for i in range(n_neighborhoods)]  # 从1开始编号


# 颜色配置
colors = plt.cm.tab20(np.linspace(0, 1, n_poi_types))
# 绘制堆叠柱状图
plt.figure(figsize=(12, 6))
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

# 将标题放在图下方，调整位置和边距
plt.title('LosAngeles POI Distribution',
          fontsize=18,
          y=-0.15,      # 控制标题的垂直位置（负值表示在图下方）
          pad=-5)       # 调整标题与图的间距
plt.xlabel('Areas', fontsize=16)
plt.ylabel('POI', fontsize=16)
plt.xticks(neighborhood_ids, rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

