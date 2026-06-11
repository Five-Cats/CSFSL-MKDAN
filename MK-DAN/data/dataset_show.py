import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('E:/CSFSL&MK-DAN/MK-DAN/data/Washington/dataset_day.npy')  # 替换为你的npy文件路径

# 计算每个时间步和小区上的平均车流量
average_flow = data.mean(axis=2)  # 形状变为 (时间步, 小区数)

# 创建热图
plt.figure(figsize=(12, 8))

# 使用seaborn的热图函数
heatmap = sns.heatmap(average_flow.T,  # 转置以使小区在y轴，时间在x轴
                     cmap='YlOrRd',    # 颜色映射
                     annot=False,      # 是否在每个单元格显示数值
                     cbar_kws={'label': 'Traffic Flow'})
cbar = heatmap.collections[0].colorbar
cbar.set_label('Traffic Flow', size=18)

# 设置横坐标（days）刻度间隔
xticks = np.arange(0, average_flow.shape[0], 2)  # 每2天显示一个刻度
plt.xticks(xticks, xticks, fontsize=16)  # 设置刻度位置和标签

# 设置纵坐标（areas）刻度显示为1-36
yticks = np.arange(0.5, average_flow.shape[1]+0.5, 1)  # 每个小区都显示
plt.yticks(yticks, np.arange(1, 37, 1), fontsize=16)  # 显示1-36所有标签

# 设置坐标轴标签
plt.xlabel('Days', fontsize=18)
plt.ylabel('Areas', fontsize=18)

# 将标题放在图下方，调整位置和边距
plt.title('Washington Traffic Distribution',
          fontsize=20,
          y=-0.15,      # 控制标题的垂直位置（负值表示在图下方）
          pad=15)       # 调整标题与图的间距

# 调整整体布局，防止标题被截断
plt.tight_layout()
# plt.subplots_adjust(bottom=0.2)  # 增加底部边距，确保标题可见

plt.show()