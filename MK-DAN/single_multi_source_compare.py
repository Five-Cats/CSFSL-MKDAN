from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
def load_data(city_names):
    datasets = {}
    for city in city_names:
        data = np.load(f"data/{city}/dataset0.npy")  # (timesteps, regions, flow)
        data = data.squeeze(-1)  # (2161, 36)
        datasets[city] = data
    data = np.load(f"data/NYCBike/dataset1.npy")  # (timesteps, regions, flow)
    data = data.squeeze(-1)
    datasets['NYCBike2'] = data
    return datasets
cities = ['NYCBike', 'Chicago', 'Washington', 'LABike']
datasets = load_data(cities)


# distances = []
# for i in range(36):
#     src = datasets['NYCBike'][:, i]
#     tgt = datasets['LABike'][:, i]
#     dist = wasserstein_distance(src, tgt)
#     distances.append(dist)
# 拼接多个城市的数据在时间维度上
multi_source1 = np.concatenate([datasets['NYCBike'], datasets['NYCBike2']], axis=0)  # shape: (2161*3, 36)
multi_source2 = np.concatenate([datasets['NYCBike'], datasets['Washington'], datasets['Chicago']], axis=0)
distances_multi1 = []
distances_multi2 = []
for i in range(36):
    src1 = multi_source1[:, i]
    src2 = multi_source2[:, i]
    tgt = datasets['LABike'][:, i]
    dist1 = wasserstein_distance(src1, tgt)
    dist2 = wasserstein_distance(src2, tgt)
    distances_multi1.append(dist1)
    distances_multi2.append(dist2)

plt.figure(figsize=(10, 5))
plt.plot(distances_multi1, label='Single Source (NewYork City)')
plt.plot(distances_multi2, label='Multi Source (NewYork City+Washington+Chicago)')
plt.xlabel('Areas', fontsize=16)
plt.ylabel('Wasserstein Distance', fontsize=16)
# 将标题放在图下方，调整位置和边距
plt.title('Wasserstein Distance per Area (to LosAngeles)',
          fontsize=18,
          y=-0.15,      # 控制标题的垂直位置（负值表示在图下方）
          pad=-20)       # 调整标题与图的间距
plt.legend(fontsize=14, loc='upper right')
plt.grid(True)
plt.subplots_adjust(bottom=0.2)  # 增加底部边距
plt.show()
