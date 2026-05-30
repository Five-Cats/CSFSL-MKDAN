import numpy as np

# 加载原始的 POI 数据
poi = np.load('E:/trans/DAN/MK-DAN/MyData10%/LABike/LABike_poi.npy')  # 形状为 (6, 6, 14)

# reshape 成 (36, 14)
poi_reshaped = poi.reshape(36, 14)

# 保存为新文件（可选）
np.save('E:/trans/DAN/MK-DAN/MyData10%/LABike/LABike_poi.npy', poi_reshaped)
