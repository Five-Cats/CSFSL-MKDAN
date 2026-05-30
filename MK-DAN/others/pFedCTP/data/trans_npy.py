import numpy as np
import pandas as pd

# 加载 .npy 文件
data = np.load('E:/trans/ST/ST-GFSL/data/chicago/datasetfew.npy')

# 查看数组的内容
print(data)

# 查看数组的形状、类型等信息
print("Shape:", data.shape)
print("Data type:", data.dtype)