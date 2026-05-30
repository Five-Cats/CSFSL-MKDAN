import numpy as np
import h5py
from data_pre import Time_Start, Time_End, adj_mx

# 获取数组
array1 = Time_Start.T
array2 = Time_End.T
array3 = adj_mx

# 创建H5文件并写入数据
file_path1 = 'E:/trans/ST/FlashST/data/NYCBike/data.h5'  # H5文件路径
file_path2 = 'E:/trans/ST/FlashST/data/NYCBike/matrix.h5'  # H5文件路径
with h5py.File(file_path1, 'w') as h5_file1:
    # 创建数据集
    h5_file1.create_dataset('bike_drop', data=array1)
    h5_file1.create_dataset('bike_pick', data=array2)

print(f"H5文件已保存到 {file_path1}")

with h5py.File(file_path2, 'w') as h5_file2:
    # 创建数据集
    h5_file2.create_dataset('dis_bb', data=array3)
print(f"H5文件已保存到 {file_path2}")
