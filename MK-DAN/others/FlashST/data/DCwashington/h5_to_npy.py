import h5py
import numpy as np

# 打开 HDF5 文件
hdf5_file_path = 'matrix.h5'  # 替换为你的 HDF5 文件路径
npy_file_path = 'matrix.npy'  # 输出的 .npy 文件路径

# 读取 HDF5 数据
with h5py.File(hdf5_file_path, 'r') as h5_file:
    # 假设文件中有一个数据集 'bike_drop'
    data = h5_file['dis_bb'][:]  # 读取数据集

# 保存为 .npy 文件
np.save(npy_file_path, data)

print(f"数据已保存为 {npy_file_path}")
