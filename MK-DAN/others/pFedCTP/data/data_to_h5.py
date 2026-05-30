import pandas as pd
import numpy as np
import h5py
from data_pre import Time_Cell, Time, Cell, adj_mx

# # 创建 HDF5 文件
# with h5py.File('E:/trans/ST/ST-GFSL/data/labike/labike.h5', 'w') as f:
#     # 创建 /df Group
#     df_group = f.create_group('df')
#
#     # 添加 block 数组为 axis0 和 block0_items
#     df_group.create_dataset('axis0', data=Cell.astype('S6'))  # S6 类型表示字符串
#     df_group.create_dataset('block0_items', data=Cell.astype('S6'))  # 同样的区域列表
#
#     # 添加 time 数组为 axis1
#     df_group.create_dataset('axis1', data=Time)  # 时间戳数组
#
#     # 添加车流量数据为 block0_values
#     df_group.create_dataset('block0_values', data=Time_Cell)


# 创建HDF5文件
file_path = 'E:/trans/ST/ST-GFSL/data/chicago/chicago.h5'
with h5py.File(file_path, 'w') as f:
    # 创建/df组
    df_group = f.create_group('df')

    # 创建数据集
    df_group.create_dataset('data', data=Time_Cell)
    df_group.create_dataset('columns', data=np.array(Cell, dtype='S6'))
    df_group.create_dataset('index', data=Time)  # 存储时间戳为整数格式

# 读取HDF5文件
file_path = 'E:/trans/ST/ST-GFSL/data/chicago/chicago.h5'
with h5py.File(file_path, 'r') as f:
    # 读取数据集
    data = f['df/data'][:]
    columns = f['df/columns'][:].astype(str)  # 区域ID
    index_int = f['df/index'][:]  # 时间戳

index = pd.to_datetime(index_int, unit='s')
# 将读取的数据转换为DataFrame
df = pd.DataFrame(data, columns=columns, index=index)

# 显示DataFrame
print(df)
