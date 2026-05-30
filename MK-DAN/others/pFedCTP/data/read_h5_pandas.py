import pandas as pd


file_path = 'labike/labike.h5'
# 读取 HDF5 文件中的数据
df = pd.read_hdf(file_path)
print(df)
