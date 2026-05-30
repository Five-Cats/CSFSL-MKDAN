import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('E:/ST/Datasets/DCwashington/DCwashington202304.csv')

# 随机抽取 3 万条数据
df_sampled = df.sample(n=30000, random_state=2)

# 保存到新的 CSV 文件
df_sampled.to_csv('E:/ST/Datasets/DCwashington/DCwashington202304few.csv', index=False)

# 查看前几行以确保正确
print(df_sampled.head())
