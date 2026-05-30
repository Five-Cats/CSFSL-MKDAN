import torch

# 指定你的 .pkl 文件路径
file_path = 'E:/trans/DAN/DASTNet/embeddings/node2vec/pems04/64_vecdim.pkl'

# 读取 pkl 文件
data = torch.load(file_path, map_location='cpu')

# 打印内容类型与示例
print("数据类型:", type(data))
print("数据内容示例:", data if isinstance(data, (int, float, str)) else str(data)[:500])
