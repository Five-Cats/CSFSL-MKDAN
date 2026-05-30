import numpy as np

# 替换为你的 .npz 文件路径
file_path = 'E:/trans/DAN/DASTNet/data/Washington/Washington.npz'

# 加载 .npz 文件
data = np.load(file_path)

# 列出其中包含的所有数组名称
print("文件中包含的键：", data.files)

# 按键访问并打印每个数组内容
for key in data.files:
    print(f"\n键名: {key}")
    print("数据内容:\n", data[key])
    print("数据形状:", data[key].shape)

# 示例：访问特定键的数据
# e.g., if the file has a key 'X':
# X = data['X']
