import numpy as np

# 1. 加载已有 .npy 文件
npy_path = 'E:/trans/DAN/DASTNet/data/LABike/LABike.npy'       # 替换成你的实际路径
data = np.load(npy_path)         # 形状: (2161, 36, 1)

# 2. 检查形状（可选）
print("Data shape:", data.shape)  # 应该是 (2161, 36, 1)

# 3. 保存为 .npz 文件，key='data'
npz_path = 'E:/trans/DAN/DASTNet/data/LABike/LABike.npz'       # 你希望保存的目标文件
np.savez(npz_path, data=data)

print(f"保存成功: {npz_path}")
