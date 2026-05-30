import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

output_dir = 'E:/trans/ST/MetaST/outputs/'

resdir = os.listdir(output_dir)
print("文件夹内容:", resdir)
data = {}

# 读取所有输出文件
for eachfile in resdir:
    if 'output' in eachfile:
        data[eachfile] = np.load(os.path.join(output_dir, eachfile))['arr_0'].squeeze()
        print(eachfile, data[eachfile].shape)

print("\n--- 数据加载完成 ---\n")

# oracle 是归一化后的值
labels = data["output_bike_oracle.npz"]
print("labels min:", labels.min())
print("labels max:", labels.max())
print("labels mean:", labels.mean())

# 自动获取 max_val
# labike
# max_val = [19, 25]
max_val = 25
# chicago
# max_val = [105, 137]
# washington
# max_val = [97, 94]
# nycbike
# max_val = [302, 292]
print("自动检测到 max_val =", max_val)

# 定义批量评估函数
def evaluate_in_batches(labels, outputs, name, batch_size=128, max_val=1.0):
    # 先保证 outputs 和 labels 维度匹配
    if outputs.ndim == 3:  # (batch, seq, feature)
        outputs = outputs.reshape(-1, outputs.shape[-1])
    if labels.ndim == 3:
        labels = labels.reshape(-1, labels.shape[-1])

    # 缩放回原始值
    labels = labels * max_val
    outputs = outputs * max_val
    print("labels shape:", labels.shape)

    n_samples = min(len(labels), len(outputs))
    labels = labels[:n_samples]
    outputs = outputs[:n_samples]

    total_rmse, total_mae = [], []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        label_batch = labels[start:end]
        output_batch = outputs[start:end]

        rmse = np.sqrt(mean_squared_error(label_batch, output_batch))
        mae = mean_absolute_error(label_batch, output_batch)

        total_rmse.append(rmse)
        total_mae.append(mae)

        print(f"[{name}] batch {start//batch_size}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    # 最终统计
    print(f"\n[{name}] RMSE -> mean={np.mean(total_rmse):.4f}, min={np.min(total_rmse):.4f}")
    print(f"[{name}] MAE  -> mean={np.mean(total_mae):.4f}, min={np.min(total_mae):.4f}\n")


# -------------------------------
# 分别评估所有预测结果
# -------------------------------
for eachfile in data:
    if eachfile == 'output_bike_oracle.npz':  # 跳过 oracle
        continue
    outputs = data[eachfile]
    print(f"开始评估 {eachfile}, shape={outputs.shape}")
    evaluate_in_batches(labels, outputs, eachfile, batch_size=128, max_val=max_val)

# 也可以直接评估整体结果（不分 batch）
def evaluate_all(labels, outputs, name, max_val=1.0):
    if outputs.ndim == 3:
        outputs = outputs.reshape(-1, outputs.shape[-1])
    if labels.ndim == 3:
        labels = labels.reshape(-1, labels.shape[-1])

    labels = labels * max_val
    outputs = outputs * max_val

    n_samples = min(len(labels), len(outputs))
    labels = labels[:n_samples]
    outputs = outputs[:n_samples]

    rmse = np.sqrt(mean_squared_error(labels, outputs))
    mae = mean_absolute_error(labels, outputs)

    print(f"[{name}] 全部数据: RMSE={rmse:.4f}, MAE={mae:.4f}\n")

# -------------------------------
# 最后也打印整体 RMSE/MAE
# -------------------------------
for eachfile in data:
    if eachfile == 'output_bike_oracle.npz':
        continue
    outputs = data[eachfile]
    evaluate_all(labels, outputs, eachfile, max_val=max_val)
