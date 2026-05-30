import h5py

# 打开 HDF5 文件
file_path = 'labike/labike.h5'
with h5py.File(file_path, 'r') as f:
    # 列出文件中所有的主键
    print("Keys in the file:", list(f.keys()))
    print("Keys in 'df' group:", list(f['df'].keys()))

    # 访问某个组或数据集
    dataset = f['df']['axis0']
    print("数据集内容：")
    print(dataset[:])  # 读取数据集内容

    dataset_name = 'axis1'
    dataset = f['df']['axis1']
    print("数据集内容：")
    print(dataset[:])  # 读取数据集内容

    dataset = f['df']['block0_items']
    print("数据集内容：")
    print(dataset[:])  # 读取数据集内容

    dataset = f['df']['block0_values']
    print("数据集内容：")
    print(dataset[:])  # 读取数据集内容

    # 或者遍历所有组和数据集
    def print_structure(name, obj):
        print(name, obj)

    f.visititems(print_structure)
