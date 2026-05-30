import h5py

# 打开H5文件
file_path = 'E:/trans/ST/FlashST/data/nyc_bike0/data.h5'  # 替换为你的H5文件路径
with h5py.File(file_path, 'r') as h5_file:
    # 查看文件中的所有组
    print("文件中的组：")
    print(list(h5_file.keys()))

    # 访问某个组或数据集
    dataset_name = 'bike_drop'  # 替换为实际的组或数据集名称
    if dataset_name in h5_file:
        dataset = h5_file[dataset_name]
        print("数据集内容：")
        print(dataset[:])  # 读取数据集内容

    # 或者遍历所有组和数据集
    def print_structure(name, obj):
        print(name, obj)

    h5_file.visititems(print_structure)
