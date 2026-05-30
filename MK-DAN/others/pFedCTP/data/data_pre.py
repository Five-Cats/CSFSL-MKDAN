import numpy as np
import pandas as pd
import datetime

# 定义经纬度划分区间
l = 6
# 定义划分的时间段(数据集中的时间都是从一天的00:00开始)
t = 1200

# print(len(list_read))

# 定义对数据进行预处理的函数
def get_list(data_read):
    # 定义存储车流量的数组
    N_list = []
    # 定义存储经纬度的数组
    lat = []
    lon = []
    length = len(data_read)
    # 处理数据中的时间
    for i in range(0, length):
        # 将started时间和ended时间整数化，表示自1970年1月1日00:00:00 UTC以来经过的秒数
        datatime1 = pd.to_datetime(data_read[i][0])
        datatime2 = pd.to_datetime(data_read[i][1])
        time1 = int(datatime1.timestamp())
        time2 = int(datatime2.timestamp())
        data_read[i].append(time1)
        data_read[i].append(time2)
        started = [datatime1, data_read[i][6], float(data_read[i][2]), float(data_read[i][3])]
        ended = [datatime2, data_read[i][7], float(data_read[i][4]), float(data_read[i][5])]
        # 存储start和end相关数据
        if not np.isnan(started[2]) and not np.isnan(ended[2]):
            N_list.append(started)
            N_list.append(ended)
            # 存储经纬度
            lat.append(started[2])
            lat.append(ended[2])
            lon.append(started[3])
            lon.append(ended[3])
    # 按照时间顺序进行排序
    N_T = sorted(N_list, key=lambda x: x[1], reverse=False)
    n = len(N_T)
    # 将经纬度转换为集合去重，再转换回来
    lat = list(set(lat))
    lon = list(set(lon))
    # 获取存储最值的数组
    Max_t_s = [N_T[n - 1][1], max(lat), max(lon)]
    Min_t_s = [N_T[0][1], min(lat), min(lon)]
    return N_T, Max_t_s, Min_t_s

def T_cell(N_list, Max, Min, l0, t0):
    # 定义存储各个小区的数组(此处为36个小区)
    C = np.arange(1, l0*l0+1)
    # 定义存储空间关系的数组
    adj_mx = []
    t_min = Min[0]
    t_max = Max[0]
    lat_min = Min[1]
    lat_max = Max[1]
    lon_min = Min[2]
    lon_max = Max[2]
    # 将各个时间点按时间段分类
    s = int((t_max - t_min) // t0) + 1
    # 定义存储时间的数组
    # T = pd.date_range(start='2023-04-01', periods=s, freq='20T')
    T = np.arange(t_min, t_min + t0 * s, t0)
    # 将各个节点按经纬度分成不同小区
    l1 = (lat_max - lat_min) / l0
    l2 = (lon_max - lon_min) / l0
    # 定义存储各个时间段各个小区的车流量数组，并对其进行初始化
    T_C = np.zeros((s, l0 * l0))
    # 获取邻接矩阵
    for i in range(0, l0 * l0):
        adj_mx.append([])
        # 按地理位置分配不同权重
        for j in range(0, l0 * l0):
            if j == i:
                adj_mx[i].append(1)
            elif j == i - 1 or j == i + 1 or j == i + l0 or j == i - l0:
                adj_mx[i].append(0.5)
            # elif j == i - 2 or j == i + 2 or j == i + l0 * 2 or j == i - l0 * 2:
            #     adj_m[i].append(0.5)
            # elif j == i - 3 or j == i + 3 or j == i + l0 * 3 or j == i - l0 * 3:
            #     adj_m[i].append(0.25)
            else:
                adj_mx[i].append(0)
    n = len(N_list)
    for i in range(0, n):
        # 根据start和end存放
        a = int((N_list[i][1] - t_min) // t0)
        b = int((N_list[i][2] - lat_min) // l1)
        c = int((N_list[i][3] - lon_min) // l2)
        # 取最大值时需减去1，放入最后的小区
        if b == l0:
            b -= 1
        elif c == l0:
            c -= 1
        # 计算该条数据中的小区标号
        c0 = c * l0 + b
        # 分别存放各个小区各个时间段为起始点和终止点的车流量数
        T_C[a][c0] += 1
    return T_C, T, C, adj_mx

def get_time_cell(N_list, Max, Min, l0, t0):
    # 定义存储空间关系的数组
    adj_m = []
    t_min = Min[0]
    t_max = Max[0]
    lat_min = Min[1]
    lat_max = Max[1]
    lon_min = Min[2]
    lon_max = Max[2]
    # 将各个时间点按时间段分类
    s = int((t_max - t_min) // t0) + 1
    # 将各个节点按经纬度分成不同小区
    l1 = (lat_max - lat_min) / l0
    l2 = (lon_max - lon_min) / l0
    # 定义存储各个时间段各个小区的车流量数组，并对其进行初始化
    T_C = np.zeros((s, l0 * l0, 1))
    # 获取邻接矩阵
    for i in range(0, l0*l0):
        adj_m.append([])
        # 按地理位置分配不同权重
        for j in range(0, l0*l0):
            if j == i:
                adj_m[i].append(1)
            elif j == i - 1 or j == i + 1 or j == i + l0 or j == i - l0:
                adj_m[i].append(0.5)
            # elif j == i - 2 or j == i + 2 or j == i + l0 * 2 or j == i - l0 * 2:
            #     adj_m[i].append(0.5)
            # elif j == i - 3 or j == i + 3 or j == i + l0 * 3 or j == i - l0 * 3:
            #     adj_m[i].append(0.25)
            else:
                adj_m[i].append(0)
    n = len(N_list)
    # 获取各个时间的时间戳，表明是某一天的那一个时间段
    dt = pd.to_datetime("2023/4/1 00:00:00")
    t = int(dt.timestamp())
    # 计算时间戳
    st = 1 / (60 * 24 / (t0 / 60))
    for i in range(0, n):
        # 根据start和end存放
        a = int((N_list[i][1] - t_min) // t0)
        b = int((N_list[i][2] - lat_min) // l1)
        c = int((N_list[i][3] - lon_min) // l2)
        # 取最大值时需减去1，放入最后的小区
        if b == l0:
            b -= 1
        elif c == l0:
            c -= 1
        # 计算该条数据中的小区标号
        c0 = c * l0 + b
        # 分别存放各个小区各个时间段为起始点和终止点的车流量数
        T_C[a][c0][0] += 1
    # 将numpy数组转换成列表便于后续添加时间戳
    T_C_L = T_C.tolist()
    for i in range(0, s):
        # 计算时间戳
        s_t = st * (i % (60 * 24 / (t0 / 60)))
        for j in range(0, l0 * l0):
            T_C_L[i][j].append(s_t)
    return T_C_L, adj_m

# 读取数据集
# 读取started_at,ended_at,start_latitude,start_longitude,end_latitude,end_longitude六个数据
# 纽约2023年4月
# data = pd.read_csv('E:/ST/Datasets/NYCBike/NYCBike202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 华盛顿2023年4月
# data = pd.read_csv('E:/ST/Datasets/DCwashington/DCwashington202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 芝加哥2023年4月
data = pd.read_csv('E:/ST/Datasets/Chicago/Chicago202304.csv')
selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 洛杉矶2023年4月
# data = pd.read_csv('E:/ST/Datasets/LABike/LABike202304.csv')
# selected_data = data.iloc[:, [2, 3, 5, 6, 8, 9]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
# 对数据集进行预处理
L, Max_T_C, Min_T_C = get_list(list_read)
# Time_Cell, adj_mx = get_time_cell(L, Max_T_C, Min_T_C, l, t)
Time_Cell, Time, Cell, adj_mx = T_cell(L, Max_T_C, Min_T_C, l, t)
# np.save('E:/trans/ST/ST-GFSL/data/washington/datasetfew.npy', Time_Cell)
# np.save('E:/trans/ST/ST-GFSL/data/washington/matrixfew.npy', adj_mx)
print(Max_T_C, Min_T_C)