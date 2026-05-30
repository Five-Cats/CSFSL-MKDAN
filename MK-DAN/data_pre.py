import numpy as np
import pandas as pd
from collections import defaultdict
import datetime

# 定义经纬度划分区间
l = 6
# 定义划分的时间段(数据集中的时间都是从一天的00:00开始，1200s即20min，一天的时间为72*t)
t = 1200

# print(len(list_read))

# 定义对数据进行预处理的函数
def get_list(data_read):
    # 定义存储车流量的数组
    # N_list = []
    S_list = []
    E_list = []
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
            S_list.append(started)
            E_list.append(ended)
            # N_list.append(started)
            # N_list.append(ended)
            # 存储经纬度
            lat.append(started[2])
            lat.append(ended[2])
            lon.append(started[3])
            lon.append(ended[3])
    # 按照时间顺序进行排序
    S_T = sorted(S_list, key=lambda x: x[1], reverse=False)
    E_T = sorted(E_list, key=lambda x: x[1], reverse=False)
    # N_T = sorted(N_list, key=lambda x: x[1], reverse=False)
    n = len(S_T)
    # 将经纬度转换为集合去重，再转换回来
    lat = list(set(lat))
    lon = list(set(lon))
    # 获取存储最值的数组
    Max_S = [S_T[n - 1][1], max(lat), max(lon)]
    Min_S = [S_T[0][1], min(lat), min(lon)]
    Max_E = [E_T[n - 1][1], max(lat), max(lon)]
    Min_E = [E_T[0][1], min(lat), min(lon)]
    # Max = [N_T[n - 1][1], max(lat), max(lon)]
    # Min = [N_T[0][1], min(lat), min(lon)]

    return S_T, E_T, Max_S, Min_S, Max_E, Min_E
    # return N_T, Max, Min

# 统计起始-终止点（OD对）
def ODPairs(data_read, l0):
    # 定义存储车流量的数组
    N_list = []
    # 定义存储经纬度的数组
    lat = []
    lon = []
    length = len(data_read)
    # 遍历数据
    for i in range(0, length):
        # [s_lat, s_lon, e_lat, e_lon]
        temp = [float(data_read[i][2]), float(data_read[i][3]), float(data_read[i][4]), float(data_read[i][5])]
        # 存储start和end相关数据
        if not np.isnan(temp[0]) and not np.isnan(temp[2]):
            N_list.append(temp)
            # 存储经纬度
            lat.append(temp[0])
            lat.append(temp[2])
            lon.append(temp[1])
            lon.append(temp[3])
    n = len(N_list)
    # 将经纬度转换为集合去重，再转换回来
    lat = list(set(lat))
    lon = list(set(lon))
    # 获取存储最值的数组
    Max = [max(lat), max(lon)]
    Min = [min(lat), min(lon)]
    l1 = (Max[0] - Min[0]) / l0
    l2 = (Max[1] - Min[1]) / l0
    # 初始化OD对
    OD_Pairs = defaultdict(int)
    for i in range(0, n):
        # 根据start和end所在小区统计OD对
        a = int((N_list[i][0] - Min[0]) // l1)
        b = int((N_list[i][1] - Min[1]) // l2)
        c = int((N_list[i][2] - Min[0]) // l1)
        d = int((N_list[i][3] - Min[1]) // l2)
        # 取最大值时需减去1，放入最后的小区
        if a == l0:
            a -= 1
        if b == l0:
            b -= 1
        if c == l0:
            c -= 1
        if d == l0:
            d -= 1
        # 计算该条数据中start和end分别对应的小区标号
        # A = b * l0 + a
        # B = d * l0 + c
        OD_Pairs[(a, b), (c, d)] += 1
    # 转换为标准字典
    OD_Pairs = dict(OD_Pairs)
    return OD_Pairs

def T_cell(N_list, Max, Min, l0, t0):
    # 定义存储空间关系的数组
    adj_mx = []
    t_min = Min[0]
    t_max = Max[0]
    lat_min = Min[1]
    lat_max = Max[1]
    lon_min = Min[2]
    lon_max = Max[2]
    # 获取各个时间的时间戳，表明是某一天的那一个时间段
    dt1 = pd.to_datetime("2023/4/1 00:00:00")
    t1 = int(dt1.timestamp())
    dt2 = pd.to_datetime("2023/5/1 00:00:00")
    t2 = int(dt2.timestamp())
    # 将各个时间点按时间段分类
    s = int((t2 - t1) // t0) + 1
    # 定义存储时间的数组
    # T = pd.date_range(start='2023-04-01', periods=s, freq='20T')
    # T = np.arange(t_min, t_min + t0 * s, t0)
    # 将各个节点按经纬度分成不同小区
    l1 = (lat_max - lat_min) / l0
    l2 = (lon_max - lon_min) / l0
    # 定义存储各个时间段各个小区的车流量数组，并对其进行初始化
    T_C = np.zeros((s, l0, l0))
    # 获取邻接矩阵
    for i in range(0, l0):
        adj_mx.append([])
        for j in range(0, l0):
            adj_mx[i].append([])
            for k in range(0, l0):
                adj_mx[i][j].append([])
                # 按地理位置分配不同权重
                for l in range(0, l0):
                    if k == i and l == j:
                        adj_mx[i][j][k].append(1)
                    elif (k == i - 1 and l == j) or (k == i and l == j - 1) or (k == i and l == j + 1) or (k == i + 1 and l == j):
                        adj_mx[i][j][k].append(0.5)
                    else:
                        adj_mx[i][j][k].append(0)
    adj_mx = np.array(adj_mx)
    n = len(N_list)
    for i in range(0, n):
        # 根据start和end存放
        a = int((N_list[i][1] - t1) // t0)
        b = int((N_list[i][2] - lat_min) // l1)
        c = int((N_list[i][3] - lon_min) // l2)
        # 取最大值时需减去1，放入最后的小区
        if b == l0:
            b -= 1
        if c == l0:
            c -= 1
        # 分别存放各个小区各个时间段为起始点和终止点的车流量数
        T_C[a][b][c] += 1
    T_C = np.array(T_C)
    return T_C, adj_mx

# 加入时间戳的车流量
def get_time_cell(N_list, Max, Min, l0, t0):
    # 定义存储空间关系的数组
    adj_m = []
    t_min = Min[0]
    t_max = Max[0]
    lat_min = Min[1]
    lat_max = Max[1]
    lon_min = Min[2]
    lon_max = Max[2]
    # 获取各个时间的时间戳，表明是某一天的那一个时间段
    dt1 = pd.to_datetime("2023/4/1 00:00:00")
    t1 = int(dt1.timestamp())
    dt2 = pd.to_datetime("2023/5/1 00:00:00")
    t2 = int(dt2.timestamp())
    # 将各个时间点按时间段分类
    s = int((t2 - t1) // t0) + 1
    # 将各个节点按经纬度分成不同小区
    l1 = (lat_max - lat_min) / l0
    l2 = (lon_max - lon_min) / l0
    # 定义存储各个时间段各个小区的车流量数组，并对其进行初始化
    T_C = np.zeros((s, l0 * l0, 1))
    # 获取邻接矩阵
    for i in range(0, l0 * l0):
        adj_m.append([])
        # 按地理位置分配不同权重
        for j in range(0, l0 * l0):
            if j == i:
                adj_m[i].append(1)
            elif j == i - 1 or j == i + 1 or j == i + l0 or j == i - l0:
                adj_m[i].append(0.5)
            elif j == i - 2 or j == i + 2 or j == i + l0 * 2 or j == i - l0 * 2:
                adj_m[i].append(0.25)
            # elif j == i - 3 or j == i + 3 or j == i + l0 * 3 or j == i - l0 * 3:
            #     adj_m[i].append(0.25)
            else:
                adj_m[i].append(0.1)
    n = len(N_list)
    # 计算时间戳
    st = 1 / (60 * 24 / (t0 / 60))
    for i in range(0, n):
        # 根据start和end存放
        a = int((N_list[i][1] - t1) // t0)
        b = int((N_list[i][2] - lat_min) // l1)
        c = int((N_list[i][3] - lon_min) // l2)
        # 取最大值时需减去1，放入最后的小区
        if b == l0:
            b -= 1
        if c == l0:
            c -= 1
        # 计算该条数据中的小区标号
        c0 = c * l0 + b
        # 分别存放各个小区各个时间段为起始点和终止点的车流量数
        T_C[a][c0][0] += 1
    # # 将numpy数组转换成列表便于后续添加时间戳
    # T_C_L = T_C.tolist()
    # for i in range(0, s):
    #     # 计算时间戳
    #     s_t = st * (i % (60 * 24 / (t0 / 60)))
    #     d_t = 2 * (i % (60 * 24 / (t0 / 60)))
    #     for j in range(0, l0 * l0):
    #         T_C_L[i][j].append(s_t)
    #         T_C_L[i][j].append(d_t)
    #         T_C_L[i][j].append(d_t)
    return T_C, adj_m

# 读取数据集
# 读取started_at,ended_at,start_latitude,start_longitude,end_latitude,end_longitude六个数据
# 纽约2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/NYCBike/NYCBike202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 华盛顿2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Washington/Washington202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 芝加哥2023年4月
data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Chicago/Chicago202304.csv')
selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 洛杉矶2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/LABike/LABike202304.csv')
# selected_data = data.iloc[:, [2, 3, 5, 6, 8, 9]]
# 随机获取 10% 的数据
sampled_data = selected_data.sample(frac=0.1, random_state=42)
sampled_data = sampled_data.reset_index(drop=True)
# 将读取到的数据转换成list形式
list_read = np.array(sampled_data).tolist()
# sampled_data = selected_data.sample(n=320000, replace=False)
# list_read = sampled_data.values.tolist()
# 对数据集进行预处理
L_S, L_E, Max_S, Min_S, Max_E, Min_E = get_list(list_read)
# L, Max, Min = get_list(list_read)
# print(len(L))
# Time_Cell, adj_mx = get_time_cell(L, Max, Min, l, t)
Time_Cell_S, adj_mx = T_cell(L_S, Max_S, Min_S, l, t)
Time_Cell_E, _ = T_cell(L_E, Max_E, Min_E, l, t)
OD = ODPairs(list_read, l)
# np.save('E:/trans/DAN/MK-DAN/data/Chicago/dataset_10%.npy', Time_Cell)
# np.save('E:/trans/ST/MTPB/data/nycbike/dataset_expand.npy', Time_Cell[:-1])
np.save('E:/trans/DAN/CrossTReS/data/MyData10%/Chicago/Chicago_pickup.npy', Time_Cell_S)
np.save('E:/trans/DAN/CrossTReS/data/MyData10%/Chicago/Chicago_dropoff.npy', Time_Cell_E)
# np.save('E:/trans/ST/MTPB/data/nycbike/matrix.npy', adj_mx)
# np.save('E:/trans/ST/ST-GFSL/data/washington/matrixfew.npy', adj_mx)
print(OD)
print(Max_S, Min_S, Max_E, Min_E)
