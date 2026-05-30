import numpy as np
import pandas as pd

'''
本代码是将数据集中的起始点和终止点分开进行处理，
得到不同城市按时间和空间分成小区后各个小区在各个时间段的车流量数据，
便于后续求两小区相似性
'''

# 定义经纬度划分区间
l = 6
# 定义划分的时间段(数据集中的时间都是从一天的00:00开始)
t = 1200

def get_s_e_list(data_read):
    # 定义started和ended数组
    s_list = []
    e_list = []
    # 定义存储经纬度的数组
    lat = []
    lon = []
    length = len(data_read)
    j = 0
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
            s_list.append(started)
            e_list.append(ended)
            # 存储经纬度
            lat.append(s_list[j][2])
            lat.append(e_list[j][2])
            lon.append(s_list[j][3])
            lon.append(e_list[j][3])
            j += 1
    # 按照时间顺序进行排序
    S_list = sorted(s_list, key=lambda x: x[1], reverse=False)
    E_list = sorted(e_list, key=lambda x: x[1], reverse=False)
    n = len(s_list)
    # 将经纬度转换为集合去重，再转换回来
    lat = list(set(lat))
    lon = list(set(lon))
    # 获取存储最值的数组
    Max_t_s = [(max(S_list[n-1][1], E_list[n-1][1])), max(lat), max(lon)]
    Min_t_s = [min(S_list[0][1], E_list[0][1]), min(lat), min(lon)]
    return s_list, e_list, Max_t_s, Min_t_s

def get_time_cell(s_list, e_list, Max, Min, l0, t0):
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
    # 对数组进行初始化，先按小区总数初始化
    T_S = np.zeros((l0*l0, s), dtype=int)
    T_E = np.zeros((l0*l0, s), dtype=int)
    # 获取邻接矩阵
    for i in range(0, l0*l0):
        adj_m.append([])
        for j in range(0, l0 * l0):
            if j == i:
                adj_m[i].append(1)
            elif (j == i - 1 and j % l0 != l0 - 1) or (j == i + 1 and j % l0 != 0) or (abs(j - i)) / l0 == 1:
                adj_m[i].append(0.5)
            else:
                adj_m[i].append(0)
    n = len(s_list)
    for i in range(0, n):
        # 根据start和end存放
        a = int((s_list[i][1] - t_min) // t0)
        b1 = int((s_list[i][2] - lat_min) // l1)
        c1 = int((s_list[i][3] - lon_min) // l2)
        b2 = int((e_list[i][2] - lat_min) // l1)
        c2 = int((e_list[i][3] - lon_min) // l2)
        # 取最大值时需减去1，放入最后的小区
        if b1 == l0:
            b1 -= 1
        if c1 == l0:
            c1 -= 1
        if b2 == l0:
            b2 -= 1
        if c2 == l0:
            c2 -= 1
        # 计算该条数据中start和end的小区标号
        s1 = c1 * l0 + b1
        e1 = c2 * l0 + b2
        # 分别存放各个小区各个时间段为起始点和终止点的车流量数
        T_S[s1][a] += 1
        T_E[e1][a] += 1
    return T_S, T_E, adj_m

# 读取数据集
# 读取started_at,ended_at,start_latitude,start_longitude,end_latitude,end_longitude六个数据
# 德克萨斯洲奥斯汀市2016年10月（数据分布过于不均衡！）
# data = pd.read_csv('E:/ST/Datasets/RideAustin_Weather/RideAustin_Weather10.csv', nrows=31350)
# selected_data = data.iloc[:, [4, 0, 14, 13, 2, 3]]
# 纽约2016年4月
# data = pd.read_csv('E:/ST/Datasets/NYCBike/2016-citibike-tripdata/2016-citibike-tripdata/4_April/201604-citibike-tripdata_0.csv', nrows=155000)
# selected_data = data.iloc[:, [1, 2, 5, 6, 9, 10]]
# 纽约2023年4月
data = pd.read_csv('E:/trans/DAN/MK-DAN/data/NYCBike/NYCBike202304.csv')
selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 华盛顿2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Washington/Washington202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 芝加哥2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Chicago/Chicago202304.csv')
# selected_data = data.iloc[:, [2, 3, 8, 9, 10, 11]]
# 洛杉矶2023年4月
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/LABike/LABike202304.csv')
# selected_data = data.iloc[:, [2, 3, 5, 6, 8, 9]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
S_L, E_L, Max_S, Min_S = get_s_e_list(list_read)
Time_Start, Time_End, adj_mx = get_time_cell(S_L, E_L, Max_S, Min_S, l, t)
print(Max_S, Min_S)
