import numpy as np
import pandas as pd

# 读取数据的行数
# ROW = 10000000
# 纬度和经度均分成X_SPACE部分和Y_SPACE部分，即一共分为X_SPACE*Y_SPACE个小区
X_SPACE = [1, 1, 1, 1]
# Y_SPACE = [30, 30, 30, 30]
Y_SPACE = [25, 30, 15, 45]
# X_SPACE = [2, 2, 2, 2]
# Y_SPACE = [15, 15, 15, 15]

Block = []
for i in range(0, len(X_SPACE)):
    Block.append(X_SPACE[i] * Y_SPACE[i])
# 各道路时间最值
max_t = []
min_t = []
# 各道路X坐标最值
max_X = []
min_X = []
# 各道路Y坐标最值
max_Y = []
min_Y = []
# 各道路X坐标和Y坐标的小区间隔
X_space = []
Y_space = []
# 定义时间段长度
T = [10, 10, 30, 5000]
# 定义用来按经纬度分类的数组
LIST_Block = []
# 定义每个小区存储各个时间段车流量的数组
LIST_Num = []
# 定义存储每条道路中各个小区之间关系的数组
LIST_Relation = []

data_read = pd.read_csv('E://CSFSL/datasets/NGSIM/data.csv')
# 读取Global_Time,Global_X,Global_Y,Location四个数据
selected_data = data_read.iloc[:, [3, 6, 7, 24]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
length = len(list_read)

# 定义用来按道路分类的数组
LIST_Location = []
# 一共有us-101,lankershim,i-80,peachtree四条道路
for i in range(0, 4):
    LIST_Location.append([])
for i in range(0, length):
    # 将X坐标和Y坐标均保留三位小数，方便计算
    list_read[i][1] = float(format(list_read[i][1], '.3f'))
    list_read[i][2] = float(format(list_read[i][2], '.3f'))
    if list_read[i][0] > 1000000000000:
        list_read[i][0] //= 1000
    if list_read[i][3] == 'us-101':
        LIST_Location[0].append(list_read[i])
    elif list_read[i][3] == 'lankershim':
        LIST_Location[1].append(list_read[i])
    elif list_read[i][3] == 'i-80':
        LIST_Location[2].append(list_read[i])
    elif list_read[i][3] == 'peachtree':
        LIST_Location[3].append(list_read[i])
# print(LIST_Location)

# 获取道路上时间,X坐标和Y坐标的最值,并计算出X和Y轴小区间隔
def extr_column(data, x, y):
    max_time = max((row[0], i) for i, row in enumerate(data))[0]
    min_time = min((row[0], i) for i, row in enumerate(data))[0]
    max_X = max((row[1], i) for i, row in enumerate(data))[0]
    min_X = min((row[1], i) for i, row in enumerate(data))[0]
    max_Y = max((row[2], i) for i, row in enumerate(data))[0]
    min_Y = min((row[2], i) for i, row in enumerate(data))[0]
    X_space = (max_X - min_X) / x
    Y_space = (max_Y - min_Y) / y
    # T_space = (max_time - min_time) /
    L = (max_time, min_time, max_X, min_X, max_Y, min_Y, X_space, Y_space)
    list(L)
    return L

# 获取四条道路(us-101,lankershim,i-80,peachtree)的相关数据
for i in range(0, len(LIST_Location)):
    L = extr_column(LIST_Location[i], X_SPACE[i], Y_SPACE[i])
    max_t.append(L[0])
    min_t.append(L[1])
    max_X.append(L[2])
    min_X.append(L[3])
    max_Y.append(L[4])
    min_Y.append(L[5])
    X_space.append(L[6])
    Y_space.append(L[7])
# print(max_t, min_t, max_X, min_X, max_Y, min_Y, X_space, Y_space)

# 将数据按照道路-小区-时间段分类存放在大数组内
for i in range(0, len(LIST_Location)):
    LIST_Block.append([])
    # 定义每小区时间段的最大值
    N = (max_t[i] - min_t[i]) // T[i]
    for j in range(0, Block[i]):
        # 定义存储数据的数组
        LIST_Block[i].append([])
        for k in range(0, N + 1):
            LIST_Block[i][j].append([])
for i in range(0, len(LIST_Location)):
    for j in range(0, len(LIST_Location[i])):
        # 按经纬度划分为多个区域
        x = (LIST_Location[i][j][1] - min_X[i]) // X_space[i]
        y = (LIST_Location[i][j][2] - min_Y[i]) // Y_space[i]
        if LIST_Location[i][j][1] == max_X[i]:
            x -= 1
        if LIST_Location[i][j][2] == max_Y[i]:
            y -= 1
        block = int(x * Y_SPACE[i] + y)
        # 最小时间值为该道路的最小时间
        t = (LIST_Location[i][j][0] - min_t[i]) // T[i]
        # print(t, block)
        # 按照经纬度分为各个小区,再按照时间在各个小区内分为不同时间段
        LIST_Block[i][block][t].append(LIST_Location[i][j])
# 首先按道路分为四大组，每大组里面又按坐标分为100小区，每个小组包含坐标在该小区范围内的一条数据，即相当于1车流量
# print(LIST_Block)

# 计算每条道路每个小区内每个时间段的车流量
# 遍历四条道路
for i in range(0, len(LIST_Location)):
    LIST_Num.append([])
    LIST_Relation.append([])
    # 定义每小区时间段的最大值
    N = (max_t[i] - min_t[i]) // T[i]
    # 遍历每个时间段
    for j in range(0, N + 1):
        LIST_Num[i].append([])
        # 遍历每个小区内该时间段的车流量
        for k in range(0, Block[i]):
            LIST_Num[i][j].append(len(LIST_Block[i][k][j]))
            # LIST_Num[i][j][k] = LIST_Num[i][j][k] / 1000
            # 构建每条街道中各个小区关系矩阵
            if j == 0:
                LIST_Relation[i].append([])
                for a in range(0, Block[i]):
                    # 相邻小区关系为1(自身为0)
                    if a == k + 1 or a == k - 1:
                        LIST_Relation[i][k].append(1)
                    else:
                        LIST_Relation[i][k].append(0)
# print(LIST_Num)
