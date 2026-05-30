import numpy as np
import pandas as pd

data_read = pd.read_csv('E://CSFSL/datasets/others/forecasting/airquality.csv')
# 将时间分隔开为int格式存储
data_read['year'] = data_read['time'].apply(lambda x: int(x[0:4]))
data_read['month'] = data_read['time'].apply(lambda x: int(x[5:7]))
data_read['day'] = data_read['time'].apply(lambda x: int(x[8:10]))
# 读取station_id,PM2.5,year,month,day五个数据
selected_data = data_read.iloc[:, [0, 2, 8, 9, 10]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
length = len(list_read)

# 定义用来按区域分类的数组
LIST_Location = []
# 定义用来存储各小区每天平均PM2.5指数
LIST_Num = []
# 定义用来按时间步和小区存放PM2.5分类的数组
LIST_Time = []
# 定义用来存储小区间关系的数组
LIST_Relation = []
# 每小区按时间分为400组
b = 20


# 查找数据集中共有几个地区
m = 0
n = 0
for i in range(0, length):
    if list_read[i][0] != m:
        m = list_read[i][0]
        n += 1
        # print(list_read[i][0])
        LIST_Location.append([])
        # 若数据为NULL则不加入数组
        if not np.isnan(list_read[i][1]):
            LIST_Location[n - 1].append(list_read[i])
    else:
        if not np.isnan(list_read[i][1]):
            LIST_Location[n - 1].append(list_read[i])
# print(LIST_Location)

# 计算各个小区每天平均PM2.5指数
for i in range(0, len(LIST_Location)):
    LIST_Num.append([])
    w = len(LIST_Location[i]) // b
    k = 0
    z = 0
    L = []
    for j in range(0, len(LIST_Location[i])):
        # 定义一个空数组用来计算平均每天的空气质量
        if k < w-1:
            L.append(LIST_Location[i][j][1])
            k += 1
        else:
            k = 0
            # 计算是第几组
            z += 1
            if z <= b:
                # 计算每天平均PM2.5指数
                Pm = round(np.mean(L), 2)
                LIST_Num[i].append([LIST_Location[i][j-1][0], Pm])
                L = []
            else:
                continue
# print(LIST_Num)

# 按时间步来存储各小区PM2.5数值
for i in range(0, b):
    LIST_Time.append([])
    for j in range(0, len(LIST_Num)):
        LIST_Time[i].append(LIST_Num[j][i][1])
        if i == 0:
            LIST_Relation.append([])
            for a in range(0, len(LIST_Num)):
                # 相邻小区关系为1
                if a == j + 1 or a == j - 1:
                    LIST_Relation[j].append(1)
                else:
                    LIST_Relation[j].append(0)
# print(LIST_Time)
# print(LIST_Relation)




