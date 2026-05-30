import numpy as np
import pandas as pd

# 纬度和经度均分成3部分，即一共分为9个小区
X_SPACE = 3
Y_SPACE = 3
# 定义用来按经纬度分类的数组
LIST_Block = []
# 定义按照时间排序后的数组
LIST_Block_Sorted = []
# 定义时间最大值
M_time = 0
# 定义时间步长
T = 50000
# 定义以时间为步长的数组
LIST_Time = []
# 定义小区关系数组
LIST_Relation = []

data_read = pd.read_csv('E://CSFSL/datasets/others/crowd_temperature/crowd_temperature.csv')
# 读取数据集的地理范围并进行区域划分
max_x = data_read['Latitude'].max()
min_x = data_read['Latitude'].min()
max_y = data_read['Longitude'].max()
min_y = data_read['Longitude'].min()
# print(max_x, min_x, max_y, min_y)

x_space = (max_x - min_x) / X_SPACE
y_space = (max_y - min_y) / Y_SPACE
Block = X_SPACE * Y_SPACE
# print(x_space, y_space, Block)

# 将时间分隔开为int格式存储
data_read['year'] = data_read['Date'].apply(lambda x: int(x[6:8]))
data_read['month'] = data_read['Date'].apply(lambda x: int(x[0:2]))
data_read['day'] = data_read['Date'].apply(lambda x: int(x[3:5]))
data_read['hour'] = data_read['Time'].apply(lambda x: int(x[0:2]))
data_read['minute'] = data_read['Time'].apply(lambda x: int(x[3:5]))
data_read['second'] = data_read['Time'].apply(lambda x: int(x[6:8]))

# 读取Latitude,Longitude,Temperature,day,hour,minute,second七个数据
selected_data = data_read.iloc[:, [3, 4, 5, 8, 9, 10, 11]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
length = len(list_read)

for i in range(0, Block):
    # 定义存储数据的数组
    LIST_Block.append([])
    # 构造小区间关系数组
    LIST_Relation.append([])
    for j in range(0, Block):
        if j == i + 1 or j == i - 1:
            LIST_Relation[i].append(1)
        else:
            LIST_Relation[i].append(0)
# print(LIST_Relation)


for j in range(0, length):
    day = int(list_read[j][3]) - 1
    second = day * 86400 + ((list_read[j][4] * 60 + list_read[j][5]) * 60) + list_read[j][6]
    # 找到时间最大值，便于后面确定时间步长
    if second > M_time:
        M_time = second
    # 将day,hour,minute,second均表示成second，便于接下来的计算
    list_read[j].append(second)
    # 按经纬度划分为多个区域
    x = (list_read[j][0] - min_x) // x_space
    y = (list_read[j][1] - min_y) // y_space
    if list_read[j][0] == max_x:
        x -= 1
    if list_read[j][1] == max_y:
        y -= 1
    block = int(x*Y_SPACE+y)
    # 按照经纬度分为各个小区
    LIST_Block[block].append(list_read[j])
# print(LIST_Block)
# print(M_time)

# 将上述分好的小区中的数据按照时间进行排列
for i in range(0, len(LIST_Block)):
    sorted_array = sorted(LIST_Block[i], key = lambda x:x[7])
    LIST_Block_Sorted.append(sorted_array)
# print(LIST_Block_Sorted)

# 计算按时间步分成了多少时间段
a = int(M_time // T)
# print(a)
for i in range(0, a):
    LIST_Time.append([])
for i in range(0, Block):
    b = 0
    # 定义一个临时数组用来存储每个小区在该时间段内的温度值，便于计算平均值
    Mean = []
    j = 0
    while j in range(0, len(LIST_Block_Sorted[i])):
        time = LIST_Block_Sorted[i][j][7]
        if time >= b * T and time < (b + 1) * T:
            Mean.append(LIST_Block_Sorted[i][j][2])
            j += 1
        # 数组内的数据已经按照时间顺序排好了
        else:
            mean_t = np.mean(Mean)
            if Mean == []:
                if i == 0 and b == 0:
                    mean_t = 0
                elif i == 0:
                    mean_t = LIST_Time[b-1][i]
                elif b == 0:
                    mean_t = LIST_Time[b][i-1]
                else:
                    # 同时间段前一小区的温度与同一小区前一时间段温度的平均值
                    mean_t = (LIST_Time[b][i-1] + LIST_Time[b-1][i]) / 2
            LIST_Time[b].append(mean_t)
            Mean = []
            b += 1
    if b < a:
        LIST_Time[b].append(LIST_Time[b-1][i])
# print(LIST_Time)

