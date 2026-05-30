import numpy as np
import pandas as pd

l = 6
num_opi = 14

# NYC区域范围
# lat_min, lat_max = 40.58, 40.91
# lon_min, lon_max = -74.0711, -73.53
# Lat_Lon = [40.63333452, 40.83299, -74.024714, -73.9022]
# Washington的区域范围
# lat_min, lat_max = 38.71, 39.1259
# lon_min, lon_max = -77.39, -76.8256
# Lat_Lon = [38.876737, 38.95, -77.08, -76.98]
# Chicago的区域范围
# lat_min, lat_max = 41.6486, 42.08
# lon_min, lon_max = -88.11, -87.52
# Lat_Lon = [41.86, 41.96, -87.69, -87.612043]
# LA的区域范围
# lat_min, lat_max = 33.9285, 34.1777
# lon_min, lon_max = -118.4914, -118.2255
Lat_Lon = [33.928459, 34.168751, -118.491341, -118.231277]


def get_opi(data_read, l0, L):
    # 定义一个用来调试的数组
    test = np.zeros(num_opi)
    # 将各个节点按经纬度分成不同小区
    l1 = (L[1] - L[0]) / l0
    l2 = (L[3] - L[2]) / l0
    # 定义存储各个区域OPI的数组
    T_C = np.zeros((l0 * l0, num_opi))
    length = len(data_read)
    # 遍历OPI数据，进行分类
    for i in range(0, length):
        a = int((float(data_read[i][0]) - L[0]) // l1)
        b = int((float(data_read[i][1]) - L[2]) // l2)
        # 取最大值时需减去1，放入最后的小区
        if a == l0:
            a -= 1
        elif b == l0:
            b -= 1
        # 计算该条数据中的小区标号
        c = b * l0 + a
        # 进行分类
        if data_read[i][2] == "amenity":
            # 第5类为金融机构，如银行
            if data_read[i][3] in ["bank", "atm"]:
                T_C[c][4] += 1
                test[4] += 1
            # 第6类为体育休闲服务
            elif data_read[i][3] in ["cinema", "theatre"]:
                T_C[c][5] += 1
                test[5] += 1
            # 第7类为文化教育服务
            elif data_read[i][3] in ["school", "kindergarten"]:
                T_C[c][6] += 1
                test[6] += 1
            # 第10类为政府及组织
            elif data_read[i][3] in ["place_of_worship"]:
                T_C[c][9] += 1
                test[9] += 1
            # 第12类为餐饮
            elif data_read[i][3] in ["restaurant", "fast_food", "cafe", "ice_cream", "pub", "bar"]:
                T_C[c][11] += 1
                test[11] += 1
            # 第14类为公共服务
            elif data_read[i][3] in ["parking", "fire_station", "parking_entrance", "toilets",
                                  "social_facility", "bicycle_parking", "post_office", "post_box", "fuel"]:
                T_C[c][13] += 1
                test[13] += 1
            else:
                pass
        elif data_read[i][2] == "shop":
            # 第3类为家政服务
            if data_read[i][3] in ["laundry", "dry_cleaning"]:
                T_C[c][2] += 1
                test[2] += 1
            # 第8类为购物
            elif data_read[i][3] in ["convenience", "clothes", "shoes", "gift", "variety_store", "supermarket",
                                  "mobile_phone", "beauty", "bicycle", "jewelry", "cosmetics", "furniture",
                                  "electronics", "craft", "greengrocer", "pet", "florist", "hairdresser",
                                  "optician", "car_repair", "tobacco", "cannabis", "books", "department_store"]:
                T_C[c][7] += 1
                test[7] += 1
            # 第12类为餐饮
            elif data_read[i][3] in ["bakery", "pastry", "deli", "alcohol", "ice_cream"]:
                T_C[c][11] += 1
                test[11] += 1
            else:
                pass
        elif data_read[i][2] == "tourism":
            # 第1类为旅游景区
            if data_read[i][3] in ["attraction"]:
                T_C[c][0] += 1
                test[0] += 1
            # 第6类为体育休闲服务
            elif data_read[i][3] in ["gallery"]:
                T_C[c][5] += 1
                test[5] += 1
            # 第9类为住房服务，如酒店
            elif data_read[i][3] in ["hotel"]:
                T_C[c][8] += 1
                test[8] += 1
            else:
                pass
        elif data_read[i][2] == "leisure":
            # 第6类为体育休闲服务
            if data_read[i][3] in ["park", "playground", "sports_centre", "fitness_centre", "swimming_pool"]:
                T_C[c][5] += 1
                test[5] += 1
            else:
                pass
        elif data_read[i][2] == "historic":
            # 第1类为旅游景区
            T_C[c][0] += 1
            test[0] += 1
        elif data_read[i][2] == "public_transport":
            # 第13类为交通运输
            if data_read[i][3] in ["station", "stop_position"]:
                T_C[c][12] += 1
                test[12] += 1
            else:
                pass
        elif data_read[i][2] == "office":
            # 第10类为政府及组织
            if data_read[i][3] in ["diplomatic", "government"]:
                T_C[c][9] += 1
                test[9] += 1
            # 第11类为企业
            elif data_read[i][3] in ["tax_advisor", "accountant", "company", "estate_agent", "lawyer", "insurance"]:
                T_C[c][10] += 1
                test[10] += 1
            else:
                pass
        elif data_read[i][2] == "healthcare":
            # 第2类为医疗卫生服务
            T_C[c][1] += 1
            test[1] += 1
        elif data_read[i][2] == "landuse":
            # 第4类为居住区
            if data_read[i][3] in ["residential"]:
                T_C[c][3] += 1
                test[3] += 1
            else:
                pass
        elif data_read[i][2] == "building":
            # 第4类为居住区
            if data_read[i][3] in ["apartment", "residential", "house"]:
                T_C[c][3] += 1
                test[3] += 1
            else:
                pass
        else:
            pass
    return T_C, test

# 读取NY的OPI数据
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/NYCBike/NYC_POI.csv', encoding="ISO-8859-1")
# 读取Washington的OPI数据
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Washington/Washington_POI.csv', encoding="ISO-8859-1")
# 读取Chicago的OPI数据
# data = pd.read_csv('E:/trans/DAN/MK-DAN/data/Chicago/Chicago_POI.csv', encoding="ISO-8859-1")
# 读取LA的OPI数据
data = pd.read_csv('E:/trans/DAN/MK-DAN/data/LABike/LA_POI.csv', encoding="ISO-8859-1")
selected_data = data.iloc[:, [0, 1, 2, 3]]
# 将读取到的数据转换成list形式
list_read = np.array(selected_data).tolist()
L_OPI, test = get_opi(list_read, l, Lat_Lon)
np.save('E:/trans/DAN/MK-DAN/data/LABike/LABike_poi.npy', L_OPI)
print(L_OPI)
print(type(L_OPI))
print(L_OPI.shape)
print(test)
