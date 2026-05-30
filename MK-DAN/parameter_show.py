import matplotlib.pyplot as plt
from matplotlib import font_manager

# ===== 中文字体 =====
chinese_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")

# ===== 横轴：dropout_rate =====
dropout_rates = [0, 0.10, 0.20, 0.30, 0.40]

# ===== RMSE 数据 =====
rmse_LA = [1.45, 1.49, 1.43, 1.32, 2.64]
rmse_CHI = [5.57, 5.24, 4.68, 4.03, 5.82]
rmse_WA = [7.32, 4.41, 4.33, 4.26, 8.06]
rmse_NY = [10.77, 11.53, 13.19, 10.93, 12.09]

# ===== MAE 数据 =====
mae_LA = [0.51, 0.52, 0.50, 0.48, 0.79]
mae_CHI = [2.95, 2.76, 2.53, 2.28, 3.01]
mae_WA = [3.61, 2.42, 2.38, 2.34, 4.29]
mae_NY = [5.26, 5.41, 6.14, 5.11, 5.73]

# ================= RMSE 图 =================
plt.figure()

plt.plot(dropout_rates, rmse_LA, marker='o', linestyle='--', label='LosAngeles')
plt.plot(dropout_rates, rmse_CHI, marker='o', linestyle='--', label='Chicago')
plt.plot(dropout_rates, rmse_WA, marker='o', linestyle='--', label='Washington')
plt.plot(dropout_rates, rmse_NY, marker='o', linestyle='--', label='NewYork')

plt.xlabel("dropout_rate", fontsize=16)
plt.ylabel("RMSE", fontsize=16)
plt.title("不同dropout_rate下RMSE变化", fontproperties=chinese_font, fontsize=18)

plt.tick_params(axis='both', labelsize=14)

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85), fontsize=14)

plt.grid()
plt.subplots_adjust(bottom=0.15)

plt.show()


# ================= MAE 图 =================
plt.figure()

plt.plot(dropout_rates, mae_LA, marker='o', label='LosAngeles')
plt.plot(dropout_rates, mae_CHI, marker='o', label='Chicago')
plt.plot(dropout_rates, mae_WA, marker='o', label='Washington')
plt.plot(dropout_rates, mae_NY, marker='o', label='NewYork')

plt.xlabel("dropout_rate", fontsize=16)
plt.ylabel("MAE", fontsize=16)
plt.title("不同dropout_rate下MAE变化", fontproperties=chinese_font, fontsize=18)

plt.tick_params(axis='both', labelsize=14)

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85), fontsize=14)

plt.grid()
plt.subplots_adjust(bottom=0.15)

plt.show()