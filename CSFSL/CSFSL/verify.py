import torch
from datasets import dataset_factory
from utils import rmse
import NGSIM_data222

# 加载模型
model = torch.load('E://CSFSL/STNN_NGSIM/output1/stnn/model.pt')
# 确保模型处于评估模式
# model.eval()
# 读取新的交通流量数据（假设数据为一个.csv文件，包含日期、时间、地点和流量数据）
data_1 = NGSIM_data222.LIST_Num[1]
relation_1 = NGSIM_data222.LIST_Relation[1]
# 从datasets中的dataset_factory函数中的heat函数来确定一些基本参数的值，如nt，nx，nd等
opt, (train_data, test_data), relations = dataset_factory(data_1, relation_1, 100, k=1)

with torch.no_grad():
    # 生成预测结果
    x_pred, _ = model.generate(opt.nt - opt.nt_train)
    # print(x_pred)
    # 计算预测结果x_pred和测试数据test_data之间的RMSE。第一次计算时，reduce参数设置为False，所以返回的是每个样本的RMSE，而第二次计算返回的是所有样本的RMSE
    score_ts = rmse(x_pred, test_data, reduce=False)
    score = rmse(x_pred, test_data)
print(score)