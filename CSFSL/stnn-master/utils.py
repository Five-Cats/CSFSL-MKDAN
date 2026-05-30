import os
import json
from collections import defaultdict

import torch


def rmse(x_pred, x_target, reduce=True):
    if reduce:
        # 返回所有样本的RMSE的平均值(这是通过从每个样本中计算RMSE，然后取平均值得到的)
        return x_pred.sub(x_target).pow(2).sum(-1).sqrt().mean().item()
    # 返回一个张量，其中包含每个样本的RMSE
    return x_pred.sub(x_target).pow(2).sum(2).sqrt().mean(1).squeeze()

# 实现矩阵的行归一化,即每行的元素和都为1
def normalize(mx):
    """Row-normalize matrix"""
    # 计算矩阵mx每一行的和
    rowsum = mx.sum(1)
    r_inv = 1 / rowsum
    # 将r_inv中所有无穷大的值(即那些在分母为0的情况下产生的)替换为0(避免在后续计算中产生NaN值)
    r_inv[r_inv == float('Inf')] = 0.
    # 创建一个对角矩阵，其对角线元素是r_inv中的元素，这个矩阵就是mx的归一化矩阵
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.matmul(mx)
    return mx


def identity(input):
    return input


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    # 访问字典中不存在的键时，它会返回None
    __getattr__ = dict.get
    # 设置DotDict的一个属性时，实际上是在设置字典的一个键值对
    __setattr__ = dict.__setitem__
    # 删除DotDict的一个属性时，实际上是在删除字典的一个键值对
    __delattr__ = dict.__delitem__


class Logger(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger, self).__init__()
        os.makedirs(os.path.join(log_dir, name))
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'model.pt')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)
