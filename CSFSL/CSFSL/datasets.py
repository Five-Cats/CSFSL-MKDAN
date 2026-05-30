import os
import NGSIM_data222
import numpy as np
import torch
from utils import DotDict, normalize

# data_0 = NGSIM_data222.LIST_Num[0]
# relation_0 = NGSIM_data222.LIST_Relation[0]

def dataset_factory(data_0, relation_0, T_d, k=1):
    # 获取数据集
    opt, data, relations = data_processing(data_0, relation_0, T_d)  # 从heat中获取数据
    # else:
    #     raise ValueError('Non dataset.')
    # make k hop
    new_rels = [relations]
    for n in range(k - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    # split train / test  分割训练集和测试集
    train_data = data[:opt.nt_train]
    test_data = data[opt.nt_train:]
    return opt, (train_data, test_data), relations


def data_processing(data, relation, train):
    # dataset configuration
    opt = DotDict()
    opt.nt = len(data)
    # 前train个为训练集
    opt.nt_train = train
    opt.nx = len(data[0])
    opt.nd = 1
    opt.periode = len(data)
    # loading data
    data = torch.Tensor(data).view(opt.nt, opt.nx, opt.nd)
    # load relations
    relations = torch.Tensor(relation)
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations
