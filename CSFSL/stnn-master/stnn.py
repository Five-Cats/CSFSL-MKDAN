import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import svm

from module import MLP
from utils import identity


class SaptioTemporalNN(nn.Module):
    def __init__(self, relations, nx, nt, nd, nz, mode=None, nhid=0, nlayers=1, dropout_f=0., dropout_d=0.,
                 activation='tanh', periode=1):
        super(SaptioTemporalNN, self).__init__()
        # 进行一个断言，确保nhid（神经网络的隐藏层神经元数量）和nlayers（神经网络的层数）的值是合理的
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        # 根据传入的activation参数设置神经网络的激活函数
        self.activation = F.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        # self.relations表示某种关系矩阵
        if mode is None or mode == 'refine':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1),
                                        torch.ones(nx, 1, nx).to(device)), 1)
        # 获取self.relations的第二维的大小，也就是它的列数
        self.nr = self.relations.size(1)
        # modules
        # 创建一个dropout层，dropout是一种正则化技术，用于防止过拟合
        self.drop = nn.Dropout(dropout_f)
        # 创建一个参数张量，这个张量的形状是(nt, nx, nz)
        self.factors = nn.Parameter(torch.Tensor(nt, nx, nz))
        # 创建一个多层感知机（MLP）网络
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        # self.dynamic = svm.SVC(kernel='rbf', gamma='scale')
        # 创建一个线性层
        self.decoder = nn.Linear(nz, nd, bias=False)
        # 若mode是'refine'，则self.rel_weights是一个标量；若mode是'discover'，则self.rel_weights是一个形状为(nx, 1, nx)的张量
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        # 调用一个初始化权重的方法
        self._init_weights(periode)

    # 定义初始化神经网络权重的方法
    # 使用随机值初始化神经网络的权重，同时根据输入的periode值和时间步来调整权重的初始化方式。
    def _init_weights(self, periode):
        # 定义初始化范围initrange为0.1，用于后续生成权重值的范围
        initrange = 0.1
        if periode >= self.nt:
            # 使用uniform_方法为self.factors数据创建一个在-initrange和initrange之间的均匀分布的随机值
            self.factors.data.uniform_(-initrange, initrange)
        else:
            # 创建一个时间序列timesteps
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                # 找出所有模periode等于当前时间步的索引
                idx = timesteps % periode == t
                # 用这些索引创建一个与self.factors形状相同的掩码idx_data
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                # 创建一个形状为(self.nx, self.nz)的张量，其中的值是从-initrange到initrange的均匀分布的随机值，然后重复这个张量，使其第一个维度的大小等于当前时间步的索引总数
                init = torch.Tensor(self.nx, self.nz).uniform_(-initrange, initrange).repeat(idx.sum().item(), 1, 1)
            # 使用masked_scatter_方法，它会将self.factors数据中对应idx_data位置的值替换为刚刚创建的随机值
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            # 所有的权重都被初始化为0.5
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            # 所有的权重都被初始化为1/self.nx
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        # self.mode可能表示处理这些关系的模式（例如，refine细化或discover发现）
        if self.mode is None:
            return self.relations
        else:
            # 使用F.hardtanh函数将self.rel_weights的值限制在0和1之间
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                # 创建一个新的关系矩阵，其中，intra是复制自self.relations[:, 0]的一个新的矩阵，并且增加了一个维度
                intra = self.rel_weights.new(self.nx, self.nx).copy_(self.relations[:, 0]).unsqueeze(1)
                # inter是一个全零矩阵
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1, self.nx)
                # inter中的元素会根据self.relations[:, 1:]和刚刚计算得到的权重weights进行填充
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                # 将self.relations[:, 0]增加一个维度后赋值给intra
                intra = self.relations[:, 0].unsqueeze(1)
                # inter直接等于权重weights
                inter = weights
            # 将intra和inter沿着第二个维度（索引为1）进行连接，并返回结果
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        # 使用矩阵乘法（matmul）将z与get_relations()得到的关系矩阵进行运算，结果被重塑成一个新的二维矩阵（-1表示自动计算其余维度的大小）
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    # 解码器
    def dec_closure(self, t_idx, x_idx):
        # 从self.factors中获取一个元素，创建一个dropout层防止过拟合
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    # 动态
    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        # 初始化z为self.factors中的最后一个元素
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        # 将z_gen列表转换为一个PyTorch张量
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    # 生成器函数通常用于迭代地返回一系列值
    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights
