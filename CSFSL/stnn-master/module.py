import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout):
        super(MLP, self).__init__()
        # 定义输入层大小
        self.ninp = ninp
        # modules
        if nlayers == 1:
            self.module = nn.Linear(ninp, nout)
        # 此时网络有多个层，每一层的结构是：线性层（nn.Linear）→ReLU激活函数→Dropout层（用于防止过拟合）
        else:
            modules = [nn.Linear(ninp, nhid), nn.ReLU(), nn.Dropout(dropout)]
            nlayers -= 1
            # 除了最后一层外，每一层的输入和输出大小都是nhid
            while nlayers > 1:
                modules += [nn.Linear(nhid, nhid), nn.ReLU(), nn.Dropout(dropout)]
                nlayers -= 1
            # 最后一层的输入大小是nhid，输出大小是nout
            modules.append(nn.Linear(nhid, nout))
            # nn.Sequential可以将一系列的神经网络层组合成一个神经网络模型，按照顺序逐层进行计算
            self.module = nn.Sequential(*modules)

    # 直接将输入传递到前面定义的模块中，得到输出结果
    def forward(self, input):
        return self.module(input)
