import os
import random
import json
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

import configargparse
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from datasets import dataset_factory
from utils import DotDict, Logger, rmse
from stnn import SaptioTemporalNN

# 定义存储损失的数组
L = []
# 定义存储RMSE的数组
R = []

#######################################################################################################################
# Options - CUDA - Random seed
#######################################################################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')  # 数据集路径
p.add('--dataset', type=str, help='dataset name', default='heat')  # 数据集名称
# -- xp
p.add('--outputdir', type=str, help='path to save xp', default='output')  # 保存实验结果的路径
p.add('--xp', type=str, help='xp name', default='stnn')  # 实验名称
# -- model
p.add('--mode', type=str, help='STNN mode (default|refine|discover)', default='default')  # stnn模式
p.add('--nz', type=int, help='laten factors size', default=1)  # 潜在因素大小
p.add('--activation', type=str, help='dynamic module activation function (identity|tanh)', default='identity')  # 动态模块激活函数
p.add('--khop', type=int, help='spatial depedencies order', default=1)  # 空间依赖关系阶数
p.add('--nhid', type=int, help='dynamic function hidden size', default=0)   # 动态函数隐藏层大小
p.add('--nlayers', type=int, help='dynamic function num layers', default=1)  # 动态函数层数
p.add('--dropout_f', type=float, help='latent factors dropout', default=.0)  # 潜在因素dropout的比例
p.add('--dropout_d', type=float, help='dynamic function dropout', default=.0)  # 动态函数dropout的比例
p.add('--lambd', type=float, help='lambda between reconstruction and dynamic losses', default=.1)  # 重构损失和动态损失之间的权重系数
# -- optim
p.add('--lr', type=float, help='learning rate', default=3e-3)  # 学习率
p.add('--beta1', type=float, default=.0, help='adam beta1')  # Adam优化器的beta1参数
p.add('--beta2', type=float, default=.999, help='adam beta2')  # Adam优化器的beta2参数
p.add('--eps', type=float, default=1e-9, help='adam eps')  # Adam优化器的epsilon参数
p.add('--wd', type=float, help='weight decay', default=1e-6)  # 权重衰减系数
p.add('--wd_z', type=float, help='weight decay on latent factors', default=1e-7)  # 潜在因素上的权重衰减系数
p.add('--l2_z', type=float, help='l2 between consecutives latent factors', default=0.)  # 潜在因素之间的L2正则化系数
p.add('--l1_rel', type=float, help='l1 regularization on relation discovery mode', default=0.)  # 关系发现模式的L1正则化系数
# -- learning
p.add('--batch_size', type=int, default=1000, help='batch size')  # 批处理大小
p.add('--patience', type=int, default=150, help='number of epoch to wait before trigerring lr decay')  # 提前触发学习率下降的epoch数量
p.add('--nepoch', type=int, default=10000, help='number of epochs to train for')  # 训练的总epoch数
# -- gpu
p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
# -- seed
p.add('--manualSeed', type=int, default=2021, help='manual seed')

# parse
opt = DotDict(vars(p.parse_args()))
opt.mode = opt.mode if opt.mode in ('refine', 'discover') else None

# cudnn
# 若opt.device大于-1，代码将在指定的GPU上运行，否则代码将在CPU上运行
if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# seed
# 设置随机种子，若opt.manualSeed是None，种子将被设置为1到10000之间的一个随机整数
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
# 若设备号大于-1（即选择了GPU作为运行设备），那么就在所有的GPU上设置随机种子。这是为了确保在GPU上运行的随机过程也能复现
if opt.device > -1:
    torch.cuda.manual_seed_all(opt.manualSeed)


#######################################################################################################################
# Data
#######################################################################################################################
# -- load data
setup, (train_data, test_data), relations = dataset_factory(opt.datadir, opt.dataset, opt.khop)
# .to(device)用于将数据或模型移动到指定的设备上
train_data = train_data.to(device)
test_data = test_data.to(device)
relations = relations.to(device)
# 更新opt字典，用setup字典中的值来覆盖opt字典中对应的值
for k, v in setup.items():
    opt[k] = v

# -- train inputs
# 创建两个一维的整数张量，分别表示时间索引和特征索引
t_idx = torch.arange(opt.nt_train, out=torch.LongTensor()).unsqueeze(1).expand(opt.nt_train, opt.nx).contiguous()
x_idx = torch.arange(opt.nx, out=torch.LongTensor()).expand_as(t_idx).contiguous()
# dynamic
# 将这两种索引组合成二维的索引张量
# idx_dyn是动态部分的索引，由t_idx[1:]和x_idx[1:]组成，即从第二个时间步和第二个特征开始的索引
idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
nex_dyn = idx_dyn.size(1)
# decoder
# idx_dec是解码器的索引，由t_idx和x_idx组成，即包含全部时间步和全部特征的索引
idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
nex_dec = idx_dec.size(1)
# nex_dyn和nex_dec分别计算并存储了idx_dyn和idx_dec的第二维的大小（即列数）

#######################################################################################################################
# Model
#######################################################################################################################
model = SaptioTemporalNN(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers,
                         opt.dropout_f, opt.dropout_d, opt.activation, opt.periode).to(device)


#######################################################################################################################
# Optimizer
#######################################################################################################################
params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
          {'params': model.dynamic.parameters()},
          {'params': model.decoder.parameters()}]
# 每个参数集合都配有一个权重衰减（weight decay），这是一种正则化技术用于防止过拟合
if opt.mode in ('refine', 'discover'):
    params.append({'params': model.rel_parameters(), 'weight_decay': 0.})
optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
# 如果opt.patience大于0，代码会创建一个学习率调度器。这个调度器会在训练过程中，每过opt.patience个epoch就观察一次验证损失，如果连续10个epoch（默认）验证损失都没有改善，就会降低学习率。这种策略叫做ReduceLROnPlateau
if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


#######################################################################################################################
# Logs
#######################################################################################################################
# 创建一个日志记录器（Logger）对象
logger = Logger(opt.outputdir, opt.xp, 100)
# 打开（或创建）一个名为'config.json'的文件，该文件位于opt.outputdir/opt.xp目录下。'w'参数表示文件以写入模式打开，如果文件已经存在，其内容将被覆盖
with open(os.path.join(opt.outputdir, opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)


#######################################################################################################################
# Training
#######################################################################################################################
lr = opt.lr
# 生成一个从0到'opt.nepoch'的序列
pb = trange(opt.nepoch)
for e in pb:
    # ------------------------ Train ------------------------
    model.train()
    # --- decoder ---
    # 生成一个随机排列的整数张量，长度为nex_dec，并将它转移到指定的设备（CPU或GPU）上
    idx_perm = torch.randperm(nex_dec).to(device)
    # 将上一步生成的随机排列的整数张量分割成多个大小为opt.batch_size的子张量，存储在batches列表中。这样做是为了在训练过程中分批次处理数据
    batches = idx_perm.split(opt.batch_size)
    # 创建一个defaultdict类型的字典logs_train，如果尝试访问一个不存在的键，它将返回一个浮点数类型的默认值
    logs_train = defaultdict(float)
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        # data
        # 获取时间和特征索引
        input_t = idx_dec[0][batch]
        input_x = idx_dec[1][batch]
        x_target = train_data[input_t, input_x]
        # closure
        # 调用模型的解码器（decoder）函数，将输入数据传递进去，并获取输出
        x_rec = model.dec_closure(input_t, input_x)
        # 计算模型输出和目标值之间的均方误差损失
        mse_dec = F.mse_loss(x_rec, x_target)
        # backward
        # 执行反向传播算法，计算损失函数关于模型参数的梯度
        mse_dec.backward()
        # step
        # 根据上一步计算得到的梯度，更新模型的参数
        optimizer.step()
        # log
        # 将当前迭代的损失值记录到日志中
        logger.log('train_iter.mse_dec', mse_dec.item())
        # 将当前批次的损失值累加到logs_train字典中对应的键上
        logs_train['mse_dec'] += mse_dec.item() * len(batch)
    # --- dynamic ---
    idx_perm = torch.randperm(nex_dyn).to(device)
    batches = idx_perm.split(opt.batch_size)
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        # data
        input_t = idx_dyn[0][batch]
        input_x = idx_dyn[1][batch]
        # closure
        # 从模型中获取因子（可能是某种预测或计算的结果）
        z_inf = model.factors[input_t, input_x]
        # 调用模型的动态闭包函数，将输入数据传递进去，并获取输出
        z_pred = model.dyn_closure(input_t - 1, input_x)
        # loss
        # 计算模型输出和因子之间的均方误差损失
        mse_dyn = z_pred.sub(z_inf).pow(2).mean()
        loss_dyn = mse_dyn * opt.lambd
        # 若opt.l2_z > 0，则计算L2正则化损失，并将其加到总损失中
        if opt.l2_z > 0:
            loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
        # 若满足下面条件，则计算L1关系损失，并将其加到总损失中。
        if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
            # rel_weights_tmp = model.rel_weights.data.clone()
            loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
        # backward
        loss_dyn.backward()
        # step
        optimizer.step()
        # clip
        # if opt.mode == 'discover' and opt.l1_rel > 0:  # clip
        #     sign_changed = rel_weights_tmp.sign().ne(model.rel_weights.data.sign())
        #     model.rel_weights.data.masked_fill_(sign_changed, 0)
        # log
        # 将当前迭代的均方误差损失值记录到日志中
        logger.log('train_iter.mse_dyn', mse_dyn.item())
        # 将当前批次的损失值累加到logs_train字典中对应的键上
        logs_train['mse_dyn'] += mse_dyn.item() * len(batch)
        logs_train['loss_dyn'] += loss_dyn.item() * len(batch)
        # 整个循环结束后，logs_train字典将包含每个批次的损失值之和，可以用于后续的训练统计和分析
    # --- logs ---
    # 计算解码器部分的平均均方误差损失
    logs_train['mse_dec'] /= nex_dec
    # 计算动态部分的平均均方误差损失
    logs_train['mse_dyn'] /= nex_dyn
    # 动态部分的平均总损失
    logs_train['loss_dyn'] /= nex_dyn
    # 计算解码器部分和动态部分的平均损失之和，并将其存储到logs_train字典中键为'loss'的位置
    logs_train['loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
    # 将logs_train字典记录到日志中，键名为'train_epoch'。这样可以在训练过程中跟踪每个epoch的损失值
    logger.log('train_epoch', logs_train)
    # ------------------------ Test ------------------------
    model.eval()
    # torch.no_grad()是PyTorch中的一个上下文管理器，它禁用了在其管理的代码块中的梯度计算，主要用于推理阶段，以减少内存使用量
    with torch.no_grad():
        # 生成预测结果
        x_pred, _ = model.generate(opt.nt - opt.nt_train)
        # 计算预测结果x_pred和测试数据test_data之间的RMSE。第一次计算时，reduce参数设置为False，所以返回的是每个样本的RMSE，而第二次计算返回的是所有样本的RMSE
        score_ts = rmse(x_pred, test_data, reduce=False)
        score = rmse(x_pred, test_data)
    logger.log('test_epoch.rmse', score)
    logger.log('test_epoch.ts', {t: {'rmse': scr.item()} for t, scr in enumerate(score_ts)})
    # checkpoint
    logger.log('train_epoch.lr', lr)
    loss = logs_train['loss']
    # 设置进度条的后缀，显示的内容包括训练损失和测试RMSE
    L.append(loss)
    R.append(score)
    # print('epoch:%d, loss=%.3f, rmse_test=%.2f' % (e, loss, score))
    pb.set_postfix(loss=logs_train['loss'], rmse_test=score)
    # 保存模型的状态
    logger.checkpoint(model)
    # schedule lr
    # 若RMSE小于1，且设置了学习率衰减的patience，那么就会根据当前的RMSE调整学习率
    if opt.patience > 0 and score < 1:
        lr_scheduler.step(score)
    # 获取当前的学习率
    lr = optimizer.param_groups[0]['lr']
    if lr <= 1e-5:
        break
logger.save(model)

#######################################################################################################################
# Painting
#######################################################################################################################

# 创建第一个折线图
plt.plot(L, label='loss')
plt.title('Line Chart of Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.xlim((0, 6000))
# plt.ylim((0, 30000))
plt.legend()
plt.show()

# 创建第二个折线图
plt.plot(R, label='RMSE')
plt.title('Line Chart of RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
# plt.xlim((0, 6000))
# plt.ylim((0, 10000))
plt.legend()
plt.show()