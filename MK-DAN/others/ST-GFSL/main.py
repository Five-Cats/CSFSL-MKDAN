import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from maml import STMAML
from tqdm import tqdm

def train_epoch(train_dataloader):
    train_losses = []
    # 遍历 train_dataloader 中的每个批次数据（data, A_wave）。
    for step, (data, A_wave) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        # 将数据（A_wave 和 data）移动到指定的设备上（如GPU），并转换为浮点数格式。
        A_wave = A_wave.to(device=args.device)
        A_wave = A_wave.float()
        data = data.to(device=args.device)
        # 通过模型前向传播计算输出 out 和元数据图 meta_graph。
        out, meta_graph = model(data, A_wave)
        loss_predict = loss_criterion(out, data.y)
        loss_reconsturct = loss_criterion(meta_graph, A_wave)
        loss = loss_predict + loss_reconsturct
        loss.backward()
        optimizer.step()
        # print("loss_predict: {}, loss_reconsturct: {}".format(loss_predict.detach().cpu().numpy(), loss_reconsturct.detach().cpu().numpy()))
        train_losses.append(loss.detach().cpu().numpy())
    return sum(train_losses)/len(train_losses)

def test_epoch(test_dataloader):
    # 使用 torch.no_grad() 禁用梯度计算，以节省内存和计算资源。
    with torch.no_grad():
        model.eval()
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=args.device)
            data = data.to(device=args.device)
            # 通过模型前向传播计算输出 out（这里不需要元数据图，因此用 _ 忽略）。
            out, _ = model(data, A_wave)
            # 初始化 outputs 和 y_label 列表（仅在第一个批次时）。
            if step == 0:
                outputs = out
                y_label = data.y
            # 对于后续批次，使用 torch.cat 将输出和标签拼接起来。
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))
        # 将 outputs 和 y_label 从 PyTorch 张量转换为 NumPy 数组，并调整其维度（通常是为了匹配特定的数据格式需求）。
        # 返回测试数据的模型输出和真实标签，都是 NumPy 数组格式。
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label


parser = argparse.ArgumentParser(description='MAML-based')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='labike', type=str)
parser.add_argument('--source_epochs', default=200, type=int)
parser.add_argument('--source_lr', default=1e-2, type=float)
parser.add_argument('--target_epochs', default=120, type=int)
parser.add_argument('--target_lr', default=1e-2, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--meta_dim', default=16, type=int)
parser.add_argument('--target_days', default=3, type=int)
parser.add_argument('--model', default='GRU', type=str)
parser.add_argument('--loss_lambda', default=1.5, type=float)
parser.add_argument('--memo', default='revise', type=str)
args = parser.parse_args()

# print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim, "target_days = ", args.target_days)

if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")
    # 加载配置文件
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(7)
    # config 字典中包含了 data、task 和 model 三个部分，分别用于配置数据处理、任务设定和模型配置。
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    # model_args 更新了 meta_dim 和 loss_lambda 参数，这  两个参数的值来自 args
    model_args['meta_dim'] = args.meta_dim
    model_args['loss_lambda'] = args.loss_lambda
    # 这里创建了一个源数据集 source_dataset，该数据集使用了之前从配置文件中读取的 data_args 和 task_args 参数，并且指定了 source 数据集的标识。
    source_dataset = traffic_dataset(data_args, task_args, "source", test_data=args.test_dataset)

    model = STMAML(data_args, task_args, model_args, model=args.model).to(device=args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr)
    loss_criterion = nn.MSELoss()

    source_training_losses, target_training_losses = [], []
    best_result = ''
    min_MAE = 10000000
    start_time = time.time()
    for epoch in tqdm(range(args.source_epochs)):
        # Meta-Train
        # start_time = time.time()
        spt_task_data, spt_task_A, qry_task_data, qry_task_A = source_dataset.get_maml_task_batch(task_args['task_num'])
        loss = model.meta_train_revise(spt_task_data, spt_task_A, qry_task_data, qry_task_A)

        # loss = model.meta_train(spt_task_data, spt_task_A, qry_task_data, qry_task_A)
        end_time = time.time()
        if epoch % 20 == 0:
            print("[Source Train] epoch #{}/{}: loss is {}, training time is {}".format(epoch+1, args.source_epochs, loss, (end_time-start_time)/60))

    print("Source dataset meta-train finish.")

    target_dataset = traffic_dataset(data_args, task_args, "target", test_data=args.test_dataset, target_days=args.target_days)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    model.finetuning(target_dataloader, test_dataloader, args.target_epochs)
    print(args.memo)
    