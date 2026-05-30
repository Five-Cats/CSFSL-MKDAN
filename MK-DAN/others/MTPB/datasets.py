from enum import EnumMeta
from select import select
import torch
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
from utils import *
import random
random.seed(7)
# 自定义异常类
class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class traffic_dataset(Dataset):
    def __init__(self, data_args, task_args, data_list=None, stage='source', test_data='metr-la', add_target=True, target_days=3):
        super(traffic_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.predefined_data_list = data_list
        # if(self.stage == 'pretrain' or self.stage == 'cluster'):
        #     self.add_target = False
        self.load_data(stage, test_data)
        
        print("[INFO] Dataset init finished!")


    # according to the stage, output x_list and y_list, both of them are dict
    def load_data(self, stage, test_data):
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}
        self.batchnum_list = {}
        
        data_keys = np.array(self.data_args['data_keys'])
        if(self.predefined_data_list != None):
            data_keys = self.predefined_data_list
            if(self.add_target):
                data_keys += [self.test_data]
                
        if stage == 'source' or stage == 'pretrain' or self.stage == 'cluster' or self.stage == 'source_train':
            # self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            self.data_list = data_keys
            # self.data_list = np.array(['metr-la', 'chengdu_m'])
        elif stage == 'target' or stage == 'target_maml':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        else:
            print("stage is : {}".format(stage))
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))

        
        for dataset_name in self.data_list:
            print("dataset_name : {}".format(dataset_name))
            # 加载邻接矩阵A，计算归一化形式
            A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
            # 构造PyG的edge_index
            edge_index, edge_attr, node_feature = self.get_attr_func(
            self.data_args[dataset_name]['adjacency_matrix_path']
            )

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            # 加载交通数据，shape: [L, N, 4] -> 转置为 [N, D, L]
            X = np.load(self.data_args[dataset_name]['dataset_path'])
            # [L, N, 4]
            # [:,:,0] : speed, [:,:,1] : some symbol of time?
            # (N, D, L)
            
            X = X.transpose((1, 2, 0))
            X = torch.tensor(X, dtype=torch.double)
            
            # [N, 2, L]
            # 提取speed和time特征
            X = torch.cat((X[:,0, :].unsqueeze(1), X[:,-1,:].unsqueeze(1)), dim = 1)
            
            # Interpolation. Chengdu and Shenzhen interpolated to 5min level.
            # 插值处理
            interp = False
            if(dataset_name in ['chengdu_m', 'shenzhen', 'nycbike', 'washington', 'chicago', 'labike']):
                interp = True
            
            if(interp):
                interp_X = torch.nn.functional.interpolate(X, size = 2 * X.shape[-1] - 1,mode='linear',align_corners=True)  # 将时间长度从 L → 2L - 1（5分钟间隔）
                # inter_speed.squeeze_(1)
                interp_X = torch.cat((interp_X[:,:,:1],interp_X),dim=-1)  # 插值后，第一帧数据（时间为 0）可能丢失，因此下一步需要计算原来的第一个时间步，并将其复制并拼接在最前面
                interp_X[:,1,0] = ((interp_X[:,1,1] - 1) + 2016 ) % 2016 # 2016 is the week slot（2016 = 一周 7 天 × 24h × 12 个 5min）,通过第二个时间步的值（interp_X[:,1,1]）反推第一个时间步的值
                X = interp_X
            # 对交通数据中的 speed（车速）特征进行归一化处理
            X = X.numpy()
            # mean and std 
            X[:,0,:] = X[:,0,:].astype(np.float64)
            print(X.shape)
            # 只对speed特征做归一化
            means = np.expand_dims(np.mean(X[:,0,:]),0)
            X[:,0,:] = X[:,0,:] - means.reshape(1, -1, 1)
            stds = np.expand_dims(np.std(X[:,0,:]),0)
            self.means_list[dataset_name], self.stds_list[dataset_name] = means, stds
            X[:,0,:] = X[:,0,:] / stds.reshape(1, -1, 1)

            # [N, 2, L] and 0 is normalized
            if stage == 'source' or stage == 'dann' or stage == 'pretrain' or stage == 'source_train':
                if(dataset_name == self.test_data):
                    X = X[:, :, :288 * self.target_days]   # 测试集只保留 target_days 数据（如前3天）
                else:
                    X = X  # 源域数据不裁剪，保留所有时间步

            # target, small sample to finetune, 288 = 24 * 12 is one day data.
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]  # 目标域微调时，仅使用前 target_days 天（如3天）
                
            # test, choose rest of data
            elif stage == 'test':
                X = X[:, :, 288 * self.target_days:]  # 测试阶段用后面的所有数据

            # X : [N, 2, L]
            # 如果当前阶段是 cluster（例如用于 pattern 聚类），只保留输入 X，不生成预测目标 y，直接跳过。
            if(self.stage == 'cluster'):
                self.x_list[dataset_name] = X
                self.y_list[dataset_name] = []
                continue
                
            # else:           
            his_num = self.task_args['his_num']  # 历史时间步数（输入序列长度）
            pred_num = self.task_args['pred_num']  # 未来预测步数（输出序列长度）
            
            # inter_step 控制滑窗的“步长”
            if(self.stage == 'pretrain'):
                # gap 1 day
                inter_step = 12 * 3  # 每隔 3 小时采一个样本（288 / 8）
                # inter_step = 1 
            elif(self.stage == 'source_train'):
                inter_step = 12 * 6  # 每隔 6 小时采一个样本（减少冗余）
            elif(self.stage == 'target_maml'):
                inter_step = 12  # 每隔 1 小时采一个样本（采样更密集）
            else:
                inter_step = 12  # 默认每小时采一个样本
            # x, y : [num_samples, num_vertices, L, D]
            # x, y : [B, N, L, D]
            # 采用滑动窗口方式生成输入输出对，构造样本对 x, y
            x_inputs, y_outputs = generate_dataset(X, his_num, pred_num, means, stds, inter_step)
            print('{} : x shape : {}, y shape : {}'.format(dataset_name, x_inputs.shape, y_outputs.shape))
            self.x_list[dataset_name] = x_inputs  # [B, N, his_num, D]，即 B 个样本，每个样本包含 N 个节点、his_num 个历史步、D=2 个特征
            self.y_list[dataset_name] = y_outputs  # [B, N, pred_num]，即对应每个节点未来 pred_num 个时间步的预测目标值（只预测 speed）
        
        
        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            self.pretrain_batchnum = 0
            batch_size = self.task_args['batch_size']
            for dataset_name in self.data_list:
                this_data_total_batches = int(self.x_list[dataset_name].shape[0] // batch_size)
                self.batchnum_list[dataset_name] = this_data_total_batches
                self.pretrain_batchnum += this_data_total_batches
                
            self.pretrain_which_data = torch.zeros((self.pretrain_batchnum))
            self.pretrain_which_pos = torch.zeros((self.pretrain_batchnum))
            cur = 0
            for idx, dataset_name in enumerate(self.data_list):
                self.pretrain_which_data[cur : cur + self.batchnum_list[dataset_name]] = int(idx)
                self.pretrain_which_pos[cur : cur + self.batchnum_list[dataset_name]] = torch.arange(cur, cur + self.batchnum_list[dataset_name]) - cur
                cur += self.batchnum_list[dataset_name]
            self.random_permutation =torch.randperm(self.pretrain_batchnum)

    # 构造edge_index
    # 从邻接矩阵提取非零项的边，构造 edge_index，用于 GNN 建模
    def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        # edge_feature_matrix = np.load(edge_feature_matrix_path)
        # node_feature = np.load(node_feature_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a, b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass


    # index the dataset
    # used in finetune and testing
    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

        # if 'pretrain', randomly choose a city and randome cut a window
        # here data is [N, 2, L]
        
        ################
        ## pretrain, random return a batch size data [B, N, 2016, 2]
        # if(self.stage == 'pretrain'):
        #     select_dataset = random.choice(self.data_list)
        #     batch_size = self.task_args['batch_size']
            
        #     permutation = torch.randperm(self.x_list[select_dataset].shape[2] - self.his_num)
        #     indices = permutation[0:batch_size]
        #     slices = list(zip(indices,indices + self.his_num))
            
        #     # [B, N, L, 2]
        #     x_data = [torch.tensor(self.x_list[select_dataset][:,:,x:y]).unsqueeze(0) for x, y in slices]
        #     x_data = torch.cat(x_data,axis=0)
        #     x_data = x_data.permute((0,1,3,2))
            
        #     y_data = torch.tensor([])
            
        ################
        ## pretrain, random iterate all dataset
        # if(self.stage == 'pretrain'):
        #     batch_size = self.task_args['batch_size']
        #     idx = self.random_permutation[index] * batch_size
        #     for ii, enumerate_dataset in enumerate(self.data_list):
        #         dataset_length = self.x_list[enumerate_dataset].shape[2] - self.his_num
        #         if(idx >= dataset_length):
        #             idx -= dataset_length
        #         else:
        #             select_dataset = enumerate_dataset
            
        #     # select_dataset = random.choice(self.data_list)
        #     # permutation = torch.randperm(self.x_list[select_dataset].shape[2] - self.his_num)
            
        #     indices = torch.tensor(list(range(idx,idx+batch_size)))
        #     slices = list(zip(indices,indices + self.his_num))
            
        #     # [B, N, L, 2]
        #     x_data = [torch.tensor(self.x_list[select_dataset][:,:,x:y]).unsqueeze(0) for x, y in slices]
        #     x_data = torch.cat(x_data,axis=0)
        #     x_data = x_data.permute((0,1,3,2))
            
        #     y_data = torch.tensor([])
        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            # need query *batch_size* continuous batches
            
            idx = self.random_permutation[index]
            # print('out index : {}, in index: {}'.format(index, idx))
            select_dataset = self.data_list[self.pretrain_which_data[idx].detach().cpu().numpy().astype(int)]
            pos = self.pretrain_which_pos[idx].detach().cpu().numpy().astype(int)
            batch_size = self.task_args['batch_size']
            # print('idx : {}, select_dataset : {}, pos : {}'.format(idx, select_dataset, pos))
            indices = torch.tensor(list(range(pos,pos+batch_size)))
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        # if 'source', randomly choose a city and random choose a batch
        elif (self.stage == 'source'):
            select_dataset = random.choice(self.data_list)
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        # if 'target_maml', choose the first city and randomly choose a batch
        else:
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        x_data = x_data.float()
        y_data = y_data.float()
        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data,means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        
        # x_data is [batch, n, HisStep, D], y_data is [batch, n, HisStep]
        # last, return data_i is a torch.geometric.data, contains x, y, edge index, which dataset
        # A_wave contains a adjacent matrix. Used to make reconstruction loss
        return data_i, A_wave
    
    
    # maml task, used in source training
    # each task is a graph. some batch of data on a graph
    # 元学习任务采样（用于meta-train）
    def get_maml_task_batch(self, task_num):
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        # first choose a random dataset
        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        # equally distribute support set and qry set
        # 随机从某城市中采样task_num × 2个batch
        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            # data_i.edge_attr = self.edge_attr_list[select_dataset]
            # data_i.node_feature = self.node_feature_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            # 每对奇偶交替为 spt/qry，支持 Reptile 或 MAML 框架
            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())
            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave
    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        if self.stage == 'pretrain' or self.stage == 'source_train':
            # print("pretrain use datasets of {} cities".format(self.data_list))
            # [L, N, 2016, 2]
            return self.pretrain_batchnum
        if self.stage == 'target_maml' or self.stage == 'test':
            return int(self.x_list[self.data_list[0]].shape[0] //  self.task_args['batch_size'])
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length



if __name__ == "__main__":
    
    
# # ----------------------- #
# #    test code(pretrain)
# # ----------------------- #
#     import yaml
#     with open('config.yaml') as f:
#             config = yaml.load(f)        

#     mydataset = traffic_dataset(config['data'], config['task']['mae'], stage='pretrain', test_data='metr-la',add_target=False)


#     train_batch_num = 10

#     for i in range(train_batch_num):
#         data, A_wave = mydataset[i]
#         print("node_num:{}, edge_index:{}, x:{}, A_wave:{}".format(data.node_num, data.edge_index.shape, data.x.shape, A_wave.shape))
        


#     # ----------------------- #
#     #    test code(source)
#     # ----------------------- #
#     import yaml
#     with open('config.yaml') as f:
#             config = yaml.load(f)        

#     mydataset = traffic_dataset(config['data'], config['task']['maml'], stage='source', test_data='metr-la')

#     train_batch_num = 10

#     for i in range(train_batch_num):
#         data, A_wave = mydataset[i]
#         print("node_num:{}, edge_index:{}, x:{}, y:{}, A_wave:{}".format(data.node_num, data.edge_index.shape, data.x.shape, data.y.shape, A_wave.shape))
    


# ----------------------- #
#    test code(test)
# ----------------------- #

    import yaml

    with open('configs/config.yaml') as f:
            config = yaml.load(f)   
    data_list = "chengdu_shenzhen_metr"
    test_dataset = 'pems-bay'     
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    finetune_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'target_maml', test_data=test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'test', test_data=test_dataset)

    print("length of dataset is", len(finetune_dataset))
    print("length of dataset is", len(test_dataset))

    print('finetune dataset')
    for idx in range(len(finetune_dataset)):
        data, A_wave = finetune_dataset[idx]
        print(idx, data, A_wave)
        print("node_num is {}, x_data shape is {}, y_data shape is {}".format(data.node_num, data.x.shape, data.y.shape))
        print("A_wave shape is", A_wave.shape)
    
    print('test_dataset')
    
    for idx in range(len(test_dataset)):
        data, A_wave = test_dataset[idx]
        print(idx, data, A_wave)
        print("node_num is {}, x_data shape is {}, y_data shape is {}".format(data.node_num, data.x.shape, data.y.shape))
        print("A_wave shape is", A_wave.shape)

