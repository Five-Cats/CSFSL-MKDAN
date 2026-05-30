import time
import numpy as np
import torch
import dgl
import copy
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from dgl.nn.pytorch import GATConv, GraphConv
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999


# ========== 归一化函数 ==========
# 对数据进行归一化处理
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    norm_data = (data - min_val) / (max_val - min_val + 1e-6)
    return norm_data, max_val, min_val

# ========== 数据加载模块 ==========
class TrafficDataset(Dataset):
    def __init__(self, data, lookback, pred_len, start_date="2023-04-01 00:00:00", timestep_min=20, idx=None, augment=True):
        """
            data: 原始数据 (T, N, F)，三维numpy数组 (timesteps, nodes, features)
            lookback: 输入窗口长度
            pred_len: 预测长度
            start_date: 数据起始时间（已知4月1日是周六）
            timestep_min: 时间步长(分钟)
            idx: 指定使用的样本索引列表
        """
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len
        self.timestep = timedelta(minutes=timestep_min)
        self.time_points = [datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")]
        self.augment = augment
        # 生成时间序列
        for _ in range(1, len(data)):
            self.time_points.append(self.time_points[-1] + self.timestep)
        self.indices = idx if idx is not None else np.arange(len(data) - lookback - pred_len + 1)

    def __len__(self):
        return len(self.indices)
    
    def _augment_data(self, x):
        if not self.augment:
            return x
            
        # 时间掩码
        if np.random.random() < 0.3:
            mask_ratio = np.random.uniform(0.1, 0.2)
            mask = np.random.random(x.shape) < mask_ratio
            x[mask] = 0
            
        # 高斯噪声
        if np.random.random() < 0.5:
            noise_std = np.random.uniform(0.005, 0.015)
            noise = torch.randn_like(x) * noise_std
            x = x + noise
            
        # 时间偏移
        if np.random.random() < 0.2:
            shift = np.random.randint(-1, 2)
            x = torch.roll(x, shifts=shift, dims=0)
            
        return x

    def __getitem__(self, i):
        idx = self.indices[i]
        # 原始数据 (lookback, 36, 1)
        x = self.data[idx:idx + self.lookback]
        # 时间特征 (lookback, 36, 3)
        time_feat = self._get_time_features(idx)
        # 合并特征 (lookback, 36, 4)
        enhanced_x = np.concatenate(
            [x[..., :1],  # 原始value特征
            time_feat],  # 新增时间特征
            axis=-1)
        y = self.data[idx + self.lookback:idx + self.lookback + self.pred_len, :, 0:1]
        return torch.FloatTensor(enhanced_x).to(device), torch.FloatTensor(y).to(device)

    # 提取简化版时间特征矩阵 (lookback, 36, 3)
    def _get_time_features(self, idx):
        selected_times = self.time_points[idx:idx + self.lookback]
        time_features = []
        for t in selected_times:
            # 基础特征
            hour = t.hour + t.minute / 60  # 0-24连续值
            is_weekend = t.weekday() >= 5  # 周六=5, 周日=6
            time_features.append([
                hour / 24,  # 归一化到[0,1]
                t.weekday() / 7,  # 归一化星期几
                float(is_weekend)  # 是否周末
            ])
        return np.repeat(np.array(time_features)[:, np.newaxis, :], self.data.shape[1], axis=1)

# 支持多种图结构的多源数据加载器
class MultiSourceDataLoader:
    def __init__(self, source_cities, target_city, lookback=12, pred_len=6, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
        """
        Args:
            source_cities: 源城市列表 ['city1', 'city2']
            target_city: 目标城市 'target_city'
            lookback: 输入时间窗口长度
            pred_len: 预测步长
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
        """
        self.source_train_loaders = {}
        self.source_val_loaders = {}
        self.source_test_loaders = {}
        self.city_graphs = {}
        self.max_vals = {}  # 存储各城市最大值
        self.min_vals = {}  # 存储各城市最小值

        # 计算流量分布相似性矩阵
        def _compute_similarity_matrix(flow_data):
            # 归一化处理，计算每个时间步所有区域的总流量，添加1e-6防止除数为0
            norm_flows = flow_data / (flow_data.sum(axis=(1, 2), keepdims=True) + 1e-6)
            sim_matrix = np.zeros((36, 36))
            for i in range(36):
                for j in range(36):
                    sim_matrix[i, j] = np.exp(
                        -np.linalg.norm(norm_flows[:, i // 6, i % 6] - norm_flows[:, j // 6, j % 6]))
            return sim_matrix

        # 加载源城市数据
        for city in source_cities:
            # 加载基础数据
            traffic_data = np.load(f'data/{city}/dataset0.npy')
            # traffic_data = np.load(f'MyData10%/{city}/dataset_10%.npy')
            traffic_data, max_val, min_val = min_max_normalize(traffic_data)
            self.max_vals[city] = max_val
            self.min_vals[city] = min_val
            matrix_adj = np.load(f'data/{city}/{city}_matrix.npy')
            poi = np.load(f'data/{city}/{city}_poi.npy')
            source_flow = np.load(f'data/{city}/{city}_pickup.npy')  # (T,6,6)
            dest_flow = np.load(f'data/{city}/{city}_dropoff.npy')  # (T,6,6)
            # matrix_adj = np.load(f'MyData10%/{city}/{city}_matrix.npy')
            # poi = np.load(f'MyData10%/{city}/{city}_poi.npy')
            # source_flow = np.load(f'MyData10%/{city}/{city}_pickup.npy')  # (T,6,6)
            # dest_flow = np.load(f'MyData10%/{city}/{city}_dropoff.npy')  # (T,6,6)

            # 计算四种图结构
            poi_normalized = poi / (np.linalg.norm(poi, axis=1, keepdims=True) + 1e-6)
            poi_sim = poi_normalized @ poi_normalized.T
            # poi_sim = poi @ poi.T  # POI相似度
            source_sim = _compute_similarity_matrix(source_flow)  # 出发分布相似性
            dest_sim = _compute_similarity_matrix(dest_flow)  # 到达分布相似性
            self.city_graphs[city] = {
                'poi': poi,
                'pickup': source_flow,
                'dropoff': dest_flow,
                'matrix_adj': matrix_adj,
                'poi_sim': poi_sim,
                'source_sim': source_sim,
                'dest_sim': dest_sim
            }
            # 数据集划分
            n_samples = len(traffic_data) - lookback - pred_len + 1
            indices = np.arange(n_samples)
            # 分层划分（保持时间连续性）
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)

            train_dataset = TrafficDataset(traffic_data, lookback, pred_len, idx=indices[:train_end])
            val_dataset = TrafficDataset(traffic_data, lookback, pred_len, idx=indices[train_end:val_end])
            test_dataset = TrafficDataset(traffic_data, lookback, pred_len, idx=indices[val_end:])
            # 创建数据集
            self.source_train_loaders[city] = DataLoader(train_dataset, batch_size=64, shuffle=True)
            self.source_val_loaders[city] = DataLoader(val_dataset, batch_size=64)
            self.source_test_loaders[city] = DataLoader(test_dataset, batch_size=64)

        # 加载目标城市数据
        target_data = np.load(f'data/{target_city}/dataset0.npy')
        # target_data = np.load(f'MyData10%/{target_city}/dataset_10%.npy')
        target_data, max_val, min_val = min_max_normalize(target_data)
        self.max_vals[target_city] = max_val
        self.min_vals[target_city] = min_val
        matrix_adj = np.load(f'data/{target_city}/{target_city}_matrix.npy')
        poi = np.load(f'data/{target_city}/{target_city}_poi.npy')
        source_flow = np.load(f'data/{target_city}/{target_city}_pickup.npy')
        dest_flow = np.load(f'data/{target_city}/{target_city}_dropoff.npy')
        # matrix_adj = np.load(f'MyData10%/{target_city}/{target_city}_matrix.npy')
        # poi = np.load(f'MyData10%/{target_city}/{target_city}_poi.npy')
        # source_flow = np.load(f'MyData10%/{target_city}/{target_city}_pickup.npy')
        # dest_flow = np.load(f'MyData10%/{target_city}/{target_city}_dropoff.npy')

        # 计算目标城市的四种图
        poi_normalized = poi / (np.linalg.norm(poi, axis=1, keepdims=True) + 1e-6)
        poi_sim = poi_normalized @ poi_normalized.T
        source_sim = _compute_similarity_matrix(source_flow)
        dest_sim = _compute_similarity_matrix(dest_flow)
        self.city_graphs[target_city] = {
            'poi': poi,
            'pickup': source_flow,
            'dropoff': dest_flow,
            'matrix_adj': matrix_adj,
            'poi_sim': poi_sim,
            'source_sim': source_sim,
            'dest_sim': dest_sim
        }

        # 目标城市划分（保持相同比例）
        n_samples = len(target_data) - lookback - pred_len + 1
        indices = np.arange(n_samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_dataset = TrafficDataset(target_data, lookback, pred_len, idx=indices[:train_end])
        val_dataset = TrafficDataset(target_data, lookback, pred_len, idx=indices[train_end:val_end])
        test_dataset = TrafficDataset(target_data, lookback, pred_len, idx=indices[val_end:])

        self.target_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.target_val_loader = DataLoader(val_dataset, batch_size=32)
        self.target_test_loader = DataLoader(test_dataset, batch_size=32)

    # 获取指定城市的完整图数据
    def get_city_graph_data(self, city_name):
        return self.city_graphs[city_name]

    # 获取指定城市的归一化参数
    def get_norm_params(self, city_name):
        return self.max_vals[city_name], self.min_vals[city_name]

# ========== 模型架构模块 ==========
# 梯度反转层
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 保存 alpha 到 ctx（用于 backward 时用）
        ctx.alpha = alpha
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# 多视图图注意力网络
class MVGAT(nn.Module):
    def __init__(self, in_feats, h_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = h_dim // num_heads
        # 为每个图定义独立的GAT层
        self.gat_layers = nn.ModuleDict({
            'adj': GATConv(in_feats, head_dim, num_heads),
            'poi': GATConv(in_feats, head_dim, num_heads),
            'source': GATConv(in_feats, head_dim, num_heads),
            'dest': GATConv(in_feats, head_dim, num_heads)
        })
        self.view_proj = nn.ModuleDict({
            gtype: nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3)    # dropout防止过拟合
            )for gtype in ['adj', 'poi', 'source', 'dest']
        })

    def forward(self, graphs, feat):
        """
            graphs: 包含四个静态图结构的字典
            feat: 输入特征 (batch_size*seq_len, num_nodes, in_feats)
            views: 四种视图特征列表，每个形状为 (batch_size*seq_len, num_nodes, h_dim)
        """
        batch_size_seq, num_nodes, _ = feat.shape
        views = []
        for gtype in ['adj', 'poi', 'source', 'dest']:
            # 构建批次图
            single_graph = graphs[gtype]
            
            # 确保单个图的节点数量正确
            if single_graph.num_nodes() != num_nodes:
                print(f"Warning: Graph {gtype} has {single_graph.num_nodes()} nodes, expected {num_nodes}")
                # 如果节点数量不匹配，重新创建图
                if gtype == 'adj':
                    # 对于邻接矩阵图，创建完全连接图
                    src, dst = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
                    src, dst = src.flatten(), dst.flatten()
                    single_graph = dgl.graph((src, dst), num_nodes=num_nodes).to(single_graph.device)
                    single_graph = dgl.add_self_loop(single_graph)
                else:
                    # 对于其他图，创建自环图
                    src = dst = np.arange(num_nodes)
                    single_graph = dgl.graph((src, dst), num_nodes=num_nodes).to(single_graph.device)
                    single_graph = dgl.add_self_loop(single_graph)
            
            # 复制图结构以匹配批次维度
            batch_graph = dgl.batch([single_graph] * batch_size_seq)
            # 调整特征形状
            feat_flat = feat.reshape(-1, feat.shape[-1])  # (batch_size_seq*num_nodes, in_feats)

            # 检查节点数是否匹配
            assert batch_graph.num_nodes() == feat_flat.shape[0], \
                f"Node count mismatch: {batch_graph.num_nodes()} vs {feat_flat.shape[0]}"

            # GAT计算 (batch_size_seq*num_nodes, num_heads, head_dim)
            x = self.gat_layers[gtype](batch_graph, feat_flat)
            # 恢复形状并合并多头
            x = x.view(batch_size_seq, num_nodes, self.num_heads, -1)  # (batch_size_seq, nodes, heads, dim)
            x = x.permute(0, 1, 3, 2).reshape(batch_size_seq, num_nodes, -1)  # (batch_size_seq, nodes, h_dim)
            # 投影变换
            views.append(self.view_proj[gtype](x))  # (batch_size_seq, nodes, h_dim)
        return views


# 多视图融合模块
class FusionModule(nn.Module):
    def __init__(self, num_graphs, emb_dim, alpha=0.8):
        super().__init__()
        self.emb_dim = emb_dim
        self.alpha = alpha

        # 自注意力组件
        self.self_q = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_graphs)])
        self.self_k = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_graphs)])

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(emb_dim, 4, batch_first=True)

        # 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        )
        # 初始化门控偏置，使初始状态接近alpha值
        with torch.no_grad():
            self.fusion_gate[0].bias.data.fill_(np.log(alpha / (1 - alpha)))
        self.fusion_out = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, views):
        # 自注意力融合
        self_attn_outs = []
        for i, view in enumerate(views):
            Q = self.self_q[i](view)
            K = self.self_k[i](view)
            # torch.matmul(Q, K.transpose(1, 2))：计算Query和Key的相似度矩阵。
            # 除以 np.sqrt(self.emb_dim)：缩放注意力分数，防止梯度爆炸。
            # F.softmax(..., dim=-1)：对最后一维归一化，得到注意力权重。
            attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim=-1)  # (4, num_nodes, num_nodes)
            self_attn_outs.append(torch.bmm(attn, view))

        # 交叉注意力
        views_stack = torch.stack(views, dim=1)  # [B, num_views, N, D]
        B, V, N, D = views_stack.shape
        cross_out, _ = self.cross_attn(
            views_stack.mean(dim=1),  # 全局查询
            views_stack.reshape(B, V * N, D),
            views_stack.reshape(B, V * N, D)
        )

        # 门控融合
        self_attn_mean = torch.stack(self_attn_outs).mean(dim=0)
        gate = self.fusion_gate(torch.cat([self_attn_mean, cross_out], dim=-1))
        # 应用alpha系数调整门控范围
        gate = self.alpha * gate + (1 - self.alpha) * 0.5  # 将输出控制在[alpha/2, (1+alpha)/2]之间
        fused = gate * self_attn_mean + (1 - gate) * cross_out
        fused = F.dropout(fused, p=0.3, training=self.training)  # 防止过拟合
        return fused, views

def cosine_similarity_np(a, b):
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

# 计算各个城市间相似度
def compute_city_similarity_simple(src_graph, tgt_graph):
    sim_adj = cosine_similarity_np(src_graph['matrix_adj'], tgt_graph['matrix_adj'])
    sim_poi = cosine_similarity_np(src_graph['poi_sim'], tgt_graph['poi_sim'])
    sim_source = cosine_similarity_np(src_graph['source_sim'], tgt_graph['source_sim'])
    sim_dest = cosine_similarity_np(src_graph['dest_sim'], tgt_graph['dest_sim'])
    return (sim_adj + sim_poi + sim_source + sim_dest) / 4

# 计算节点级别相似度
def compute_node_level_similarity_full(src_graph, tgt_graph):
    sims = []
    poi_src = src_graph['poi']
    poi_tgt = tgt_graph['poi']
    # 对每个小区的 POI 向量进行 L2 归一化
    poi_src_normalized = poi_src / (np.linalg.norm(poi_src, axis=1, keepdims=True) + 1e-6)  # (36, 14)
    poi_tgt_normalized = poi_tgt / (np.linalg.norm(poi_tgt, axis=1, keepdims=True) + 1e-6)  # (36, 14)
    # 计算余弦相似度：矩阵乘法 (36, 14) @ (14, 36) = (36, 36)
    poi_sim = poi_src_normalized @ poi_tgt_normalized.T
    sims.append(poi_sim)
    pickup_sim = compute_cross_city_similarity(src_graph['pickup'], tgt_graph['pickup'])
    sims.append(pickup_sim)
    dropoff_sim = compute_cross_city_similarity(src_graph['dropoff'], tgt_graph['dropoff'])
    sims.append(dropoff_sim)
    adj_sim = cosine_similarity_np(src_graph['matrix_adj'], tgt_graph['matrix_adj'])
    sims.append(adj_sim)
    return sum(sims) / len(sims)

# 计算不同城市起始和终止的流量分布相似性
def compute_cross_city_similarity(flow_data_src, flow_data_tgt):
    # Normalize: 对每个时间步归一化整张图的总流量
    norm_src = flow_data_src / (flow_data_src.sum(axis=(1, 2), keepdims=True) + 1e-6)
    norm_tgt = flow_data_tgt / (flow_data_tgt.sum(axis=(1, 2), keepdims=True) + 1e-6)
    # Flatten为 (T, 36)，每列是一个小区的时间序列
    norm_src_flat = norm_src.reshape(-1, 36)  # shape (T, 36)
    norm_tgt_flat = norm_tgt.reshape(-1, 36)  # shape (T, 36)
    sim_matrix = np.zeros((36, 36))
    for i in range(36):
        for j in range(36):
            # 时间序列上的欧氏距离
            diff = norm_src_flat[:, i] - norm_tgt_flat[:, j]
            dist = np.linalg.norm(diff)
            sim_matrix[i, j] = np.exp(-dist)
    return sim_matrix

# 平均后再对齐的MMD
class NodeSimAwareMMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(NodeSimAwareMMD, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, x, y, sim_matrix):
        x = x.mean(dim=0)  # (nodes, h_dim)
        y = y.mean(dim=0)  # (nodes, h_dim)

        # 确保sim_matrix是正确的形状
        if sim_matrix.dim() == 0:  # 如果是标量
            sim_matrix = sim_matrix.expand(x.size(0), y.size(0))
        elif sim_matrix.size(0) != x.size(0) or sim_matrix.size(1) != y.size(0):
            # 如果不匹配，使用默认相似度
            sim_matrix = torch.ones(x.size(0), y.size(0), device=x.device)

        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        bandwidth = torch.mean(xy)
        # 高斯核RBF，k_xy[i,j]表示目标城市节点 i 与源城市节点 j 的嵌入距离的核函数值（越相似越大）
        k_xy = torch.exp(-xy / (2 * bandwidth))  # (36, 36)
        k_xx = torch.exp(-xx / (2 * bandwidth)).mean()
        k_yy = torch.exp(-yy / (2 * bandwidth)).mean()
        # 将每对节点之间的特征相似性核值（k_xy）按结构相似性进行加权
        weighted_xy = torch.mean(k_xy * sim_matrix)
        loss = k_xx + k_yy - 2 * weighted_xy
        # 限制损失范围
        loss = torch.clamp(loss, min=0.0, max=10.0)
        return loss



# 无结构感知MMD
class VanillaMMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(VanillaMMD, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, x, y):
        x = x.mean(dim=0)  # (nodes, h_dim)
        y = y.mean(dim=0)  # (nodes, h_dim)
        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        bandwidth = torch.mean(xy)
        k_xx = torch.exp(-xx / (2 * bandwidth)).mean()
        k_yy = torch.exp(-yy / (2 * bandwidth)).mean()
        k_xy = torch.exp(-xy / (2 * bandwidth)).mean()
        loss = k_xx + k_yy - 2 * k_xy

        return loss

# 节点级域分类
class DomainAdapter(nn.Module):
    def __init__(self, h_dim, num_sources):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.domain_head = nn.Linear(h_dim, num_sources)
        self.grl = GradientReversalLayer()

    def forward(self, feat, alpha=0.1):
        """
        输入: feat - (batch, nodes, h_dim)
        输出: (batch, nodes, num_sources)
        """
        reversed_feat = self.grl.apply(feat, alpha)
        # 保持节点维度独立处理
        batch, nodes, _ = reversed_feat.shape
        shared = self.shared_layer(reversed_feat.reshape(-1, self.shared_layer[0].in_features))
        shared = shared.view(batch, nodes, -1)
        return self.domain_head(shared)

# 时空图卷积网络
class MSDAN(nn.Module):
    def __init__(self, in_feats, h_dim, city_graph_data_list, device, use_mvgat=True, use_gcn=True, use_temporal=True, use_dan=True, num_sources=3, sim_weights=None, node_sim_matrices=None, mmd_mode="nodesim", dropout_rate=0.3):
        super().__init__()
        self.use_mvgat = use_mvgat
        self.use_gcn = use_gcn
        self.use_temporal = use_temporal
        self.use_dan = use_dan
        self.device = device
        self.num_sources = num_sources
        self.sim_weights = sim_weights if sim_weights is not None else [1.0] * num_sources
        self.h_dim = h_dim
        # 增加dropout
        self.dropout = nn.Dropout(dropout_rate)

        # 图结构缓存
        self.graphs_cache = []
        for data in city_graph_data_list:
            self.graphs_cache.append({
                'adj': self._build_graph(data['matrix_adj']),
                'poi': self._build_graph(data['poi_sim']),
                'source': self._build_graph(data['source_sim']),
                'dest': self._build_graph(data['dest_sim'])
            })

        self.node_sim_matrices = node_sim_matrices

        # # 多视图GAT
        # self.mvgat = MVGAT(in_feats, h_dim)
        # # 多视图融合
        # self.fusion_module = FusionModule(num_graphs=4, emb_dim=h_dim)

        # 消融实验
        if self.use_mvgat:
            self.mvgat = MVGAT(in_feats, h_dim)
            self.fusion_module = FusionModule(num_graphs=4, emb_dim=h_dim)
        else:
            self.input_proj = nn.Linear(in_feats, h_dim)  # 简化处理

        # 时空特征提取
        # self.spatial_gcn = GraphConv(h_dim, h_dim)
        # self.temporal_conv = nn.Sequential(
        #     nn.Conv2d(in_feats, h_dim, kernel_size=(1, 3), padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(h_dim, h_dim, kernel_size=(1, 3), padding=(0, 1))
        # )

        if self.use_gcn:
            self.spatial_gcn = GraphConv(h_dim, h_dim)
        if self.use_temporal:
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(in_feats, h_dim, kernel_size=(1, 3), padding=(0, 1)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=(1, 3), padding=(0, 1))
            )


        # 域适应模块
        self.domain_adapter = DomainAdapter(h_dim, num_sources) if self.use_dan else None
        # 去除域适应模块
        # self.domain_adapter = None
        print(mmd_mode)
        if mmd_mode == "nodesim":
            self.mmd_loss_type = "nodesim"
            self.mmd_loss = NodeSimAwareMMD()
        elif mmd_mode == "vanilla":
            self.mmd_loss_type = "vanilla"
            self.mmd_loss = VanillaMMD()
        else:
            self.mmd_loss_type = "none"
            self.mmd_loss = None
        print(self.mmd_loss_type)
        self.source_feats = {i: [] for i in range(num_sources)}

        # 预测头
        self.predictor = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(h_dim, 6, kernel_size=3, padding=1)
        )

    def _build_graph(self, adj_matrix):
        # 确保邻接矩阵是36x36的
        if adj_matrix.shape != (36, 36):
            # 如果维度不匹配，创建36x36的零矩阵并填充
            new_adj = np.zeros((36, 36))
            min_dim = min(adj_matrix.shape[0], 36)
            new_adj[:min_dim, :min_dim] = adj_matrix[:min_dim, :min_dim]
            adj_matrix = new_adj
        
        # 找到非零元素的位置
        rows, cols = np.where(adj_matrix > 0)
        
        # 如果没有边，创建自环边
        if len(rows) == 0:
            rows = np.arange(36)
            cols = np.arange(36)
        else:
            # 转换为1D索引
            src = rows
            dst = cols
            
            # 确保所有节点都存在（添加自环）
            all_nodes = np.arange(36)
            missing_nodes = np.setdiff1d(all_nodes, np.union1d(rows, cols))
            if len(missing_nodes) > 0:
                rows = np.concatenate([rows, missing_nodes])
                cols = np.concatenate([cols, missing_nodes])
        
        # 创建图
        g = dgl.graph((rows, cols), num_nodes=36).to(self.device)
        # 添加自环边
        g = dgl.add_self_loop(g)
        return g

    def forward(self, x, domain_idx=None, is_target=False):
        batch_size, seq_len, num_nodes, feats = x.shape
        graphs = self.graphs_cache[domain_idx if domain_idx is not None else -1]

        # 1. 多视图特征提取
        # # 合并batch和seq_len维度，保持nodes独立
        # x_reshaped = x.reshape(-1, num_nodes, feats)  # (batch*seq_len, nodes, feats)
        # views = self.mvgat(graphs, x_reshaped)  # 每个view形状 (batch*seq_len, 36, h_dim)
        # fused, _ = self.fusion_module(views)

        if self.use_mvgat:
            x_reshaped = x.reshape(-1, num_nodes, feats)
            views = self.mvgat(graphs, x_reshaped)
            fused, _ = self.fusion_module(views)
        else:
            fused = self.input_proj(x.reshape(-1, num_nodes, feats))  # 简单投影

        # 恢复时空维度
        fused = fused.view(batch_size, seq_len, num_nodes, -1)  # (64,12,36,64)

        # 2. 时空特征交互
        # 先处理时间维度，再处理空间维度
        # spatial_feats = []
        # for t in range(seq_len):
        #     # 对每个时间步单独进行图卷积
        #     time_slice = fused[:, t, :, :]  # (batch_size, num_nodes, h_dim)
        #     time_slice_flat = time_slice.reshape(-1, self.h_dim)  # (batch_size*num_nodes, h_dim)
        #     # 复制图结构以匹配batch维度
        #     batch_graph = dgl.batch([graphs['adj']] * batch_size)
        #     # 图卷积处理
        #     spatial_feat = self.spatial_gcn(batch_graph, time_slice_flat)
        #     spatial_feat = spatial_feat.view(batch_size, num_nodes, -1)
        #     spatial_feats.append(spatial_feat)
        # # 重新组合时空特征
        # spatial_feat = torch.stack(spatial_feats, dim=1)  # (batch_size, seq_len, num_nodes, h_dim)


        if self.use_gcn:
            spatial_feats = []
            for t in range(seq_len):
                # 对每个时间步单独进行图卷积
                time_slice = fused[:, t, :, :]  # (batch_size, num_nodes, h_dim)
                time_slice_flat = time_slice.reshape(-1, self.h_dim)  # (batch_size*num_nodes, h_dim)
                # 复制图结构以匹配batch维度
                batch_graph = dgl.batch([graphs['adj']] * batch_size)
                # 图卷积处理
                spatial_feat = self.spatial_gcn(batch_graph, time_slice_flat)
                # 防止过拟合
                spatial_feat = self.dropout(spatial_feat)
                # 残差连接：只能加当前时间步的特征
                residual = time_slice_flat  # 或 fused[:, t, :, :].reshape(-1, self.h_dim)
                spatial_feat = spatial_feat + residual
                spatial_feat = F.layer_norm(spatial_feat, [self.h_dim])
                spatial_feat = spatial_feat.view(batch_size, num_nodes, -1)
                spatial_feats.append(spatial_feat)
            # 重新组合时空特征 
            spatial_feat = torch.stack(spatial_feats, dim=1)  # (batch_size, seq_len, num_nodes, h_dim)
        else:
            spatial_feat = fused  # 直接跳过 GCN



        # temporal_feat = self.temporal_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # (batch_size, seq_len, num_nodes, h_dim)

        if self.use_temporal:
            temporal_feat = self.temporal_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        else:
            temporal_feat = torch.zeros_like(spatial_feat)


        st_feat = F.relu(spatial_feat + temporal_feat)  # (batch_size, seq_len, num_nodes, h_dim)
        # 防止过拟合
        st_feat = F.dropout(st_feat, p=0.3, training=self.training)

        # 3. 主任务预测
        pred = self.predictor(st_feat.permute(0, 3, 2, 1)).mean(dim=-1, keepdim=True)

        # 4. 域适应处理
        if domain_idx is not None:
            if self.domain_adapter is not None:
                # 源域：域分类(batch_size, num_nodes, num_sources)，取时间维度均值
                domain_logits = self.domain_adapter(st_feat.mean(dim=1))
                # 创建目标标签(batch_size, num_nodes)每个节点标记为当前domain_idx
                target = torch.full((batch_size, num_nodes), domain_idx, dtype=torch.long, device=self.device)
                # 计算节点级域分类损失
                domain_loss = F.cross_entropy(
                    domain_logits.reshape(-1, self.num_sources),  # (batch_size*num_nodes, num_sources)
                    target.reshape(-1)  # (batch_size*num_nodes)
                )
            else:
                # 不进行域分类
                domain_loss = torch.tensor(0.0, device=self.device)
            self.source_feats[domain_idx].append(st_feat.mean(dim=1).detach())
            return pred, domain_loss

        elif is_target and self.mmd_loss is not None:
            # 目标域：MMD对齐
            mmd_loss = 0
            valid_domains = 0
            target_feat = st_feat.mean(dim=1)
            for i in range(self.num_sources):
                if len(self.source_feats[i]) > 0:
                    source_feat = torch.cat(self.source_feats[i][-10:], dim=0)
                    # 跨域相似度权重
                    if self.mmd_loss_type == "nodesim":
                        sim_weight = self.sim_weights[i]
                        mmd_loss += sim_weight * self.mmd_loss(target_feat, source_feat, self._get_sim_matrix(i, target_feat.device))
                        # print("mmd_loss:%.4f" % mmd_loss)
                    else:
                        mmd_loss += self.mmd_loss(target_feat, source_feat)
                    valid_domains += 1
            mmd_loss = mmd_loss / valid_domains if valid_domains > 0 else 0
            # 添加梯度裁剪，限制MMD损失的范围
            mmd_loss = torch.clamp(mmd_loss, max=1.0)  # 限制最大值
            return pred, mmd_loss
        return pred

    def _get_sim_matrix(self, domain_idx, device):
        return torch.tensor(
            self.node_sim_matrices[domain_idx],
            dtype=torch.float32,
            device=device
        )

# ========== 训练与评估模块 ==========
# 动态权重计算
class DomainWeightCalculator:
    def __init__(self, num_sources, init_weight=0.5, ema_alpha=0.9):
        self.weights = torch.ones(num_sources) / num_sources
        self.loss_ema = torch.ones(num_sources) * init_weight
        self.ema_alpha = ema_alpha
        self.num_sources = num_sources

    # 更新域权重
    def update(self, domain_idx, loss_value):
        with torch.no_grad():
            # 指数移动平均更新损失
            self.loss_ema[domain_idx] = (
                    self.ema_alpha * self.loss_ema[domain_idx] +
                    (1 - self.ema_alpha) * loss_value
            )
            # 基于损失倒数计算权重
            inv_loss = 1 / (self.loss_ema + 1e-6)
            inv_loss = torch.clamp(inv_loss, min=1e-3)  # 防止出现极端权重
            self.weights = F.softmax(inv_loss, dim=0)

    # 获取当前权重
    def get_weights(self, device):
        eps = 0.05  # 最小权重
        max_w = 0.9  # 最大权重
        weights = F.softmax(1 / (self.loss_ema + 1e-6), dim=0)
        weights = torch.clamp(weights, min=eps)
        weights = weights / weights.sum()  # 重新归一化
        weights = torch.clamp(weights, max=max_w)
        weights = weights / weights.sum()  # 再归一化一次
        return weights.to(device)

def pretrain_on_sources(model, data_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = {'pretrain_loss': []}
    domain_weight_calculator = DomainWeightCalculator(model.num_sources)

    print("Starting source domain pretraining...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # 源域预训练
        for domain_idx, (city, loader) in enumerate(data_loader.source_train_loaders.items()):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred, domain_loss = model(x, domain_idx)
                # pred = model(x, domain_idx)
                task_loss = F.l1_loss(pred, y)
                # 获取源域权重
                weight = domain_weight_calculator.get_weights(device)[domain_idx]
                loss = (task_loss + min(0.01, (0.1 * (epoch / 100))) * domain_loss) * weight
                # loss = task_loss * weight
                print(f"Epoch {epoch}, Domain {domain_idx}, Task Loss={task_loss.item():.4f}, Domain Loss={domain_loss.item():.4f}, Weight={weight.item():.6f}, Final Loss={loss.item():.6f}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                domain_weight_calculator.update(domain_idx, loss.item())
                batch_count += 1
        avg_loss = total_loss / batch_count
        history['pretrain_loss'].append(avg_loss)

        # 在源域验证集上评估
        if (epoch + 1) % 10 == 0:
            source_val_results = {}
            for city, loader in data_loader.source_val_loaders.items():
                # 获取源城市的归一化参数
                max_val, min_val = data_loader.get_norm_params(city)
                mae, rmse, _ = evaluate(model, loader, device, target=False, max_val=max_val, min_val=min_val)
                source_val_results[city] = (mae, rmse)
            print(f"Pretrain Epoch {(epoch + 1)}: Loss={avg_loss:.4f} | "
                  f"Time {time.time() - start_time}")
            for city, (mae, rmse) in source_val_results.items():
                print(f" {city} Val MAE={mae:.4f  }, "
                      f"RMSE={rmse:.4f}")
    return history

def finetune_on_target(model, data_loader, epochs=100, device="cuda", mmd_weight=0.03, city="Unknown"):
    print(f"Starting finetuning on target {city}")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # 获取目标城市的归一化参数
    max_val, min_val = data_loader.get_norm_params(target_city)

    history = {
        "train_rmse": [], "val_rmse": [],
        "train_mae": [], "val_mae": [],
    }

    best_val_rmse = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, running_task_loss, running_mmd_loss = 0.0, 0.0, 0.0
        n_batches = 0

        for x, y in data_loader.target_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # pred, mmd_loss = model(x, None, is_target=True)
            out = model(x, None, is_target=True)
            if isinstance(out, tuple):
                pred, mmd_loss = out
            else:
                pred = out
                mmd_loss = torch.tensor(0.0, device=device)

            task_loss = F.mse_loss(pred, y)
            loss = task_loss + mmd_weight * mmd_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_task_loss += task_loss.item()
            running_mmd_loss += mmd_loss.item()
            n_batches += 1

        # ---- 评估 ----
        train_rmse, train_mae, _ = evaluate(model, data_loader.target_train_loader, device, target=True, max_val=max_val, min_val=min_val)
        val_rmse, val_mae, _ = evaluate(model, data_loader.target_val_loader, device, target=True, max_val=max_val, min_val=min_val)

        # 保存历史
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        # ---- 日志打印 ----
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Loss: {running_loss/n_batches:.4f} (Task {running_task_loss/n_batches:.4f} + MMD {running_mmd_loss/n_batches:.4f}, w={mmd_weight}) | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, history

def min_max_denormalize(norm_data, max_val, min_val):
    return norm_data * (max_val - min_val) + min_val

def evaluate(model, loader, device="cuda", target=False, max_val=None, min_val=None):
    """
        avg_mae: 平均 MAE
        avg_rmse: 平均 RMSE
        preds_list: 所有预测值（tensor列表）
    """
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            # 如果 batch 是 Data 对象（PyG风格）
            if hasattr(batch, "x") and hasattr(batch, "y"):
                x, y = batch.x.to(device), batch.y.to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)

            # forward
            if target:
                out = model(x, is_target=True)
            else:
                out = model(x)
            # 兼容返回 (pred, aux) 或 pred
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out

            preds_list.append(pred.detach().cpu())
            total_mae += F.l1_loss(pred, y, reduction='mean').item()
            total_mse += F.mse_loss(pred, y, reduction='mean').item()

            # 反归一化
            if max_val is not None and min_val is not None:
                y_denorm = min_max_denormalize(y, max_val, min_val)
                pred_denorm = min_max_denormalize(pred, max_val, min_val)
            else:
                y_denorm = y
                pred_denorm = pred

            preds_list.append(pred_denorm.detach().cpu())
            total_mae += F.l1_loss(pred_denorm, y_denorm, reduction='mean').item()
            total_mse += F.mse_loss(pred_denorm, y_denorm, reduction='mean').item()

    avg_mae = total_mae / len(loader)
    avg_rmse = torch.sqrt(torch.tensor(total_mse / len(loader))).item()

    return avg_mae, avg_rmse, preds_list


if __name__ == "__main__":
    all_cities = ['Washington', 'Chicago', 'NYCBike', 'LABike']
    mmd_modes = ["nodesim", "vanilla", "none"]
    colors = {"nodesim": "red", "vanilla": "green", "none": "blue"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 四张子图

    for idx, target_city in enumerate(all_cities):
        source_cities = [c for c in all_cities if c != target_city]
        print(f"\nRunning with target={target_city}, sources={source_cities}")

        data_loader = MultiSourceDataLoader(source_cities, target_city)
        city_graph_data = [data_loader.get_city_graph_data(c) for c in source_cities + [target_city]]
        target_graph = city_graph_data[-1]

        # 计算相似度
        sim_weights = []
        for i in range(len(source_cities)):
            src_graph = city_graph_data[i]
            sim = compute_city_similarity_simple(src_graph, target_graph)
            sim_weights.append(sim)
        sim_weights = [s / sum(sim_weights) for s in sim_weights]

        node_sim_matrices = []
        for i in range(len(source_cities)):
            src_graph = city_graph_data[i]
            sim_matrix = compute_node_level_similarity_full(src_graph, target_graph)
            node_sim_matrices.append(sim_matrix)

        # 创建模型
        for mmd_mode in mmd_modes:
            model = MSDAN(
                in_feats=4,
                h_dim=64,
                city_graph_data_list=city_graph_data,
                device=device,
                use_mvgat=True,
                use_gcn=True,
                use_temporal=True,
                use_dan=True,
                num_sources=len(source_cities),
                sim_weights=sim_weights,
                node_sim_matrices=node_sim_matrices,
                mmd_mode=mmd_mode,
                dropout_rate=0.2
            ).to(device)

            # 训练
            pretrain_on_sources(model, data_loader, epochs=150)
            # 微调
            finetuned_model, finetune_history = finetune_on_target(
                model,
                data_loader,
                epochs=150,
                device=device,
                mmd_weight=0.1,
                city=target_city
            )

            # 绘制 Overfitting Analysis 到子图
            row, col = divmod(idx, 2)
            overfitting_gap = [
                finetune_history['val_rmse'][i] - finetune_history['train_rmse'][i]
                for i in range(len(finetune_history['val_rmse']))
            ]

            # overfitting_gap = [finetune_history['val_mae'][i] - finetune_history['train_task_loss'][i]
            #                    for i in range(len(finetune_history['val_mae']))]

            # overfitting_gap = [finetune_history['train_loss'][i] - finetune_history['val_rmse'][i]
            #                    for i in range(len(finetune_history['train_loss']))]
            axes[row, col].plot(overfitting_gap, label=mmd_mode, color=colors[mmd_mode])
            axes[row, col].set_title(f"Overfitting Analysis - Target: {target_city}")
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Val RMSE - Train RMSE')
            axes[row, col].grid(True)
            axes[row, col].legend()

    plt.tight_layout()
    # plt.savefig("overfitting_analysis_all_targets.png", dpi=300, bbox_inches='tight')
    plt.show()

