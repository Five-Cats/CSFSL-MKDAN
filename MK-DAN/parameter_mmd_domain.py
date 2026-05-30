import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import dgl
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999


# ========== 重要修改说明 ==========
# 针对过拟合和少样本问题，已进行以下优化：
# 1. 移除一致性损失 (consistency loss) - 减少不必要的约束
# 2. 平衡学习率调度 - T_0=8, eta_min=1e-6，避免过快衰减
# 3. 优化早停策略 - 耐心值12，改善阈值5e-5，平衡敏感性和稳定性
# 4. 增强过拟合监控 - 多级阈值检测，轻微和严重过拟合分别处理
# 5. 添加学习率监控 - 每5个epoch显示当前学习率
# 6. 扩展性能记录 - 同时保存最佳MAE和最佳RMSE
# ==========================================

# ========== 归一化函数 ==========
# 对数据进行归一化处理
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    norm_data = (data - min_val) / (max_val - min_val + 1e-6)
    return norm_data, max_val, min_val


# ========== 数据加载模块 ==========
class TrafficDataset(Dataset):
    def __init__(self, data, lookback, pred_len, start_date="2023-04-01 00:00:00", timestep_min=20, idx=None,
                 augment=True):
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
        return torch.FloatTensor(enhanced_x), torch.FloatTensor(y)

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
    def __init__(self, source_cities, target_city, lookback=12, pred_len=6, train_ratio=0.7, val_ratio=0.1,
                 test_ratio=0.2, random_state=42):
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
            self.source_train_loaders[city] = DataLoader(train_dataset, batch_size=64, shuffle=False)
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

        self.target_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        self.target_val_loader = DataLoader(val_dataset, batch_size=32)
        self.target_test_loader = DataLoader(test_dataset, batch_size=32)

    # 获取指定城市的完整图数据
    def get_city_graph_data(self, city_name):
        return self.city_graphs[city_name]

    # 获取指定城市的归一化参数
    def get_norm_params(self, city_name):
        return self.max_vals[city_name], self.min_vals[city_name]


# ========== 模型架构模块 ==========
# --- GRL ---
class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0
    def forward(self, x, alpha=None):
        return _GradientReversalFunction.apply(x, self.alpha if alpha is None else alpha)


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
                nn.Dropout(p=0.3)  # dropout防止过拟合
            ) for gtype in ['adj', 'poi', 'source', 'dest']
        })

    def forward(self, graphs, feat):
        """
        graphs: 包含四个静态图结构的字典（adj, poi, source, dest）
        feat: 输入特征 (batch_size * seq_len, num_nodes, in_feats)
        """
        batch_size_seq, num_nodes, _ = feat.shape
        views = []

        for gtype in ['adj', 'poi', 'source', 'dest']:
            single_graph = graphs[gtype]

            # 检查节点数是否匹配
            if single_graph.num_nodes() != num_nodes:
                print(f"Warning: Graph {gtype} has {single_graph.num_nodes()} nodes, expected {num_nodes}")
                src = dst = np.arange(num_nodes)
                single_graph = dgl.graph((src, dst), num_nodes=num_nodes).to(single_graph.device)
                single_graph = dgl.add_self_loop(single_graph)

            # ---- 批处理：将图复制 batch_size_seq 次 ----
            batch_graph = dgl.batch([single_graph] * batch_size_seq)

            # ---- flatten 特征以匹配图节点 ----
            feat_flat = feat.reshape(-1, feat.shape[-1])  # (batch_size_seq * num_nodes, in_feats)

            # ---- GAT 计算 ----
            x = self.gat_layers[gtype](batch_graph, feat_flat)  # (B*N, num_heads, head_dim)

            # ---- 合并多头 & 还原形状 ----
            x = x.view(batch_size_seq, num_nodes, self.num_heads, -1)  # (B, N, heads, dim)
            x = x.permute(0, 1, 3, 2).reshape(batch_size_seq, num_nodes, -1)  # (B, N, h_dim)

            # ---- 线性投影 & dropout ----
            views.append(self.view_proj[gtype](x))  # (B, N, h_dim)

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
            attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim),
                             dim=-1)  # (4, num_nodes, num_nodes)
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

    def gaussian_kernel(self, dist2, base_bw):
        """
        多核 RBF:
        dist2: pairwise squared distance matrix  (36, 36)
        base_bw: 基础 bandwidth（标量）
        返回：所有 kernel 的 RBF 平均值 (36, 36)
        """
        kernels = []
        for i in range(self.kernel_num):
            bw = base_bw * (self.kernel_mul ** i)
            kernels.append(torch.exp(-dist2 / (2 * bw)))
        return sum(kernels) / len(kernels)  # 多核平均

    def forward(self, x, y, sim_matrix):
        """
            x, y: 交通多视图融合后的节点嵌入  shape (B, N, D)
            sim_matrix: 节点级结构相似性 (N, N)
        """
        x = x.mean(dim=0)  # (nodes, h_dim)
        y = y.mean(dim=0)  # (nodes, h_dim)
        N = x.size(0)

        # 确保sim_matrix是正确的形状
        if sim_matrix.dim() == 0:  # 如果是标量
            sim_matrix = sim_matrix.expand(N, N)
        elif sim_matrix.size(0) != N or sim_matrix.size(1) != N:
            # 如果不匹配，使用默认相似度
            sim_matrix = torch.ones(N, N, device=x.device)
        # 另外确保 sim_matrix 在同设备 dtype
        sim_matrix = sim_matrix.to(x.device).to(x.dtype)

        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        # 多核RBF基础带宽
        bandwidth = torch.mean(xy).clamp(min=1e-6)

        # 高斯核RBF，k_xy[i,j]表示目标城市节点 i 与源城市节点 j 的嵌入距离的核函数值（越相似越大）
        k_xx = self.gaussian_kernel(xx, bandwidth).mean()
        k_yy = self.gaussian_kernel(yy, bandwidth).mean()
        k_xy = self.gaussian_kernel(xy, bandwidth)
        # 将每对节点之间的特征相似性核值（k_xy）按结构相似性进行加权
        weighted_xy = torch.mean(k_xy * sim_matrix)
        loss = k_xx + k_yy - 2 * weighted_xy
        # 限制损失范围
        loss = torch.clamp(loss, min=0.0, max=10.0)
        return loss


# 无结构感知MMD
# class VanillaMMD(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=5):
#         super(VanillaMMD, self).__init__()
#         self.kernel_mul = kernel_mul
#         self.kernel_num = kernel_num
#
#     def forward(self, x, y):
#         x = x.mean(dim=0)  # (nodes, h_dim)
#         y = y.mean(dim=0)  # (nodes, h_dim)
#         xx = torch.cdist(x, x).pow(2)
#         yy = torch.cdist(y, y).pow(2)
#         xy = torch.cdist(x, y).pow(2)
#
#         bandwidth = torch.mean(xy)
#         bandwidth = torch.clamp(bandwidth, min=1e-6)
#         k_xx = torch.exp(-xx / (2 * bandwidth)).mean()
#         k_yy = torch.exp(-yy / (2 * bandwidth)).mean()
#         k_xy = torch.exp(-xy / (2 * bandwidth)).mean()
#         loss = k_xx + k_yy - 2 * k_xy
#
#         return loss


# 节点级域分类
class DomainAdapter(nn.Module):
    def __init__(self, h_dim, num_domains):
        """
        num_domains: number of domain classes (e.g. num_sources + 1 for target)
        """
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.domain_head = nn.Linear(h_dim, num_domains)
        self.num_domains = num_domains

    def forward(self, feat):
        """
        输入: feat - (batch, nodes, h_dim)
        输出: (batch, nodes, num_domains)
        """
        batch, nodes, _ = feat.shape
        # shared_layer expects input size = h_dim
        shared = self.shared_layer(feat.reshape(-1, self.shared_layer[0].in_features))
        shared = shared.view(batch, nodes, -1)  # (batch, nodes, h_dim)
        return self.domain_head(shared)  # (batch, nodes, num_domains)


# 时空图卷积网络
class MSDAN(nn.Module):
    def __init__(self, in_feats, h_dim, city_graph_data_list, device, use_fusion=True, use_gcn=True, use_temporal=True,
                 use_dan=True, num_sources=3, sim_weights=None, node_sim_matrices=None, mmd_mode="nodesim",
                 dropout_rate=0.3):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_gcn = use_gcn
        self.use_temporal = use_temporal
        self.use_dan = use_dan
        self.device = device
        self.num_sources = num_sources
        self.sim_weights = sim_weights if sim_weights is not None else [1.0] * num_sources
        self.h_dim = h_dim
        self.grl = GradientReversalLayer()
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

        # 消融实验
        if self.use_fusion:
            self.mvgat = MVGAT(in_feats, h_dim)
            self.fusion_module = FusionModule(num_graphs=4, emb_dim=h_dim)
        else:
            self.mvgat = MVGAT(in_feats, h_dim)
            # self.input_proj = nn.Linear(in_feats, h_dim)  # 简化处理


        if self.use_gcn:
            self.spatial_gcn = GraphConv(h_dim, h_dim)
        if self.use_temporal:
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(in_feats, h_dim, kernel_size=(1, 3), padding=(0, 1)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=(1, 3), padding=(0, 1))
            )

        self.lambda_grl = 0.0  # 初始梯度反转强度
        self.max_lambda_grl = 0.2  # 最大值，可调整
        self.grl_schedule = True  # 是否启用动态增长

        # 域适应模块
        if self.use_dan:
            # num_domains = number of source domains + 1 (target)
            num_domains = self.num_sources + 1
            self.domain_adapter = DomainAdapter(h_dim, num_domains)
            self.domain_adapter.to(self.device)
        else:
            self.domain_adapter = None

            # MMD 配置保持原样（nodesim / vanilla / none）
        if mmd_mode == "nodesim":
            self.mmd_loss_type = "nodesim"
            self.mmd_loss = NodeSimAwareMMD()
        # elif mmd_mode == "vanilla":
        #     self.mmd_loss_type = "vanilla"
        #     self.mmd_loss = VanillaMMD()
        else:
            self.mmd_loss_type = "none"
            self.mmd_loss = None

        self.source_feats = {i: [] for i in range(num_sources)}
        print(self.mmd_loss_type)
        self.source_feats = {i: [] for i in range(num_sources)}

        # 预测头
        self.predictor = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(h_dim, 6, kernel_size=3, padding=1)
        )
        self.layer_norm = nn.LayerNorm(h_dim)

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

        if self.use_fusion:
            x_reshaped = x.reshape(-1, num_nodes, feats)
            views = self.mvgat(graphs, x_reshaped)
            fused, _ = self.fusion_module(views)
        else:
            # fused = self.input_proj(x.reshape(-1, num_nodes, feats))  # 简单投影
            x_reshaped = x.reshape(-1, num_nodes, feats)
            views = self.mvgat(graphs, x_reshaped)
            # 简单平均融合（消融：去掉 FusionModule 的自注意力/门控）
            fused = torch.stack(views, dim=0).mean(dim=0)  # (B*seq_len, N, h_dim)

        # 恢复时空维度
        fused = fused.view(batch_size, seq_len, num_nodes, -1)  # (64,12,36,64)


        if self.use_gcn:
            spatial_feats = []
            # 复制图结构以匹配batch维度
            # batch_graph = dgl.batch([graphs['adj']] * batch_size)
            for t in range(seq_len):
                # 对每个时间步单独进行图卷积
                time_slice = fused[:, t, :, :]  # (batch_size, num_nodes, h_dim)
                # time_slice_flat = time_slice.reshape(-1, self.h_dim)  # (batch_size*num_nodes, h_dim)
                outputs = []
                for i in range(batch_size):
                    out = self.spatial_gcn(graphs['adj'], time_slice[i])  # (N, h_dim)
                    out = self.dropout(out + time_slice[i])  # residual
                    out = self.layer_norm(out)
                    outputs.append(out)
                spatial_feats.append(torch.stack(outputs, dim=0))  # (B, N, h_dim)
                # 图卷积处理
                # spatial_feat = self.spatial_gcn(batch_graph, time_slice_flat)
                # 防止过拟合
                # spatial_feat = self.dropout(spatial_feat)
                # # 残差连接：只能加当前时间步的特征
                # residual = time_slice_flat  # 或 fused[:, t, :, :].reshape(-1, self.h_dim)
                # spatial_feat = spatial_feat + residual
                # spatial_feat = F.layer_norm(spatial_feat, [self.h_dim])
                # spatial_feat = spatial_feat.view(batch_size, num_nodes, -1)
                # spatial_feats.append(spatial_feat)
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

        # 4. 域自适应处理
        if (domain_idx is not None) or is_target:
            feat_mean = st_feat.mean(dim=1)  # (B, N, d)

            # 1. Domain classification with GRL
            if self.domain_adapter is not None:
                # -------- 动态调整 λ 值（根据训练进度） --------
                # 例如让 λ 从 0 → max_lambda_grl 逐渐增大
                if self.grl_schedule and hasattr(self, "global_step"):
                    p = min(self.global_step / self.total_steps, 1.0)
                    lambda_grl = self.max_lambda_grl * (2. / (1. + np.exp(-10 * p)) - 1)
                else:
                    lambda_grl = self.max_lambda_grl
                # -------- 应用梯度反转层 --------
                feat_rev = self.grl(feat_mean, lambda_grl)
                # -------- 域分类预测 --------
                domain_logits = self.domain_adapter(feat_rev)  # (B, N, 2)
                # -------- 域标签构造 --------
                # 构造多类别域标签：源域 -> domain_idx, 目标域 -> self.num_sources
                if domain_idx is not None:
                    # source batch: label = domain_idx for every node
                    domain_labels = torch.full((batch_size, num_nodes), domain_idx, dtype=torch.long, device=self.device)
                else:
                    # target batch: label = last class index
                    domain_labels = torch.full((batch_size, num_nodes), self.num_sources, dtype=torch.long, device=self.device)
                # -------- 域分类损失 --------
                domain_loss = F.cross_entropy(
                    domain_logits.reshape(-1, domain_logits.shape[-1]),
                    domain_labels.reshape(-1),
                    label_smoothing=0.1
                )
            else:
                domain_loss = torch.tensor(0.0, device=self.device)

            # 2. MMD 对齐（仅当配置了 self.mmd_loss 时才调用）
            mmd_loss = torch.tensor(0.0, device=self.device)
            if domain_idx is not None:
                # 源域：缓存 embedding
                self.source_feats[domain_idx].append(feat_mean.detach())
            else:
                # 目标域：MMD（可能被禁用）
                valid_domains = 0
                mmd_enabled = (self.mmd_loss is not None)
                for i in range(self.num_sources):
                    if len(self.source_feats[i]) > 0:
                        source_feat = torch.cat(self.source_feats[i][-10:], dim=0)
                        if mmd_enabled:
                            if self.mmd_loss_type == "nodesim":
                                sim_weight = self.sim_weights[i]
                                # 如果 node_sim_matrices 缺失，退回到全 1 矩阵
                                if self.node_sim_matrices is not None:
                                    sim_mat = self._get_sim_matrix(i, feat_mean.device)
                                else:
                                    sim_mat = torch.ones(feat_mean.size(1), source_feat.size(1), device=feat_mean.device)
                                mmd_loss += sim_weight * self.mmd_loss(feat_mean, source_feat, sim_mat)
                            else:
                                mmd_loss += self.mmd_loss(feat_mean, source_feat)
                        valid_domains += 1

                if mmd_enabled and valid_domains > 0:
                    mmd_loss = mmd_loss / valid_domains
                else:
                    mmd_loss = torch.tensor(0.0, device=self.device)

                mmd_loss = torch.clamp(mmd_loss, max=1.0)

            # 返回
            return pred, domain_loss, mmd_loss

        # 如果不进行域自适应（如测试），保留原行为
        return pred, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

    def _get_sim_matrix(self, domain_idx, device):
        return torch.tensor(
            self.node_sim_matrices[domain_idx],
            dtype=torch.float32,
            device=device
        )


# # ========== 训练与评估模块 ==========
# # 动态权重计算
# class DomainWeightCalculator:
#     def __init__(self, num_sources, init_weight=0.5, ema_alpha=0.9):
#         self.weights = torch.ones(num_sources) / num_sources
#         self.loss_ema = torch.ones(num_sources) * init_weight
#         self.ema_alpha = ema_alpha
#         self.num_sources = num_sources
#
#     # 更新域权重
#     def update(self, domain_idx, loss_value):
#         with torch.no_grad():
#             # 指数移动平均更新损失
#             self.loss_ema[domain_idx] = (
#                     self.ema_alpha * self.loss_ema[domain_idx] +
#                     (1 - self.ema_alpha) * loss_value
#             )
#             # 基于损失倒数计算权重
#             inv_loss = 1 / (self.loss_ema + 1e-6)
#             inv_loss = torch.clamp(inv_loss, min=1e-3)  # 防止出现极端权重
#             self.weights = F.softmax(inv_loss, dim=0)
#
#     # 获取当前权重
#     def get_weights(self, device):
#         eps = 0.05  # 最小权重
#         max_w = 0.9  # 最大权重
#         weights = F.softmax(1 / (self.loss_ema + 1e-6), dim=0)
#         weights = torch.clamp(weights, min=eps)
#         weights = weights / weights.sum()  # 重新归一化
#         weights = torch.clamp(weights, max=max_w)
#         weights = weights / weights.sum()  # 再归一化一次
#         return weights.to(device)


def pretrain_on_sources(model, data_loader, epochs, device="cuda", domain_weight=0.01):
    """
    Stage 1: Adversarial Pretraining (DANN)
    - 使用源城市的 task_loss (L1)
    - 同时使用源 + 目标的 domain_loss（由 MSDAN 内部实现）
    - 本阶段不计算 MMD
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = {'pretrain_loss': []}

    # 初始化 GRL 调度
    model.global_step = 0
    model.total_steps = sum(len(loader) for loader in data_loader.source_train_loaders.values()) * epochs
    model.grl_schedule = True

    # 准备目标城市迭代器
    target_iter = (
        iter(data_loader.target_train_loader)
        if hasattr(data_loader, "target_train_loader")
        else None
    )

    print("Starting adversarial pretraining (DANN) ...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        # --- 遍历所有源城市 ---
        for domain_idx, (city, loader) in enumerate(data_loader.source_train_loaders.items()):
            for x_src, y_src in loader:

                # ====== 1. 获取目标城 batch（循环取） ======
                if target_iter is not None:
                    try:
                        x_tgt, y_tgt = next(target_iter)
                    except StopIteration:
                        target_iter = iter(data_loader.target_train_loader)
                        x_tgt, y_tgt = next(target_iter)

                    x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)
                else:
                    x_tgt = None

                x_src, y_src = x_src.to(device), y_src.to(device)

                optimizer.zero_grad()

                # ====== 2. 源域前向（返回 pred & domain_loss） ======
                pred_src, domain_loss_src, _ = model(
                    x_src,
                    domain_idx=domain_idx,
                    is_target=False,
                )

                task_loss_src = F.l1_loss(pred_src, y_src)

                # ====== 3. 目标域仅用于 domain_loss（模型内部处理） ======
                if x_tgt is not None:
                    pred_tgt, domain_loss_tgt, _ = model(
                        x_tgt,
                        domain_idx=None,
                        is_target=True,
                    )
                else:
                    domain_loss_tgt = torch.tensor(0.0, device=device)

                # ====== 4. 总域损失（多类、由 MSDAN 自动处理） ======
                domain_loss = domain_loss_src + domain_loss_tgt

                # ====== 5. 总损失：task + λ * domain ======
                loss = task_loss_src + domain_weight * domain_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                model.global_step += 1

        # 更新 epoch 结果
        avg_loss = total_loss / max(batch_count, 1)
        history['pretrain_loss'].append(avg_loss)

        # ====== 6. 每 10 epoch 验证 ======
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Time={time.time()-start_time:.1f}s")
            for city, loader in data_loader.source_val_loaders.items():
                max_val, min_val = data_loader.get_norm_params(city)
                mae, rmse, _ = evaluate(model, loader, device, target=False,
                                        max_val=max_val, min_val=min_val)
                print(f"  {city} Val MAE={mae:.4f}, RMSE={rmse:.4f}")

        # 清理缓存（如果你在模型里存了某些跨 batch 特征）
        for i in range(model.num_sources):
            model.source_feats[i].clear()

    return history




def finetune_on_target(model, data_loader, epochs, device="cuda", mmd_weight=0.05, target_city=None):
    """
    Stage 2: Fine-grained Distribution Alignment (MMD) on target domain.
    使用目标域带标签样本 + 源域缓存特征进行 NodeSim-Aware MMD 对齐。
    同时记录每 epoch 的 train/val MAE & RMSE（反归一化后），以便绘图与报告。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = {
        'finetune_loss': [],
        'train_mae': [], 'train_rmse': [],
        'val_mae': [],   'val_rmse': []
    }

    # get normalization params for denorm (if available)
    if target_city is not None:
        try:
            max_val, min_val = data_loader.get_norm_params(target_city)
        except Exception:
            max_val, min_val = None, None
    else:
        max_val, min_val = None, None

    print("Starting fine-tuning (MMD alignment on target)...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for x_tgt, y_tgt in data_loader.target_train_loader:
            x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)
            optimizer.zero_grad()

            # 目标域 forward: 会自动触发 MMD，对齐到源域缓存的特征
            pred, _, mmd_loss = model(x_tgt, domain_idx=None, is_target=True)
            task_loss = F.l1_loss(pred, y_tgt)

            # 微调阶段 domain_loss 可忽略或权重设为 0
            loss = task_loss + mmd_weight * mmd_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        history['finetune_loss'].append(avg_loss)

        # === 在每个 epoch 后计算 train/val 指标（反归一化） ===
        model.eval()
        with torch.no_grad():
            # train metrics on target training set (use evaluate utility)
            train_mae, train_rmse, _ = evaluate(
                model=model,
                loader=data_loader.target_train_loader,
                device=device,
                target=True,
                max_val=max_val,
                min_val=min_val
            )
            val_mae, val_rmse, _ = evaluate(
                model=model,
                loader=data_loader.target_val_loader,
                device=device,
                target=True,
                max_val=max_val,
                min_val=min_val
            )

        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        if (epoch + 1) % 5 == 0:
            print(f"Finetune Epoch {epoch+1}: loss={avg_loss:.4f}, "
                  f"train MAE={train_mae:.4f}, train RMSE={train_rmse:.4f}, "
                  f"val MAE={val_mae:.4f}, val RMSE={val_rmse:.4f}, "
                  f"time={time.time() - start_time:.1f}s")

    return model, history



def test_on_target(model, data_loader, device="cuda", city="Unknown"):
    """
        Stage 3: 测试阶段（不启用 DANN / MMD）。
    """
    print(f"\n=== Testing on target city: {city} ===")
    model.eval()

    max_val, min_val = data_loader.get_norm_params(city)

    test_mae, test_rmse, preds = evaluate(
        model=model,
        loader=data_loader.target_test_loader,
        device=device,
        max_val=max_val,
        min_val=min_val
    )

    print(f"[Test] {city} → MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
    return test_mae, test_rmse, preds


def min_max_denormalize(norm_data, max_val, min_val):
    return norm_data * (max_val - min_val) + min_val


def evaluate(model, loader, device="cuda", target=False, max_val=None, min_val=None):
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_elements = 0
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, "x") and hasattr(batch, "y"):
                x, y = batch.x.to(device), batch.y.to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)

            # forward
            pred, _, _ = model(x, domain_idx=None, is_target=False)

            # denormalize if possible
            if max_val is not None and min_val is not None:
                y_denorm = min_max_denormalize(y, max_val, min_val)
                pred_denorm = min_max_denormalize(pred, max_val, min_val)
            else:
                y_denorm = y
                pred_denorm = pred

            preds_list.append(pred_denorm.detach().cpu())

            # sum over all elements in this batch
            batch_mae_sum = F.l1_loss(pred_denorm, y_denorm, reduction="sum").item()
            batch_mse_sum = F.mse_loss(pred_denorm, y_denorm, reduction="sum").item()

            total_mae += batch_mae_sum
            total_mse += batch_mse_sum
            total_elements += y_denorm.numel()

    # average per element
    if total_elements == 0:
        return 0.0, 0.0, preds_list

    avg_mae = total_mae / total_elements
    avg_rmse = math.sqrt(total_mse / total_elements)

    return avg_mae, avg_rmse, preds_list



def generate_ablation_report(experiment_results, target_city):
    """生成 mmd_weight 消融实验报告"""
    print("\n" + "=" * 80)
    print(f"MMD WEIGHT ABLATION REPORT - Target City: {target_city}")
    print("=" * 80)

    print(
        f"{'MMD Weight':<12} "
        f"{'Final MAE':<12} "
        f"{'Final RMSE':<12} "
        f"{'Best MAE':<12} "
        f"{'Best RMSE':<12} "
        f"{'Overfitting Gap':<15}"
    )
    print("-" * 80)

    for mmd_weight, results in experiment_results.items():
        print(
            f"{mmd_weight:<12} "
            f"{results['final_test_mae']:<12.4f} "
            f"{results['final_test_rmse']:<12.4f} "
            f"{results['best_test_mae']:<12.4f} "
            f"{results['best_test_rmse']:<12.4f} "
            f"{results['overfitting_gap']:<15.4f}"
        )

    # 找最优
    best_weight = min(
        experiment_results.keys(),
        key=lambda w: experiment_results[w]['best_test_rmse']
    )

    print("\nBest Configuration:")
    print(f"  - Best mmd_weight: {best_weight}")
    print(f"  - Best Test RMSE: {experiment_results[best_weight]['best_test_rmse']:.4f}")
    print(f"  - Best Test MAE: {experiment_results[best_weight]['best_test_mae']:.4f}")

    # 保存到文件
    with open(f"mmd_weight_ablation_{target_city}.txt", "w") as f:
        f.write(f"MMD Weight Ablation Report - {target_city}\n")
        f.write(f"Best mmd_weight: {best_weight}\n")
        f.write(f"Best RMSE: {experiment_results[best_weight]['best_test_rmse']:.4f}\n")
        f.write(f"Best MAE: {experiment_results[best_weight]['best_test_mae']:.4f}\n")

    print(f"\nReport saved to: mmd_weight_ablation_{target_city}.txt")


# ========================== 主程序 ==========================

if __name__ == "__main__":

    all_cities = ['Washington', 'Chicago', 'NYCBike', 'LABike']
    domain_weights = [0, 0.01, 0.05, 0.1, 0.2]

    experiment_results = {}

    for target_city in all_cities:
        source_cities = [c for c in all_cities if c != target_city]
        print(f"\nRunning MMD-weight ablation: Target={target_city}, Sources={source_cities}")

        data_loader = MultiSourceDataLoader(source_cities, target_city)
        city_graph_data = [data_loader.get_city_graph_data(c) for c in source_cities + [target_city]]
        target_graph = city_graph_data[-1]

        # ---------- 城市级相似度 ----------
        sim_weights = []
        for i in range(len(source_cities)):
            src_graph = city_graph_data[i]
            sim = compute_city_similarity_simple(src_graph, target_graph)
            sim_weights.append(sim)

        sum_sim = sum(sim_weights)
        if sum_sim <= 0:
            sim_weights = [1.0 / len(sim_weights)] * len(sim_weights)
        else:
            sim_weights = [s / sum_sim for s in sim_weights]

        # ---------- 节点级相似度 ----------
        node_sim_matrices = []
        for i in range(len(source_cities)):
            src_graph = city_graph_data[i]
            sim_matrix = compute_node_level_similarity_full(src_graph, target_graph)
            node_sim_matrices.append(sim_matrix)

        experiment_results[target_city] = {}

        # ====================== MMD Weight 遍历 ======================
        for domain_weight in domain_weights:
            print(f"\n--- Training with domain_weight = {domain_weight} ---")

            model = MSDAN(
                in_feats=4,
                h_dim=64,
                city_graph_data_list=city_graph_data,
                device=device,
                use_fusion=True,
                use_gcn=True,
                use_temporal=True,
                use_dan=True,
                num_sources=len(source_cities),
                sim_weights=sim_weights,
                node_sim_matrices=node_sim_matrices,
                mmd_mode="nodesim",
                dropout_rate=0.3
            ).to(device)

            # ---------- Stage 1: DANN ----------
            pretrain_on_sources(
                model,
                data_loader,
                epochs=150,
                device=device,
                domain_weight=domain_weight
            )

            # ---------- Stage 2: NodeSim-MMD ----------
            finetuned_model, finetune_history = finetune_on_target(
                model,
                data_loader,
                epochs=60,
                device=device,
                mmd_weight=0.1,
                target_city=target_city
            )

            # ---------- Test ----------
            test_mae, test_rmse, _ = test_on_target(
                finetuned_model,
                data_loader,
                device=device,
                city=target_city
            )

            print(
                f"[Test Result] Target={target_city}, "
                f"domain_weight={domain_weight}, "
                f"RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
            )

            experiment_results[target_city][domain_weight] = {
                "final_test_rmse": test_rmse,
                "final_test_mae": test_mae,
                "best_test_rmse": min(finetune_history['val_rmse']),
                "best_test_mae": min(finetune_history['val_mae']),
                "overfitting_gap": finetune_history['val_rmse'][-1]
                                   - finetune_history['train_rmse'][-1]
            }

        # ---------- 单城市报告 ----------
        generate_ablation_report(experiment_results[target_city], target_city)
