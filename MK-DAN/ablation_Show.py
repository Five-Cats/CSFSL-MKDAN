import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 从你现有代码 import =====
from main import (
    MSDAN,
    MultiSourceDataLoader,
    compute_city_similarity_simple,
    compute_node_level_similarity_full,
    device
)

# ======================
# 提取特征函数
# ======================
def extract_features(model, loader, device, domain_idx=None, is_target=False, max_samples=2000):
    model.eval()
    feats = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            _, _, _, feat = model(
                x,
                domain_idx=domain_idx,
                is_target=is_target
            )  # feat: (B, N, D)

            feat = feat.mean(dim=1)  # (B, D)
            feats.append(feat.cpu().numpy())

            if sum(len(f) for f in feats) > max_samples:
                break

    feats = np.concatenate(feats, axis=0)
    return feats


# ======================
# KDE绘图函数
# ======================
def plot_kde(source_feats_dict, target_feats, title, save_path):
    plt.figure(figsize=(7, 5))

    # 目标域
    sns.kdeplot(
        target_feats.flatten(),
        label="Target",
        linewidth=2
    )

    # 源域
    for city, feat in source_feats_dict.items():
        sns.kdeplot(
            feat.flatten(),
            label=f"Source-{city}",
            linestyle="--"
        )

    plt.title(title)
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ======================
# 主函数
# ======================
if __name__ == "__main__":

    target_city = "LABike"
    all_cities = ['Washington', 'Chicago', 'NYCBike', 'LABike']
    source_cities = [c for c in all_cities if c != target_city]

    print("Loading data...")

    data_loader = MultiSourceDataLoader(source_cities, target_city)

    city_graph_data = [
        data_loader.get_city_graph_data(c)
        for c in source_cities + [target_city]
    ]

    target_graph = city_graph_data[-1]

    # ===== 城市相似度 =====
    sim_weights = []
    for i in range(len(source_cities)):
        sim = compute_city_similarity_simple(
            city_graph_data[i],
            target_graph
        )
        sim_weights.append(sim)

    sim_weights = np.array(sim_weights)
    sim_weights = sim_weights / sim_weights.sum()

    # ===== 节点相似度 =====
    node_sim_matrices = []
    for i in range(len(source_cities)):
        sim_matrix = compute_node_level_similarity_full(
            city_graph_data[i],
            target_graph
        )
        node_sim_matrices.append(sim_matrix)

    # ======================
    # 两个模型：none vs nodesim
    # ======================
    for mmd_mode in ["none", "nodesim"]:

        print(f"\nLoading model: {mmd_mode}")

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
            mmd_mode=mmd_mode
        ).to(device)

        checkpoint = torch.load(
            f"model_{target_city}_{mmd_mode}.pth",
            map_location=device
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 兼容你未来可能直接保存 state_dict 的情况
            model.load_state_dict(checkpoint)

        # ===== 提取 target 特征 =====
        target_feats = extract_features(
            model,
            data_loader.target_test_loader,
            device,
            domain_idx=None,
            is_target=True
        )

        # ===== 提取 source 特征 =====
        source_feats_dict = {}
        for i, city in enumerate(source_cities):
            feats = extract_features(
                model,
                data_loader.source_test_loaders[city],
                device,
                domain_idx=i,
                is_target=False
            )
            source_feats_dict[city] = feats

        # ===== 画 KDE =====
        plot_kde(
            source_feats_dict,
            target_feats,
            title=f"KDE - {mmd_mode}",
            save_path=f"kde_{target_city}_{mmd_mode}.png"
        )