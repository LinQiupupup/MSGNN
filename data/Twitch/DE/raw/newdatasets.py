import pandas as pd
import numpy as np
import json
import os
from sklearn.decomposition import PCA


def generate_twitch_npz_pca(dataset_name='DE'):
    # === 配置 ===
    target_file = f'musae_{dataset_name}_target.csv'
    edges_file = f'musae_{dataset_name}_edges.csv'
    # 自动尝试两种常见的 JSON 文件名
    json_file = f'musae_{dataset_name}.json'
    if not os.path.exists(json_file):
        json_file = f'musae_{dataset_name}_features.json'

    output_file = f'{dataset_name}.npz'

    print(f"=== 正在处理 Twitch-{dataset_name} (带 PCA 降维) ===")

    # 1. 读取标签
    print("1. 读取节点标签...")
    df_target = pd.read_csv(target_file)
    # 确保按 new_id 排序
    if 'new_id' in df_target.columns:
        df_target = df_target.sort_values('new_id').reset_index(drop=True)
        node_ids = df_target['new_id'].values
    else:
        node_ids = np.arange(len(df_target))

    # 获取 mature 列作为标签
    if 'mature' in df_target.columns:
        labels = df_target['mature'].values.astype(np.int64)
    else:
        raise ValueError("Target CSV 中找不到 'mature' 列")

    id_map = {raw_id: i for i, raw_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    print(f"   - 节点数量: {num_nodes}")

    # 2. 读取原始特征 (Multi-hot)
    print("2. 读取特征并构建矩阵...")
    with open(json_file, 'r') as f:
        feat_data = json.load(f)

    # 统计所有特征 ID
    all_features = set()
    for flist in feat_data.values():
        all_features.update(flist)
    unique_feats = sorted(list(all_features))
    feat_map = {raw_f: i for i, raw_f in enumerate(unique_feats)}
    raw_dim = len(unique_feats)
    print(f"   - 原始特征维度: {raw_dim} (即你看到的 2514)")

    # 构建原始稀疏矩阵的稠密表示 (Dense)
    # 2514维不算大，内存应该够
    features_raw = np.zeros((num_nodes, raw_dim), dtype=np.float32)

    for raw_id_str, flist in feat_data.items():
        try:
            raw_id = int(raw_id_str)
        except:
            raw_id = raw_id_str

        if raw_id in id_map:
            idx = id_map[raw_id]
            f_indices = [feat_map[f] for f in flist]
            features_raw[idx, f_indices] = 1.0

    # 3. 执行 PCA 降维 -> 128
    print("3. 执行 PCA 降维 (2514 -> 128)...")
    pca = PCA(n_components=128)
    features_reduced = pca.fit_transform(features_raw)
    print(f"   - 降维后特征形状: {features_reduced.shape}")

    # 4. 读取边并处理 (转为 PyG 偏好的 2 x E 格式)
    print("4. 读取边列表...")
    df_edges = pd.read_csv(edges_file)
    src_col, dst_col = df_edges.columns[0], df_edges.columns[1]

    edges_list = []
    for _, row in df_edges.iterrows():
        u, v = row[src_col], row[dst_col]
        if u in id_map and v in id_map:
            edges_list.append([id_map[u], id_map[v]])

    # 转为 (2, E) 格式，这是 PyG 的标准输入格式
    edges = np.array(edges_list, dtype=np.int64).T
    print(f"   - 边数量: {edges.shape[1]}")

    # 5. 保存
    np.savez(output_file, features=features_reduced, target=labels, edges=edges)
    print(f"5. 文件已保存: {output_file}")
    print("   请将其移动到 data/Twitch/DE/raw/ 目录下，并删除 processed 文件夹后重试。")


if __name__ == '__main__':
    generate_twitch_npz_pca('DE')