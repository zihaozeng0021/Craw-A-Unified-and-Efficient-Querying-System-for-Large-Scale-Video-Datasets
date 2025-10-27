from msu import build_msu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from collections import Counter, defaultdict

# -------------------------- 数据预处理 --------------------------
def pad_matrix(T: np.ndarray, target_rows: int) -> np.ndarray:
    padded = np.zeros((target_rows, 2))
    padded[:T.shape[0], :] = T  # 填充原始数据
    if T.shape[0] < target_rows:
        padded[T.shape[0]:, :] = [0, 1]  # 填充缺失值
    return padded

# -------------------------- CTMD距离计算 --------------------------
def compute_ctmd(TA: np.ndarray, TB: np.ndarray, max_rows: int) -> float:
    """计算两个轨迹矩阵的CTMD距离"""
    # 恢复为原始矩阵形状
    TA = TA.reshape(max_rows, 2)
    TB = TB.reshape(max_rows, 2)
    
    # 提取u和v分量
    Du1, Dv1 = TA[:, 0], TA[:, 1]
    Du2, Dv2 = TB[:, 0], TB[:, 1]

    def emd_1d(d1, d2):
        """计算一维Earth Mover's Distance"""
        cost_matrix = np.abs(d1[:, None] - d2[None, :])  # 构建成本矩阵
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 最优匹配
        return cost_matrix[row_ind, col_ind].sum()  # 求和得到EMD

    # 计算u和v方向的EMD并加权
    emd_u = emd_1d(Du1, Du2)
    emd_v = emd_1d(Dv1, Dv2)
    L = 1.0 / max_rows  # 权重因子
    return L * (emd_u + emd_v) / 2.0

# -------------------------- 聚类中心选择 --------------------------
def get_cluster_centers(cluster_samples, T_raw, distance_matrix):
    centers = {}
    for cluster_id, sample_indices in cluster_samples.items():
        min_dist_sum = float('inf')
        best_center = None
        
        for idx in sample_indices:
            dist_sum = sum(distance_matrix[idx, other_idx] for other_idx in sample_indices)
            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                best_center = idx
                
        centers[cluster_id] = {
            'index': best_center,
            'feature': T_raw[best_center],
            'distance_sum': min_dist_sum
        }
    return centers

# -------------------------- 可视化聚类结果 --------------------------
def visualize_clusters(distance_matrix: np.ndarray, labels: np.ndarray, n_clusters: int, 
                       cluster_info: list):
    fig = plt.figure(figsize=(16, 14))
    
    ax1 = fig.add_subplot(221)
    im = ax1.imshow(distance_matrix, cmap='viridis')
    ax1.set_title("Distance Matrix Heatmap")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Sample Index")
    fig.colorbar(im, ax=ax1)
    
    ax2 = fig.add_subplot(222)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1)
    mds_results = mds.fit_transform(distance_matrix)
    
    scatter = ax2.scatter(mds_results[:, 0], mds_results[:, 1], c=labels, 
                         cmap=plt.colormaps['viridis'], s=100, alpha=0.7)
    
    for info in cluster_info:
        center_idx = info['center']['index']
        ax2.scatter(mds_results[center_idx, 0], mds_results[center_idx, 1],
                   s=200, marker='*', edgecolors='red', linewidths=2,
                   label=f'Cluster {info["cluster_id"]} Center')
    
    ax2.set_title("Clustering Results with Centers")
    ax2.set_xlabel("MDS Dimension 1")
    ax2.set_ylabel("MDS Dimension 2")
    plt.colorbar(scatter, ax=ax2, label='Cluster Label')
    ax2.legend()
    
    ax3 = fig.add_subplot(223)
    cluster_counts = {info['cluster_id']: len(info['ids']) for info in cluster_info}
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[cluster] for cluster in clusters]
    
    viridis = plt.colormaps['viridis']
    colors = viridis(np.linspace(0, 1, n_clusters))
    
    ax3.bar(clusters, counts, color=colors)
    ax3.set_title("Sample Distribution by Cluster")
    ax3.set_xlabel("Cluster Label")
    ax3.set_ylabel("Number of Samples")
    ax3.set_xticks(clusters)
    
    ax4 = fig.add_subplot(224)
    for i, (x, y) in enumerate(mds_results):
        ax4.text(x, y, str(i), fontsize=9)
    
    ax4.scatter(mds_results[:, 0], mds_results[:, 1], c=labels, 
               cmap=plt.colormaps['viridis'], alpha=0.3, s=100)
    ax4.set_title("Sample Indices and Clusters")
    ax4.set_xlabel("MDS Dimension 1")
    ax4.set_ylabel("MDS Dimension 2")
    
    plt.tight_layout()
    plt.show(block=True)

# -------------------------- 主函数 --------------------------
def main():
    # 加载数据
    F, I, T = build_msu()  # 从msu模块获取数据
    MSU = list(zip(F, I, T))
    
    # 提取T特征并统一形状
    T_raw = [np.array(m[2]) for m in MSU]  # 原始T特征列表
    max_rows = max([t.shape[0] for t in T_raw])  # 找到最大行数用于填充
    
    # 填充并展平（用于聚类输入）
    T_padded = [pad_matrix(t, max_rows) for t in T_raw]
    T_flattened = np.array([t.flatten() for t in T_padded])  # 展平为二维数组
    
    # 创建距离函数（带参数）
    def ctmd_metric(X, Y=None):
        if Y is None:
            return pairwise_distances(X, metric=lambda a, b: compute_ctmd(a, b, max_rows))
        else:
            return np.array([compute_ctmd(x, y, max_rows) for x, y in zip(X, Y)])
    
    # 计算距离矩阵（用于后续中心选择和可视化）
    n_samples = len(T_raw)
    distance_matrix = ctmd_metric(T_flattened)
    
    # 执行层次聚类
    n_clusters = 3  # 目标聚类数量
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=ctmd_metric,  # 使用兼容的CTMD距离函数
        linkage='average'    # 平均链接方式
    )
    labels = clustering.fit_predict(T_flattened)
    
    # 收集每个簇的ID列表
    cluster_samples = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_samples[lab].append(idx)
    
    # 计算每个簇的中心
    centers = get_cluster_centers(cluster_samples, T_raw, distance_matrix)
    
    # 整合每个簇的信息
    cluster_info = []
    for cluster_id in sorted(cluster_samples.keys()):
        cluster_info.append({
            'cluster_id': cluster_id,
            'ids': cluster_samples[cluster_id],
            'center': centers[cluster_id],
            'size': len(cluster_samples[cluster_id])
        })
    
    # 打印每个簇的信息
    for info in cluster_info:
        print(f"Cluster {info['cluster_id']}:")
        print(f"  Size: {info['size']} samples")
        print(f"  Representative center index: {info['center']['index']}")
        print(f"  Member IDs: {info['ids']}\n")
    
    # 可视化聚类结果
    visualize_clusters(distance_matrix, labels, n_clusters, cluster_info)
    
    return cluster_info

if __name__ == "__main__":
    cluster_info = main()
    