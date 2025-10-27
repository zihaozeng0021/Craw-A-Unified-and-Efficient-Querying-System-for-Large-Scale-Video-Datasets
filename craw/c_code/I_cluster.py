import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from collections import Counter, defaultdict
from msu import build_msu
import time
def matrix_analysis(IA, IB):
    V1 = count_matrix(IA)
    V2 = count_matrix(IB)
    sum1 = sum(V1)
    sum2 = sum(V2)
    
    if sum1 == 0:
        sum1 = 1
    if sum2 == 0:
        sum2 = 1
            
    v1 = np.round(V1 / sum1, 2) 
    v2 = np.round(V2 / sum2, 2)
    return np.round(compute_manhattan(v1, v2), 3)

def count_matrix(I):
    V = np.zeros((5,), dtype=int)
    for val in I.flatten():
        if 0 <= val <= 4:
            V[int(val)] += 1
    return V

def compute_manhattan(V1, V2):
    return np.sum(np.abs(V1 - V2))

def get_cluster_centers(cluster_samples, I_raw, distance_matrix):
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
            'feature': I_raw[best_center],
            'distance_sum': min_dist_sum
        }
    return centers

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

def main():
    F, I, T = build_msu()
    MSU = list(zip(F, I, T))
    I_raw = [np.array(m[1]) for m in MSU]
    n_samples = len(I_raw)
    
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = matrix_analysis(I_raw[i], I_raw[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    n_clusters = 3
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 收集每个簇的ID列表
    cluster_samples = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_samples[lab].append(idx)
    
    # 计算每个簇的中心
    centers = get_cluster_centers(cluster_samples, I_raw, distance_matrix)
    
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
    