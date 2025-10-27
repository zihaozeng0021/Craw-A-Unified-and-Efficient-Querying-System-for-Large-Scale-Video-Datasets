import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from msu import build_msu

def cosine_custom(vec1: np.ndarray, vec2: np.ndarray) -> float:
    min_len = min(len(vec1), len(vec2))
    vec1_cut = vec1[:min_len]
    vec2_cut = vec2[:min_len]
    
    dot_product = np.dot(vec1_cut, vec2_cut)
    norm1 = np.linalg.norm(vec1_cut)
    norm2 = np.linalg.norm(vec2_cut)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return 1 - (dot_product / (norm1 * norm2)) 

def distance(p1: list, p2: list) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

def TMD(traj1: list, traj2: list) -> float:
    vec1 = np.array(traj1).flatten()
    vec2 = np.array(traj2).flatten()
    return cosine_custom(vec1, vec2)  

def F_info(row: np.ndarray) -> tuple[int, list]:
    class_id = int(row[0])
    coords = row[1:]
    trajectory = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
    return class_id, trajectory

def group_class(F: list) -> dict[int, list]:
    groups = defaultdict(list)
    for row in F:
        class_id, traj = F_info(row)
        groups[class_id].append(traj)
    return groups

def match_traj(group1: list, group2: list, w1: float = 0.5, w2: float = 0.5) -> list[tuple[int, int]]:
    matched_pairs = []
    used_indices = set()
    
    for i, traj1 in enumerate(group1):
        best_j = None
        best_score = float('inf')
        start1, end1 = traj1[0], traj1[-1]
        
        for j, traj2 in enumerate(group2):
            if j in used_indices:
                continue
                
            start2, end2 = traj2[0], traj2[-1]
            dist_start = distance(start1, start2)
            dist_end = distance(end1, end2)
            score = w1 * dist_start + w2 * dist_end
            
            if score < best_score:
                best_score = score
                best_j = j
                
        if best_j is not None:
            matched_pairs.append((i, best_j))
            used_indices.add(best_j)
            
    return matched_pairs

def pm_tsmd(FA: np.ndarray, FB: np.ndarray, base_penalty: float = 1.0) -> float:
    groups_A = group_class(FA)
    groups_B = group_class(FB)
    result_distance = 0.0
    total_pairs = 0

    for class_id in groups_A:
        if class_id not in groups_B:
            result_distance += base_penalty * len(groups_A[class_id])
            continue
            
        group_A = groups_A[class_id]
        group_B = groups_B[class_id]
        matched_pairs = match_traj(group_A, group_B)
        
        match_rate = len(matched_pairs) / max(len(group_A), len(group_B))
        penalty_weight = 1.0 - match_rate
        
        for i, j in matched_pairs:
            dist = TMD(group_A[i], group_B[j])
            result_distance += dist
            total_pairs += 1
        
        unmatched_A = len(group_A) - len(matched_pairs)
        unmatched_B = len(group_B) - len(matched_pairs)
        result_distance += base_penalty * penalty_weight * (unmatched_A + unmatched_B)
    
    return result_distance / max(total_pairs, 1)

def get_cluster_centers(cluster_samples, F_raw, distance_matrix):
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
            'feature': F_raw[best_center],
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
    F_raw = [np.array(m[0]) for m in MSU]
    n_samples = len(F_raw)
    
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = pm_tsmd(F_raw[i], F_raw[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    n_clusters = 4
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
    centers = get_cluster_centers(cluster_samples, F_raw, distance_matrix)
    
    # 整合每个簇的信息：包含簇ID、所属样本ID列表和代表中心
    cluster_info = []
    for cluster_id in sorted(cluster_samples.keys()):
        cluster_info.append({
            'cluster_id': cluster_id,
            'ids': cluster_samples[cluster_id],  # 该簇包含的所有样本ID
            'center': centers[cluster_id],       # 该簇的代表中心信息
            'size': len(cluster_samples[cluster_id])  # 簇大小
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
    