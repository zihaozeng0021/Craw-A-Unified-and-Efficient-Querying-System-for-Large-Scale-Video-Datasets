import numpy as np
import time
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from msu import build_msu  
from T_cluster import pad_matrix, compute_ctmd
from I_cluster import count_matrix, compute_manhattan
from F_cluster import pm_tsmd

def load_features():
    start = time.perf_counter()
    F, I, T = build_msu()
    T_raw = [np.array(t) for t in T]
    max_rows = max([t.shape[0] for t in T_raw]) if T_raw else 0
    T_padded = [pad_matrix(t, max_rows) for t in T_raw]
    T_flattened = np.array([t.flatten() for t in T_padded])
    
    end = time.perf_counter()
    return {
        "F": [np.array(f) for f in F],
        "I": [np.array(i) for i in I],
        "T_raw": T_raw,
        "T_flattened": T_flattened,
        "T_max_rows": max_rows,
        "sample_count": len(F)  
    }

start_load = time.perf_counter()
FEATURES = load_features()
end_load = time.perf_counter()



def matrix_analysis(IA, IB):
    V1 = count_matrix(IA)
    V2 = count_matrix(IB)
    sum1 = sum(V1) or 1 
    sum2 = sum(V2) or 1
    return compute_manhattan(V1 / sum1, V2 / sum2)


def get_T_labels():
    start = time.perf_counter()
    
    def ctmd_metric(X, Y=None):
        if Y is None:
            return pairwise_distances(
                X, 
                metric=lambda a, b: compute_ctmd(a, b, FEATURES["T_max_rows"])
            )
        else:
            return np.array([
                compute_ctmd(x, y, FEATURES["T_max_rows"]) 
                for x, y in zip(X, Y)
            ])
    
    # 计算距离矩阵
    distance_matrix = ctmd_metric(FEATURES["T_flattened"])
    clustering = AgglomerativeClustering(
        n_clusters=3,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 计算每个簇的代表中心
    representatives = {}
    for cluster_id in np.unique(labels):
        cluster_samples = np.where(labels == cluster_id)[0]
        if len(cluster_samples) == 1:
            representatives[cluster_id] = cluster_samples[0]
            continue
        
        # 计算簇内每个样本与其他样本的平均距离
        avg_distances = []
        for sample in cluster_samples:
            dist_sum = np.sum(distance_matrix[sample, cluster_samples])
            avg_dist = dist_sum / (len(cluster_samples) - 1)
            avg_distances.append((sample, avg_dist))
        
        # 选择平均距离最小的样本作为代表
        avg_distances.sort(key=lambda x: x[1])
        representatives[cluster_id] = avg_distances[0][0]
    
    end = time.perf_counter()
    print(f"T特征聚类耗时: {end - start:.2f}s")
    return labels, representatives, distance_matrix


def get_I_labels():
    start = time.perf_counter()
    I_features = FEATURES["I"]
    n_samples = FEATURES["sample_count"]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = matrix_analysis(I_features[i], I_features[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    clustering = AgglomerativeClustering(
        n_clusters=3,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    

    representatives = {}
    for cluster_id in np.unique(labels):
        cluster_samples = np.where(labels == cluster_id)[0]
        if len(cluster_samples) == 1:
            representatives[cluster_id] = cluster_samples[0]
            continue
        
        avg_distances = []
        for sample in cluster_samples:
            dist_sum = np.sum(distance_matrix[sample, cluster_samples])
            avg_dist = dist_sum / (len(cluster_samples) - 1)
            avg_distances.append((sample, avg_dist))
        
        avg_distances.sort(key=lambda x: x[1])
        representatives[cluster_id] = avg_distances[0][0]
    
    end = time.perf_counter()
    print(f" {end - start:.2f}s")
    return labels, representatives, distance_matrix


def get_F_labels():
    start = time.perf_counter()
    F_features = FEATURES["F"]
    n_samples = FEATURES["sample_count"]
    distance_matrix = np.zeros((n_samples, n_samples))
    

    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = pm_tsmd(F_features[i], F_features[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    clustering = AgglomerativeClustering(
        n_clusters=4,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 计算每个簇的代表中心
    representatives = {}
    for cluster_id in np.unique(labels):
        cluster_samples = np.where(labels == cluster_id)[0]
        if len(cluster_samples) == 1:
            representatives[cluster_id] = cluster_samples[0]
            continue
        
        avg_distances = []
        for sample in cluster_samples:
            dist_sum = np.sum(distance_matrix[sample, cluster_samples])
            avg_dist = dist_sum / (len(cluster_samples) - 1)
            avg_distances.append((sample, avg_dist))
        
        avg_distances.sort(key=lambda x: x[1])
        representatives[cluster_id] = avg_distances[0][0]
    
    end = time.perf_counter()
    print(f"{end - start:.2f}s")
    return labels, representatives, distance_matrix


def calculate_intra_cluster_avg_dist():
    start = time.perf_counter()
    
    T_labels, _, _ = get_T_labels()
    I_labels, _, _ = get_I_labels()
    F_labels, _, _ = get_F_labels()
    
    t_intra_dist = defaultdict(list)
    T_features = FEATURES["T_flattened"]
    for i in range(FEATURES["sample_count"]):
        for j in range(i + 1, FEATURES["sample_count"]):
            if T_labels[i] == T_labels[j]:
                t_i = T_features[i].reshape(FEATURES["T_max_rows"], 2)
                t_j = T_features[j].reshape(FEATURES["T_max_rows"], 2)
                dist = compute_ctmd(t_i, t_j, FEATURES["T_max_rows"])
                t_intra_dist[T_labels[i]].append(dist)
    
    i_intra_dist = defaultdict(list)
    I_features = FEATURES["I"]
    for i in range(FEATURES["sample_count"]):
        for j in range(i + 1, FEATURES["sample_count"]):
            if I_labels[i] == I_labels[j]:
                dist = matrix_analysis(I_features[i], I_features[j])
                i_intra_dist[I_labels[i]].append(dist)
    
    f_intra_dist = defaultdict(list)
    F_features = FEATURES["F"]
    for i in range(FEATURES["sample_count"]):
        for j in range(i + 1, FEATURES["sample_count"]):
            if F_labels[i] == F_labels[j]:
                dist = pm_tsmd(F_features[i], F_features[j])
                f_intra_dist[F_labels[i]].append(dist)

    def get_global_avg(dist_dict):
        all_dists = [d for dist_list in dist_dict.values() for d in dist_list]
        return np.mean(all_dists) if all_dists else 1.0 
    
    end = time.perf_counter()
    print(f"{end - start:.2f}s")
    
    return {
        "T_intra_avg": get_global_avg(t_intra_dist),
        "I_intra_avg": get_global_avg(i_intra_dist),
        "F_intra_avg": get_global_avg(f_intra_dist)
    }


def build_hierarchical_index():

    start = time.perf_counter()
    
    T_labels, T_reps, _ = get_T_labels()
    I_labels, I_reps, _ = get_I_labels()
    F_labels, F_reps, _ = get_F_labels()
    n_samples = FEATURES["sample_count"]

    assert len(I_labels) == n_samples and len(F_labels) == n_samples, \
        "聚类标签样本数量不一致！"
    assert len(T_reps) == 3 and len(I_reps) == 3 and len(F_reps) == 4, \
        "代表中心数量与聚类数不匹配！"
    
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx in range(n_samples):
        t = T_labels[sample_idx]
        i = I_labels[sample_idx]
        f = F_labels[sample_idx]
        index[t][i][f].append(sample_idx)  
    
    end = time.perf_counter()

    return index, T_labels, I_labels, F_labels, T_reps, I_reps, F_reps


def print_index_statistics(index, T_reps, I_reps, F_reps):
    start = time.perf_counter()
    

    for cluster_id, rep_idx in sorted(T_reps.items()):
        print(f"  T_cluster_{cluster_id}: sample_index_{rep_idx}")
    
    for cluster_id, rep_idx in sorted(I_reps.items()):
        print(f"  I_cluster_{cluster_id}: sample_index_{rep_idx}")
    
    for cluster_id, rep_idx in sorted(F_reps.items()):
        print(f"  F_cluster_{cluster_id}: sample_index_{rep_idx}")
    for t_cluster in sorted(index.keys()):
        t_total = sum(
            len(samples) 
            for i_cluster in index[t_cluster].values() 
            for samples in i_cluster.values()
        )
        print(f"\nT_cluster_{t_cluster} (总样本: {t_total})") 
        
        for i_cluster in sorted(index[t_cluster].keys()):
            i_total = sum(len(samples) for samples in index[t_cluster][i_cluster].values())
            print(f"  ├─ I_cluster_{i_cluster} (样本数: {i_total})")
            
            for f_cluster in sorted(index[t_cluster][i_cluster].keys()):
                samples = index[t_cluster][i_cluster][f_cluster]
                print(f"  │  └─ F_cluster_{f_cluster}: 样本数 {len(samples)}, 索引: {samples}")
    
    end = time.perf_counter()



def cluster_and_insert_to_milvus():

    start = time.perf_counter()
    
    hierarchical_index, T_labels, I_labels, F_labels, T_reps, I_reps, F_reps = build_hierarchical_index()

    from mil_storage import MatrixStorage, InvertedIndexManager
    
    storage = MatrixStorage(db_path="./msu_milvus_lite")
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite")

    F = FEATURES["F"]
    I = FEATURES["I"]
    T = FEATURES["T_raw"]

    # 5. 验证数据一致性（补充多维度校验）
    assert len(F) == len(I) == len(T) == len(T_labels), \
        "矩阵数据与聚类标签数量不一致！"
    assert len(F) == FEATURES["sample_count"], "数据加载后样本数量变化！"

    # 6. 批量插入Milvus（统一进度打印）
    insert_start = time.perf_counter()
    total_samples = len(F)
    for i in range(total_samples):
        storage.insert_matrix(
            f_matrix=F[i],
            i_matrix=I[i],
            t_matrix=T[i],
            t_label=T_labels[i],
            i_label=I_labels[i],
            f_label=F_labels[i],
            index_manager=index_manager
        )
        # 每插入10个样本打印一次进度（便于大样本量调试）
        if (i + 1) % 10 == 0 or i + 1 == total_samples:
            print(f"[进度] 已插入 {i + 1}/{total_samples} 个样本")
    
    insert_end = time.perf_counter()
    print(f" {insert_end - insert_start:.4f}s")

    storage.close()
    index_manager.close()
    
    end = time.perf_counter()



if __name__ == "__main__":
    total_start = time.perf_counter()
    
    start_time = time.perf_counter()
    hierarchical_index, T_labels, I_labels, F_labels, T_reps, I_reps, F_reps = build_hierarchical_index()
    end_time = time.perf_counter()

    
    print_index_statistics(hierarchical_index, T_reps, I_reps, F_reps)
    
    