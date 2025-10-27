import json
import os
import numpy as np
from collections import defaultdict
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from msu import build_msu  
from T_cluster import pad_matrix, compute_ctmd
from I_cluster import count_matrix, compute_manhattan
from F_cluster import pm_tsmd
import time


class HierarchicalClusterManager:
    def __init__(self):
        self.FEATURES = self._load_features()
        self.hierarchical_index = None  # T→I→F三级索引结构
        self.T_labels = None  # 每个MSU的T簇标签
        self.I_labels = None  # 每个MSU的I簇标签
        self.F_labels = None  # 每个MSU的F簇标签
        self.T_reps = None    # T簇代表中心
        self.I_reps = None    # I簇代表中心
        self.F_reps = None    # F簇代表中心
        self._build_cluster_index()  # 初始化时构建聚类索引

    def _defaultdict_to_dict(self, d):
        if isinstance(d, defaultdict):
            d = dict(d)  # 先将当前层defaultdict转为dict
        if isinstance(d, dict):
            # 递归处理每一层的值
            for k, v in d.items():
                d[k] = self._defaultdict_to_dict(v)
        return d
    
    def _load_features(self):
        F, I, T = build_msu()
        
        # 过滤空样本（F/I/T矩阵均为空的样本）
        valid_indices = []
        for i in range(len(F)):
            # 检查F矩阵有效性：行是列表且非空
            f_valid = isinstance(F[i], list) and len(F[i]) > 0 and any(isinstance(row, list) and len(row) > 0 for row in F[i])
            # 检查I矩阵有效性：行是列表且非空
            i_valid = isinstance(I[i], list) and len(I[i]) > 0 and any(isinstance(row, list) and len(row) > 0 for row in I[i])
            # 检查T矩阵有效性：行是列表且非空
            t_valid = isinstance(T[i], list) and len(T[i]) > 0 and any(isinstance(row, list) and len(row) > 0 for row in T[i])
            
            if f_valid and i_valid and t_valid:
                valid_indices.append(i)
        
        # 保留有效样本
        F = [F[i] for i in valid_indices]
        I = [I[i] for i in valid_indices]
        T = [T[i] for i in valid_indices]
        
        # 处理T矩阵
        T_raw = [np.array(t) for t in T]
        max_rows = max([t.shape[0] for t in T_raw]) if T_raw else 0
        T_padded = [pad_matrix(t, max_rows) for t in T_raw]
        T_flattened = np.array([t.flatten() for t in T_padded])
        
        return {
            "F": [f.tolist() for f in [np.array(f) for f in F]],
            "I": [i.tolist() for i in [np.array(i) for i in I]],
            "T_raw": [t.tolist() for t in T_raw],
            "T_flattened": T_flattened,
            "T_max_rows": max_rows,
            "sample_count": len(F)
        }

    def _get_T_labels(self):
        def ctmd_metric(X, Y=None):
            if Y is None:
                return pairwise_distances(
                    X, 
                    metric=lambda a, b: compute_ctmd(a, b, self.FEATURES["T_max_rows"])
                )
            return np.array([compute_ctmd(x, y, self.FEATURES["T_max_rows"]) for x, y in zip(X, Y)])
        
        distance_matrix = ctmd_metric(self.FEATURES["T_flattened"])
        clustering = AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        representatives = self._get_cluster_representatives(labels, distance_matrix)
        return labels, representatives

    def _get_I_labels(self):
        I_features = [np.array(i) for i in self.FEATURES["I"]]
        n_samples = self.FEATURES["sample_count"]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = self._matrix_analysis(I_features[i], I_features[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        clustering = AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        representatives = self._get_cluster_representatives(labels, distance_matrix)
        return labels, representatives

    def _get_F_labels(self):
        F_features = [np.array(f) for f in self.FEATURES["F"]]
        n_samples = self.FEATURES["sample_count"]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = pm_tsmd(F_features[i], F_features[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        clustering = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        representatives = self._get_cluster_representatives(labels, distance_matrix)
        return labels, representatives

    def _matrix_analysis(self, IA, IB):
        V1 = count_matrix(IA)
        V2 = count_matrix(IB)
        sum1 = sum(V1) or 1
        sum2 = sum(V2) or 1
        return compute_manhattan(V1 / sum1, V2 / sum2)

    def _get_cluster_representatives(self, labels, distance_matrix):
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
        return representatives

    def _build_cluster_index(self):
        self.T_labels, self.T_reps = self._get_T_labels()
        self.I_labels, self.I_reps = self._get_I_labels()
        self.F_labels, self.F_reps = self._get_F_labels()
        n_samples = self.FEATURES["sample_count"]
        
        # 验证数据一致性
        assert len(self.T_labels) == len(self.I_labels) == len(self.F_labels) == n_samples, \
            "聚类标签数量与样本数量不一致！"
        
        # 构建三级索引
        temp_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for msu_id in range(n_samples):
            t_cluster = self.T_labels[msu_id]
            i_cluster = self.I_labels[msu_id]
            f_cluster = self.F_labels[msu_id]
            temp_index[t_cluster][i_cluster][f_cluster].append(msu_id)
        
        self.hierarchical_index = self._defaultdict_to_dict(temp_index)

    def get_cluster_labels(self, msu_id):
        if 0 <= msu_id < self.FEATURES["sample_count"]:
            return {
                "T_label": self.T_labels[msu_id],
                "I_label": self.I_labels[msu_id],
                "F_label": self.F_labels[msu_id]
            }
        return None

    def query_by_cluster(self, t_cluster=None, i_cluster=None, f_cluster=None):
        result = []
        t_level = self.hierarchical_index
        
        # 过滤T簇
        if t_cluster is not None:
            if t_cluster not in t_level:
                return result
            t_level = {t_cluster: t_level[t_cluster]}
        
        # 过滤I簇
        for t_c, i_level in t_level.items():
            if i_cluster is not None:
                if i_cluster not in i_level:
                    continue
                i_level = {i_cluster: i_level[i_cluster]}
            
            # 过滤F簇并收集结果
            for i_c, f_level in i_level.items():
                if f_cluster is not None:
                    if f_cluster not in f_level:
                        continue
                    f_level = {f_cluster: f_level[f_cluster]}
                
                for f_c, msu_ids in f_level.items():
                    result.extend(msu_ids)
        
        return list(set(result))

    def get_cluster_representatives(self):
        return {
            "T_reps": self.T_reps,
            "I_reps": self.I_reps,
            "F_reps": self.F_reps
        }

    def print_cluster_statistics(self):
        reps = self.get_cluster_representatives()
        print("1. 簇代表中心：")
        for type_name, reps_dict in reps.items():
            print(f"   {type_name}: {dict(reps_dict)}")
        
        for t_c in sorted(self.hierarchical_index.keys()):
            t_level = self.hierarchical_index[t_c]
            t_total = 0
            for i_c in t_level.keys():
                i_level = t_level[i_c]
                for f_c in i_level.keys():
                    t_total += len(i_level[f_c])
            
            print(f"\n   T簇-{t_c}（总样本：{t_total}）:")
            
            for i_c in sorted(t_level.keys()):
                i_level = t_level[i_c]
                i_total = sum(len(msu_ids) for msu_ids in i_level.values())
                print(f"     ├─ I簇-{i_c}（样本：{i_total}）:")
                
                for f_c in sorted(i_level.keys()):
                    msu_ids = i_level[f_c]
                    print(f"     │  └─ F簇-{f_c}：{len(msu_ids)}个样本，MSU_ID: {msu_ids}")

class InvertedIndexManager:
    def __init__(self, db_path="./msu_milvus_lite_db", collection_name="matrix_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = None
        self.index = {}  # 格式: {矩阵类型:值: [MSU_ID列表]}
        self.index_file = os.path.join(os.path.dirname(db_path), "inverted_index.json")
        self.connect_milvus()
        self.load_index()
        self._build_time = 0.0          # 累计构建耗时
        self._build_count = 0           # 调了多少次update

    def connect_milvus(self):
        if "default" in connections.list_connections():
            current_uri = connections.get_connection_addr("default")["uri"]
            if current_uri == f"{self.db_path}.db":
                self.collection = Collection(self.collection_name)
                return
            connections.disconnect("default")
        connections.connect("default", uri=f"{self.db_path}.db")
        self.collection = Collection(self.collection_name)

    def update_index(self, msu_id, f_matrix, i_matrix):
        tic = time.perf_counter()
        # ===== 原逻辑 =====
        for row in f_matrix:
            if row:
                self._add_to_index(f"F:{row[0]}", msu_id)
        for row in i_matrix:
            for val in row:
                self._add_to_index(f"I:{val}", msu_id)
        # ==================
        toc = time.perf_counter()
        self._build_time += toc - tic
        self._build_count += 1
        self.save_index()      # 想排除I/O可把这句挪到close()里统一刷盘

    def print_build_time(self):
        print(f"[倒排索引] 共更新{self._build_count}次，累计构建时间{self._build_time:.4f}s")

    def _add_to_index(self, key, msu_id):
        if key not in self.index:
            self.index[key] = []
        if msu_id not in self.index[key]:
            self.index[key].append(msu_id)

    def query_index(self, matrix_type, value):
        return self.index.get(f"{matrix_type}:{value}", [])

    def save_index(self):
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            print(f"[倒排索引] 保存失败: {e}")

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                print(f"[倒排索引] 加载失败: {e}")

    def close(self):
        self.save_index()
        if "default" in connections.list_connections():
            connections.disconnect("default")

# -------------------------- 矩阵存储管理器 --------------------------
class MatrixStorage:
    def __init__(self, db_path="./msu_milvus_lite_db", collection_name="matrix_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = None
        self.vector_dim = 128  
        self.current_id = 0
        self._ensure_directory_exists()
        self.connect_milvus()
        self.create_collection()
        self._init_current_id()

    def _ensure_directory_exists(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def connect_milvus(self):
        if "default" in connections.list_connections():
            current_uri = connections.get_connection_addr("default")["uri"]
            if current_uri == f"{self.db_path}.db":
                return
            connections.disconnect("default")
        connections.connect("default", uri=f"{self.db_path}.db")

    def create_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="placeholder_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="F_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="I_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="T_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="T_label", dtype=DataType.INT64),
            FieldSchema(name="I_label", dtype=DataType.INT64),
            FieldSchema(name="F_label", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="存储MSU数据及聚类标签")
        self.collection = Collection(self.collection_name, schema)
        self.collection.create_index("placeholder_vector", {"index_type": "FLAT", "params": {}, "metric_type": "L2"})
        self.collection.load()

    def _init_current_id(self):
        if self.collection.num_entities > 0:
            result = self.collection.query("id >= 0", output_fields=["id"], limit=1, sort_by="id", sort_type="desc")
            self.current_id = result[0]["id"] + 1 if result else 0
        else:
            self.current_id = 0

    def insert_matrix(self, f_matrix, i_matrix, t_matrix, t_label, i_label, f_label, index_manager=None):
        if not self._is_2d_matrix(f_matrix) or not self._is_2d_matrix(i_matrix) or not self._is_2d_matrix(t_matrix):
            raise ValueError("矩阵必须是二维列表")

        msu_id = self.current_id
        self.current_id += 1

        placeholder_vector = [0.0] * self.vector_dim
        data = [
            [msu_id],
            [placeholder_vector],
            [json.dumps(f_matrix)],
            [json.dumps(i_matrix)],
            [json.dumps(t_matrix)],
            [t_label],
            [i_label],
            [f_label]
        ]

        self.collection.insert(data)

        if index_manager:
            index_manager.update_index(msu_id, f_matrix, i_matrix)

        return msu_id

    def _is_2d_matrix(self, matrix):
        return isinstance(matrix, list) and all(isinstance(row, list) for row in matrix)

    def close(self):
        if self.collection:
            self.collection.release()
        if "default" in connections.list_connections():
            connections.disconnect("default")
        print("[存储] 已关闭Milvus连接")


if __name__ == "__main__":
    cluster_manager = HierarchicalClusterManager()
    cluster_manager.print_cluster_statistics()

    storage = MatrixStorage(db_path="./msu_milvus_lite4")
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite4")
    features = cluster_manager.FEATURES
    for i in range(features["sample_count"]):
        labels = cluster_manager.get_cluster_labels(i)
        storage.insert_matrix(
            f_matrix=features["F"][i],
            i_matrix=features["I"][i],
            t_matrix=features["T_raw"][i],
            t_label=labels["T_label"],
            i_label=labels["I_label"],
            f_label=labels["F_label"],
            index_manager=index_manager
        )
    storage.close()
    index_manager.close()
    index_manager.print_build_time()  