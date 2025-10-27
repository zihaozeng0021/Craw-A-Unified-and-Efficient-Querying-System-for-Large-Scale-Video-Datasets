import json
import numpy as np
from pymilvus import connections, Collection, utility
from mil_storage import InvertedIndexManager, HierarchicalClusterManager
import time
from sklearn.metrics import pairwise_distances
from T_cluster import pad_matrix, compute_ctmd
from I_cluster import count_matrix, compute_manhattan
from F_cluster import pm_tsmd
import gc 
import math
class MSUQueryManager:
    def __init__(self, db_path="./msu_milvus_lite1_db", collection_name="matrix_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = None
        self.cluster_manager = HierarchicalClusterManager()  
        self.connect_milvus()
        self.load_collection()
        
    def connect_milvus(self):
        if "default" in connections.list_connections():
            current_uri = connections.get_connection_addr("default")["uri"]
            if current_uri == f"{self.db_path}.db":
                return
            else:
                connections.disconnect("default")
        
        connections.connect(
            "default",
            uri=f"{self.db_path}.db"
        )
    
    def load_collection(self):
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"集合 {self.collection_name} 不存在")
            
        self.collection = Collection(self.collection_name)
        self.collection.load()
    
    def get_by_id(self, msu_id):
        if not self.collection:
            self.load_collection()
            
        result = self.collection.query(
            expr=f"id == {msu_id}",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix", 
                          "T_label", "I_label", "F_label"]
        )
        
        if result:
            item = result[0]
            return {
                "id": item["id"],
                "F_matrix": json.loads(item["F_matrix"]),
                "I_matrix": json.loads(item["I_matrix"]),
                "T_matrix": json.loads(item["T_matrix"]),
                "T_label": item["T_label"],
                "I_label": item["I_label"],
                "F_label": item["F_label"]
            }
        return None
    
    def get_full_data_by_id(self, msu_id):
        return self.get_by_id(msu_id)
    
    def get_by_label(self, label_type, label_value):
        if label_type not in ["T", "I", "F"]:
            raise ValueError("标签类型必须是 'T', 'I' 或 'F'")
            
        if not self.collection:
            self.load_collection()
            
        label_field = f"{label_type}_label"
        result = self.collection.query(
            expr=f"{label_field} == {label_value}",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix", 
                          "T_label", "I_label", "F_label"]
        )
        
        return [
            {
                "id": item["id"],
                "F_matrix": json.loads(item["F_matrix"]),
                "I_matrix": json.loads(item["I_matrix"]),
                "T_matrix": json.loads(item["T_matrix"]),
                "T_label": item["T_label"],
                "I_label": item["I_label"],
                "F_label": item["F_label"]
            } for item in result
        ]
    
    def get_by_label_range(self, label_type, min_value, max_value):
        """通过标签范围查询数据"""
        if label_type not in ["T", "I", "F"]:
            raise ValueError("标签类型必须是 'T', 'I' 或 'F'")
            
        if not self.collection:
            self.load_collection()
            
        label_field = f"{label_type}_label"
        result = self.collection.query(
            expr=f"{label_field} >= {min_value} and {label_field} <= {max_value}",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix", 
                          "T_label", "I_label", "F_label"]
        )
        
        return [
            {
                "id": item["id"],
                "F_matrix": json.loads(item["F_matrix"]),
                "I_matrix": json.loads(item["I_matrix"]),
                "T_matrix": json.loads(item["T_matrix"]),
                "T_label": item["T_label"],
                "I_label": item["I_label"],
                "F_label": item["F_label"]
            } for item in result
        ]
    
    def search_by_element(self, index_manager, matrix_type, value):
        """通过矩阵元素值查询数据（依赖倒排索引）"""
        if matrix_type not in ["F", "I"]:
            raise ValueError("矩阵类型必须是 'F' 或 'I'")
            
        # 从倒排索引获取包含该元素的MSU ID列表
        msu_ids = index_manager.query_index(matrix_type, value)
        if not msu_ids:
            return []
            
        # 收集结果（只包含MSU数据，不再返回行列位置）
        results = []
        for msu_id in msu_ids:
            msu_data = self.get_by_id(msu_id)
            if msu_data:
                results.append({
                    "msu_data": msu_data,
                    "element_info": {
                        "matrix_type": matrix_type,
                        "value": value
                    }
                })
        
        return results
    
    def get_all(self, limit=100):
        if not self.collection:
            self.load_collection()
            
        result = self.collection.query(
            expr="id >= 0",  # 匹配所有记录
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix", 
                          "T_label", "I_label", "F_label"],
            limit=limit
        )
        
        return [
            {
                "id": item["id"],
                "F_matrix": json.loads(item["F_matrix"]),
                "I_matrix": json.loads(item["I_matrix"]),
                "T_matrix": json.loads(item["T_matrix"]),
                "T_label": item["T_label"],
                "I_label": item["I_label"],
                "F_label": item["F_label"]
            } for item in result
        ]
    def find_similar_by_msu_data(self, input_f_matrix, input_i_matrix, input_t_matrix, top_k=5):
        """按输入的 F/I/T 矩阵，分层找 T→I→F 最相似的 top-k 个样本"""
        # 1. 格式检查
        if not (self._is_2d_matrix(input_f_matrix) and
                self._is_2d_matrix(input_i_matrix) and
                self._is_2d_matrix(input_t_matrix)):
            raise ValueError("F/I/T 矩阵必须是二维列表，如 [[0,1],[1,2]]")

        start_time = time.perf_counter()
        input_f = np.array(input_f_matrix)
        input_i = np.array(input_i_matrix)
        input_t_flat = self._prepare_t_matrix(input_t_matrix)
        t_max_rows = self.cluster_manager.FEATURES["T_max_rows"]

        # ---------- 1. 找最近 T 簇 ----------
        t_reps = self.cluster_manager.get_cluster_representatives()["T_reps"]
        t_dist_map = {}
        for t_cid, t_rid in t_reps.items():
            msu = self.get_by_id(t_rid)
            if not msu:
                continue
            rep_flat = self._prepare_t_matrix(msu["T_matrix"])
            t_dist_map[t_cid] = compute_ctmd(
                input_t_flat.reshape(t_max_rows, -1),
                rep_flat.reshape(t_max_rows, -1),
                t_max_rows
            )
        if not t_dist_map:
            print("无有效 T 簇代表，返回空结果")
            return []
        target_t_cluster = min(t_dist_map, key=t_dist_map.get)

        # ---------- 2. 在目标 T 簇内找最近 I 簇 ----------
        t_msus = self.cluster_manager.query_by_cluster(t_cluster=target_t_cluster)
        if not t_msus:
            print("目标 T 簇内无样本，返回空结果")
            return []
        i_reps = self.cluster_manager.get_cluster_representatives()["I_reps"]
        valid_i_reps = {cid: rid for cid, rid in i_reps.items() if rid in t_msus}
        if not valid_i_reps:
            print("目标 T 簇内无有效 I 簇代表，返回空结果")
            return []

        i_dist_map = {}
        for i_cid, i_rid in valid_i_reps.items():
            msu = self.get_by_id(i_rid)
            if not msu:
                continue
            rep_i = np.array(msu["I_matrix"])
            i_dist_map[i_cid] = compute_manhattan(
                count_matrix(input_i) / (sum(count_matrix(input_i)) or 1),
                count_matrix(rep_i) / (sum(count_matrix(rep_i)) or 1)
            )
        target_i_cluster = min(i_dist_map, key=i_dist_map.get)

        # ---------- 3. 在目标 T+I 簇内重新选“局部 F 代表” ----------
        ti_msus = self.cluster_manager.query_by_cluster(
            t_cluster=target_t_cluster,
            i_cluster=target_i_cluster
        )
        if not ti_msus:
            print("目标（T+I）簇内无样本，返回空结果")
            return []

        # 加载 T+I 簇内所有 F 矩阵并计算距离
        f_dist_map = {}
        for msu_id in ti_msus:
            msu = self.get_by_id(msu_id)
            if not msu:
                continue
            f_mat = np.array(msu["F_matrix"])
            f_dist_map[msu_id] = pm_tsmd(input_f, f_mat)

        if not f_dist_map:
            print("T+I 簇内无法计算任何 F 距离，返回空结果")
            return []

        # 按 F 距离升序取 top-k（不再细分 F 簇）
        sorted_msu = sorted(f_dist_map.items(), key=lambda x: x[1])[:top_k]
        top_k_result = []
        for msu_id, f_dist in sorted_msu:
            msu_data = self.get_by_id(msu_id)
            if msu_data:
                top_k_result.append({
                    "msu_id": msu_id,
                    "msu_data": msu_data,
                    "f_distance": f_dist
                })

        end_time = time.perf_counter()
        print(f"\n查询完成！耗时 {(end_time - start_time) * 1000:.2f} 毫秒")
        return top_k_result
        
    def count_by_label(self, label_type):
        """统计不同标签值的数量分布"""
        if label_type not in ["T", "I", "F"]:
            raise ValueError("标签类型必须是 'T', 'I' 或 'F'")
            
        if not self.collection:
            self.load_collection()
            
        label_field = f"{label_type}_label"
        result = self.collection.query(
            expr="id >= 0",
            output_fields=[label_field, "id"],
        )
        
        # 统计数量
        count_dict = {}
        for item in result:
            label = item[label_field]
            count_dict[label] = count_dict.get(label, 0) + 1
            
        return count_dict
    
    

    def _is_2d_matrix(self, matrix):
        """验证是否为二维列表（如[[0,1],[1,2]]）"""
        if not isinstance(matrix, list):
            return False
        if len(matrix) == 0:
            return False
        # 每行必须是列表
        return all(isinstance(row, list) for row in matrix)
    
    # -------------------------- 新增：聚类索引相关查询方法 --------------------------
    def find_similar_by_cluster(self, input_msu_id, top_k=5, include_self=False):
        # 1. 获取输入MSU数据（先验证输入ID存在）
        input_msu = self.get_by_id(input_msu_id)
        if not input_msu:
            raise ValueError(f"输入的MSU ID {input_msu_id} 不存在")
        start_time = time.perf_counter()

        # 2. 步骤1：找距离最小的T簇（与所有T簇代表对比）
        t_reps = self.cluster_manager.get_cluster_representatives()["T_reps"]
        t_cluster_dist = {}
        input_t_flat = self._prepare_t_matrix(input_msu["T_matrix"])
        t_max_rows = self.cluster_manager.FEATURES["T_max_rows"]

        for t_cluster_id, t_rep_id in t_reps.items():
            t_rep_msu = self.get_by_id(t_rep_id)
            if not t_rep_msu:
                continue
            rep_t_flat = self._prepare_t_matrix(t_rep_msu["T_matrix"])
            t_dist = compute_ctmd(
                input_t_flat.reshape(t_max_rows, -1),
                rep_t_flat.reshape(t_max_rows, -1),
                t_max_rows
            )
            t_cluster_dist[t_cluster_id] = t_dist

        min_t_dist = min(t_cluster_dist.values()) if t_cluster_dist else float("inf")
        min_t_clusters = [cid for cid, dist in t_cluster_dist.items() if dist == min_t_dist]
        target_t_cluster = min_t_clusters[0]

        # 3. 步骤2：在目标T簇内，找距离最小的I簇
        t_cluster_msu_ids = self.cluster_manager.query_by_cluster(t_cluster=target_t_cluster)
        if not t_cluster_msu_ids:
            print("目标T簇内无样本，返回空结果")
            return []

        i_reps = self.cluster_manager.get_cluster_representatives()["I_reps"]
        valid_i_reps = {cid: rid for cid, rid in i_reps.items() if rid in t_cluster_msu_ids}
        if not valid_i_reps:
            print("目标T簇内无有效I簇代表，返回空结果")
            return []

        i_cluster_dist = {}
        input_i = np.array(input_msu["I_matrix"])

        for i_cluster_id, i_rep_id in valid_i_reps.items():
            i_rep_msu = self.get_by_id(i_rep_id)
            if not i_rep_msu:
                continue
            rep_i = np.array(i_rep_msu["I_matrix"])
            i_dist = compute_manhattan(
                count_matrix(input_i) / (sum(count_matrix(input_i)) or 1),
                count_matrix(rep_i) / (sum(count_matrix(rep_i)) or 1)
            )
            i_cluster_dist[i_cluster_id] = i_dist

        min_i_dist = min(i_cluster_dist.values()) if i_cluster_dist else float("inf")
        min_i_clusters = [cid for cid, dist in i_cluster_dist.items() if dist == min_i_dist]
        target_i_cluster = min_i_clusters[0]

        # 4. 步骤3：在目标（T+I）簇内，找距离最小的F簇（修复变量名错误）
        ti_cluster_msu_ids = self.cluster_manager.query_by_cluster(
            t_cluster=target_t_cluster,
            i_cluster=target_i_cluster
        )
        if not ti_cluster_msu_ids:
            print("目标（T+I）簇内无样本，返回空结果")
            return []

        f_reps = self.cluster_manager.get_cluster_representatives()["F_reps"]
        valid_f_reps = {cid: rid for cid, rid in f_reps.items() if rid in ti_cluster_msu_ids}
        if not valid_f_reps:
            print("目标（T+I）簇内无有效F簇代表，返回空结果")
            return []

        f_cluster_dist = {}
        input_f = np.array(input_msu["F_matrix"])

        for f_cluster_id, f_rep_id in valid_f_reps.items():
            f_rep_msu = self.get_by_id(f_rep_id)
            if not f_rep_msu:
                continue
            rep_f = np.array(f_rep_msu["F_matrix"])
            f_dist = pm_tsmd(input_f, rep_f)
            f_cluster_dist[f_cluster_id] = f_dist

        # -------------------------- 修复此处的变量名错误 --------------------------
        min_f_dist = min(f_cluster_dist.values()) if f_cluster_dist else float("inf")  # 先计算最小距离值
        min_f_clusters = [cid for cid, dist in f_cluster_dist.items() if dist == min_f_dist]  # 用最小距离值筛选簇
        # --------------------------------------------------------------------------
        target_f_cluster = min_f_clusters[0]

        # 5. 步骤4：在最终目标（T+I+F）簇内，按F距离升序取top-k
        target_cluster_ids = self.cluster_manager.query_by_cluster(
            t_cluster=target_t_cluster,
            i_cluster=target_i_cluster,
            f_cluster=target_f_cluster
        )
        if not target_cluster_ids:
            return []

        # 排除输入自身（可控制）
        if not include_self:
            target_cluster_ids = [cid for cid in target_cluster_ids if cid != input_msu_id]
        if not target_cluster_ids:
            print("排除输入后，目标簇内无样本，返回空结果")
            return []

        # 计算目标簇内所有样本的F距离
        f_dist_list = []
        for msu_id in target_cluster_ids:
            msu_data = self.get_by_id(msu_id)
            if not msu_data:
                continue
            msu_f = np.array(msu_data["F_matrix"])
            f_dist = pm_tsmd(input_f, msu_f)
            f_dist_list.append({
                "msu_id": msu_id,
                "msu_data": msu_data,
                "f_distance": f_dist
            })

        # 按F距离升序排序（值越小排名越高）
        f_dist_list.sort(key=lambda x: x["f_distance"])
        top_k_result = f_dist_list[:top_k]

        # 输出结果
        end_time = time.perf_counter()
        print(f"\n查询完成！耗时 {(end_time - start_time) * 1000:.2f} 毫秒")
        print(f"最终目标簇内共 {len(target_cluster_ids)} 个样本，返回top-{top_k}（F距离升序）：")

        return top_k_result

    def _prepare_t_matrix(self, t_matrix):
        t_np = np.array(t_matrix)
        max_rows = self.cluster_manager.FEATURES["T_max_rows"]
        if t_np.size == 0:
            return np.zeros((max_rows, 2)).flatten()
        if t_np.shape[0] < max_rows:
            padding = np.zeros((max_rows - t_np.shape[0], t_np.shape[1]))
            t_padded = np.vstack([t_np, padding])
        else:
            t_padded = t_np[:max_rows]
        return t_padded.flatten()

    def _matrix_analysis(self, IA, IB):
        from I_cluster import count_matrix, compute_manhattan
        V1 = count_matrix(IA)
        V2 = count_matrix(IB)
        sum1 = sum(V1) or 1
        sum2 = sum(V2) or 1
        return compute_manhattan(V1 / sum1, V2 / sum2)
    
    def _prepare_t_matrix(self, t_matrix):
        t_np = np.array(t_matrix)
        max_rows = self.cluster_manager.FEATURES["T_max_rows"]
        if t_np.size == 0:
            return np.zeros((max_rows, 2)).flatten()  
        if t_np.shape[0] < max_rows:
            padding = np.zeros((max_rows - t_np.shape[0], t_np.shape[1]))
            t_padded = np.vstack([t_np, padding])
        else:
            t_padded = t_np[:max_rows]
        return t_padded.flatten()
    
    def close(self):
        """关闭连接"""
        if self.collection:
            self.collection.release()
        if "default" in connections.list_connections():
            connections.disconnect("default")
        #print("[查询] 已关闭Milvus连接")


if __name__ == "__main__":

    #DQ1.1
    '''
    query_manager = MSUQueryManager(db_path="./msu_milvus_lite")
    time1 = time.perf_counter()
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite")

    THRESHOLD_SQUARED = 400 * 400  
    results = index_manager.query_index(matrix_type="F", value=0)
    ids_str = ", ".join(map(str, results)) if results else ""

    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]", 
        output_fields=["id", "F_matrix"]
    )

    id_results = []         
    doubtful_ids = []    

    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        found_valid = False  
        found_doubtful = False  
        
        for row in f_matrix:
            x1, y1 = row[1], row[2]
            x2, y2 = row[-2], row[-1]

            dist12_sq = (x2 - x1) **2 + (y2 - y1)** 2
            if dist12_sq <= THRESHOLD_SQUARED:
                continue

            length = len(row) - 1
            mid_idx = length // 2
            if mid_idx % 2 == 1:
                x3, y3 = row[mid_idx], row[mid_idx + 1]
            else:
                x3, y3 = row[mid_idx - 1], row[mid_idx]
            dist13_sq = (x3 - x1) **2 + (y3 - y1)** 2
            dist23_sq = (x2 - x3) **2 + (y2 - y3)** 2
            if dist13_sq > THRESHOLD_SQUARED or dist23_sq > THRESHOLD_SQUARED:
                id_results.append(item['id'])
                found_valid = True
                break 
            else:
                found_doubtful = True
        if not found_valid and found_doubtful:
            doubtful_ids.append(item['id'])

    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000

    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    print("正确的ID列表:", id_results)
    print("存疑的ID列表:", doubtful_ids)
    '''

    #DQ1.2
    '''
    query_manager = MSUQueryManager(db_path="./msu_milvus_lite")
    time1 = time.perf_counter()
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite")
    id_result = []
    results = index_manager.query_index(matrix_type="F", value=0)
    ids_str = ", ".join(map(str, results)) if results else ""
    x = 640 
    y = 160
    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]", 
        output_fields=["id", "F_matrix"]
    )
    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        has_valid_row = False  
        for row in f_matrix:
            distance = 0
            valid = True  
            for i in range(1, len(row)-1, 2):
                x1, y1 = row[i], row[i+1]
                if i == 1:
                    distance = (x1 - x)**2 + (y1 - y)** 2
                else:
                    distance1 = (x1 - x)**2 + (y1 - y)** 2
                    if distance1 > distance:
                        distance = distance1
                    else:
                        valid = False  
                        break
            if valid:  
                has_valid_row = True
                break
        if has_valid_row:  
            id_result.append(item['id'])
    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000

    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    print(id_result)
    '''

    #DQ3.1
    '''
    query_manager = MSUQueryManager(db_path="./msu_milvus_lite3")
    time1 = time.perf_counter()
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite3")
    id_result = []
    results = index_manager.query_index(matrix_type="F", value=0)
    ids_str = ", ".join(map(str, results)) if results else ""
    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]", 
        output_fields=["id", "F_matrix"]
    )
    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        for row in f_matrix:
            x = row[1]
            y = row[2]
            x1 = row[-2]
            y1 = row[-1]
            if 440 < x < 560 and 0 < y < 60:
                if 500 < x1 < 1000 and 150 < y1 < 460:
                    id_result.append(item['id'])
                    break
            elif 0 < x < 450 and 250 < y < 460:
                if 950 < x1 < 1080 and 0 < y1 < 160:
                    id_result.append(item['id'])
                    break
            elif 1000 < x < 1280 and 300 < y < 460:
                if 560 < x1 < 800 and 0 < y1 < 200:
                    id_result.append(item['id'])
                    break
            elif 800 < x < 980 and 0 < y < 100:
                if 0 < x1 < 450 and 130 < y1 < 320:
                    id_result.append(item['id'])
                    break

    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000

    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    print(id_result)
    '''


    #DQ3.2
    '''
    query_manager = MSUQueryManager(db_path="./msu_milvus_lite3")
    time1 = time.perf_counter()
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite3")
    id_result = []
    results = index_manager.query_index(matrix_type="F", value=1.0)
    ids_str = ", ".join(map(str, results)) 
    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]", 
        output_fields=["id", "F_matrix"]
    )
    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        for row in f_matrix:
            count = 0
            if row[0] == 1:
                for j in range(1,len(row)-1,2):
                    x1 = row[j]
                    y1 = row[j+1]
                    if 400 < x1 < 880 and 0 < y1 < 80:
                        count +=1
                    else:
                        continue
        if count >= 1:
            id_result.append(item['id'])
    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000

    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    print(id_result)
    '''

    
    query_manager = MSUQueryManager(db_path="./msu_milvus_lite2")
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite2")
    msu_id = 30
    print(f"\n查询ID为{msu_id}的MSU完整数据")
    full_data = query_manager.get_full_data_by_id(msu_id)
    if full_data:
        print(f"找到MSU {msu_id}的完整数据")
        print(f"ID: {full_data['id']}")
        print(f"F矩阵: {full_data['F_matrix']}")
        print(f"I矩阵: {full_data['I_matrix']}")
        print(f"T矩阵: {full_data['T_matrix']}\n")
    

    '''
    # 通过倒排索引查询MSU_ID
    time1 = time.perf_counter()
    results = index_manager.query_index(matrix_type="F", value=1.0)
    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000000
    print(results,len(results))
    print(f'查询耗时: {duration_ms:.2f} 微秒')  
    '''

    '''
    results = index_manager.query_index(matrix_type="F", value=0.0)
    result2 = index_manager.query_index(matrix_type="F", value=1.0)
    query_result = []
    time1= time.perf_counter()
    ids_str = ", ".join(map(str, results))
    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]",
        output_fields=["id", "F_matrix"]  
    )
    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        count = 0
        count = sum(1 for row in f_matrix if row and row[0] == 0)
        for row in f_matrix:
            if row and row[0] == 0:
                count += 1
                if count >= 4:
                    break
        if count >= 4:
            query_result.append(item["id"])
    result3 = [id for id in query_result if id in result2]
    time2 = time.perf_counter()
    print(result3)
    duration_ms = (time2 - time1) * 1000
    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    '''

    '''
    import gc 
    
    all_query_results = []
    all_time_results = []
    total_runs = 200 
    interval_seconds = 1  

    for run in range(total_runs):

        gc.collect()
        time.sleep(interval_seconds)

        query_manager = MSUQueryManager(db_path="./msu_milvus_lite2")
        index_manager = InvertedIndexManager(db_path="./msu_milvus_lite2")

        query_result = []
        time1 = time.perf_counter()
        
        try:
            results = index_manager.query_index(matrix_type="F", value=3.0)
            ids_str = ", ".join(map(str, results))
            
            batch_result = query_manager.collection.query(
                expr=f"id in [{ids_str}]",
                output_fields=["id", "F_matrix"]
            )

            for item in batch_result:
                i_matrix = json.loads(item["F_matrix"])
                count = 0
                found_enough = False
                
                for row in i_matrix:
                    if found_enough:
                        break  
                        
                    for value in row:
                        if value == 0:
                            count += 1
                            if count >= 4:
                                found_enough = True
                                break
                
                if count >= 4:
                    query_result.append(item["id"])

        finally:
            if hasattr(query_manager, 'close'):
                query_manager.close()
            if hasattr(index_manager, 'close'):
                index_manager.close()

        time2 = time.perf_counter()
        duration_ms = (time2 - time1) * 1000
        all_query_results.append(query_result)
        all_time_results.append(round(duration_ms, 2))
        
    print(f"平均耗时: {sum(all_time_results)/total_runs:.2f} 毫秒")
    print(f"最快耗时: {min(all_time_results):.2f} 毫秒")
    print(f"最慢耗时: {max(all_time_results):.2f} 毫秒")
    print(all_time_results)
    '''
    '''
    results = []
    time1 = time.perf_counter()
    result1 = index_manager.query_index(matrix_type="F", value=3.0)
    result2 = index_manager.query_index(matrix_type="F", value=4.0)

    result3 = [id for id in result1 if id in result2]
    ids_str = ", ".join(map(str, result3))
    batch_result = query_manager.collection.query(
        expr=f"id in [{ids_str}]",
        output_fields=["id", "F_matrix", "T_matrix"]
    )
    
    for item in batch_result:
        f_matrix = json.loads(item["F_matrix"])
        has_target = any(row and row[0] == 1 for row in f_matrix)
        
        if has_target:
            t_matrix = json.loads(item["T_matrix"])
            target_times = []
            for i, row in enumerate(f_matrix):
                if row and row[0] == 1 and i < len(t_matrix) and t_matrix[i]:
                    target_times.append(t_matrix[i][0])
            
            if target_times:
                min_t, max_t = min(target_times), max(target_times)
                in_range_exists = any(row and len(row) > 0 and min_t <= row[0] <= max_t for row in t_matrix)
                results.append(item["id"])

    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000
    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    print(results)
    '''
    '''
    all_query_results = []   
    all_time_results = []    
    total_runs = 10
    interval_seconds = 0.2

    for run in range(total_runs):
        gc.collect()
        time.sleep(interval_seconds)

        query_manager = MSUQueryManager(db_path="./msu_milvus_lite3")
        index_manager = InvertedIndexManager(db_path="./msu_milvus_lite3")

        time1 = time.perf_counter()
        hit_ids = []          
        try:
            
            ids_0 = index_manager.query_index(matrix_type="F", value=0)
            ids_1 = index_manager.query_index(matrix_type="F", value=2)

            
            intersect_ids = list(set(ids_0).intersection(ids_1))

            if intersect_ids:                 
                ids_str = ", ".join(map(str, intersect_ids))
                batch_result = query_manager.collection.query(
                    expr=f"id in [{ids_str}]",
                    output_fields=["id"]
                )
                hit_ids = [item["id"] for item in batch_result]

        finally:
            # 3. 关闭连接
            if hasattr(query_manager, 'close'):
                query_manager.close()
            if hasattr(index_manager, 'close'):
                index_manager.close()

        time2 = time.perf_counter()
        duration_ms = (time2 - time1) * 1000
        all_query_results.append(hit_ids)
        all_time_results.append(round(duration_ms, 2))

    print(f"平均耗时: {sum(all_time_results)/total_runs:.2f} 毫秒")
    print(f"最快耗时: {min(all_time_results):.2f} 毫秒")
    print(f"最慢耗时: {max(all_time_results):.2f} 毫秒")
    print(all_time_results)
    '''
    
    '''
    input_msu_id = 0  # 输入的MSU ID
    top_k = 3         # 返回最相似的5个结果
    include_self = True # 是否包含输入自身（True=包含，False=排除）
    print(f"\n查询与MSU ID {input_msu_id} 最相似的{top_k}个结果：")
    similar_results = query_manager.find_similar_by_cluster(
            input_msu_id=input_msu_id,
            top_k=top_k,
            include_self=include_self
        )
    if similar_results:
        for idx, res in enumerate(similar_results, 1):
            print(f"\n第{idx}名（F距离：{res['f_distance']:.4f}）：")
            print(f"MSU ID：{res['msu_id']}")
            print(f"T标签：{res['msu_data']['T_label']} | I标签：{res['msu_data']['I_label']} | F标签：{res['msu_data']['F_label']}")
    else:
        print(f"\n未找到符合条件的相似样本")
    '''


    query_manager = MSUQueryManager(db_path="./msu_milvus_lite2")
    time1 = time.perf_counter()
    index_manager = InvertedIndexManager(db_path="./msu_milvus_lite2")
    
    input_msu_data = {
        "F_matrix": [[3.0, 618.4, 16.07, 621.91, 28.95, 629.54, 42.77, 633.71, 50.51, 640.49, 54.8, 648.23, 61.19, 651.71, 62.32], [3.0, 55.38, 13.21, 55.38, 13.21, 55.38, 13.21, 55.38, 13.21, 55.38, 13.21, 55.38, 13.21, 55.38, 13.21], [3.0, 644.14, 25.53, 646.89, 27.34, 648.18, 29.39, 649.79, 30.5, 652.43, 33.83, 654.8, 35.05, 657.07, 35.54], [3.0, 626.24, 12.23, 633.05, 17.83, 639.72, 22.5, 641.63, 23.23, 645.71, 24.87, 645.87, 26.95, 645.49, 27.04], [3.0, 681.78, 55.21, 681.92, 55.32, 682.17, 55.46, 682.17, 55.46, 684.69, 56.92, 684.88, 56.68, 685.52, 56.87], [3.0, 695.81, 20.51, 704.29, 26.47, 714.8, 32.64, 722.99, 39.04, 740.75, 50.06, 742.9, 52.44, 744.71, 55.93]],
        "I_matrix":[[0, 4, 4, 2, 4, 2], [4, 0, 4, 2, 4, 2], [4, 4, 0, 2, 4, 2], [2, 2, 2, 0, 2, 4], [4, 4, 4, 2, 0, 2], [2, 2, 2, 4, 2, 0]],
        "T_matrix": [[0,0.72],[0.29, 0.57], [0.58, 0.65], [0.73, 0.99], [0.78, 0.96], [0.85, 0.99]]
    }
    
    similar_results = query_manager.find_similar_by_msu_data(
            input_f_matrix=input_msu_data["F_matrix"],
            input_i_matrix=input_msu_data["I_matrix"],
            input_t_matrix=input_msu_data["T_matrix"],
            top_k=5
        )
    if similar_results:
        for idx, res in enumerate(similar_results, 1):
            print(f"\n第{idx}名（F距离：{res['f_distance']:.4f}）：")
            print(f"  数据库MSU ID：{res['msu_id']}")
    else:
        print("\n未找到符合条件的相似样本")
    time2 = time.perf_counter()
    duration_ms = (time2 - time1) * 1000
    print(f'查询耗时: {duration_ms:.2f} 毫秒')
    query_manager.close()
    index_manager.close()
    









