import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
import time

def cosine_custom(vec1, vec2):

    min_len = min(len(vec1), len(vec2))  # 取较短向量长度作为有效维度
    vec1_cut = vec1[:min_len]  # 截断较长向量
    vec2_cut = vec2[:min_len]
    
    dot_product = np.dot(vec1_cut, vec2_cut)
    norm1 = np.linalg.norm(vec1_cut)  # L2范数（欧几里得长度）
    norm2 = np.linalg.norm(vec2_cut)
    
    if norm1 == 0 or norm2 == 0:  # 避免除以零（零向量无意义）
        return 0.0
    return 1 - (dot_product / (norm1 * norm2))  # 余弦相似度转距离（1-相似度）

def distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))

def TMD(traj1, traj2):
    """
    轨迹匹配距离（Trajectory Matching Distance）：将轨迹展平为向量后计算余弦距离
    traj1/traj2：轨迹列表，格式为[[x1,y1], [x2,y2], ..., [xn,yn]]
    返回值：轨迹间的TMD距离（值越小轨迹越相似）
    """
    vec1 = np.array(traj1).flatten()  # 轨迹展平为1D向量（如[[x1,y1],[x2,y2]]→[x1,y1,x2,y2]）
    vec2 = np.array(traj2).flatten()
    return cosine_custom(vec1, vec2)  

def F_info(row):
    """
    解析输入行的轨迹信息
    row：输入行，格式为[class_id, x1, y1, x2, y2, ..., xn, yn]
    返回值：(class_id, trajectory)，其中trajectory为[[x1,y1], [x2,y2], ..., [xn,yn]]
    """
    class_id = int(row[0])
    coords = row[1:]  # 从第2个元素开始为坐标（排除class_id）
    
    # 确保坐标成对（若长度为奇数，丢弃最后一个元素）
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    
    # 按“x,y”成对解析为轨迹点
    trajectory = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
    return class_id, trajectory

def group_class(F):
    """
    按类别ID分组轨迹
    F：轨迹数据集，每行格式为[class_id, x1, y1, x2, y2, ...]
    返回值：defaultdict，key为class_id，value为该类别的所有轨迹列表
    """
    groups = defaultdict(list)
    for row in F:
        class_id, traj = F_info(row)
        groups[class_id].append(traj)
    return groups

def match_traj(group1, group2, w1=0.5, w2=0.5):
    """
    轨迹一对一匹配：基于轨迹的起点和终点距离加权评分，选择最优匹配
    group1/group2：同一类别的两个轨迹组
    w1/w2：起点距离/终点距离的权重（总和建议为1）
    返回值：匹配对列表，格式为[(group1中轨迹索引, group2中轨迹索引), ...]
    """
    matched_pairs = []
    used_indices = set()  # 标记group2中已匹配的轨迹索引（避免重复匹配）
    
    for i, traj1 in enumerate(group1):
        best_j = None
        best_score = float('inf')  # 评分越小越优（距离越小）
        start1, end1 = traj1[0], traj1[-1]  # traj1的起点和终点
        
        for j, traj2 in enumerate(group2):
            if j in used_indices:
                continue  # 跳过已匹配的traj2
            start2, end2 = traj2[0], traj2[-1]  # traj2的起点和终点
            
            # 计算起点距离和终点距离的加权和（评分）
            dist_start = distance(start1, start2)
            dist_end = distance(end1, end2)
            score = w1 * dist_start + w2 * dist_end
            
            if score < best_score:  # 更新最优匹配
                best_score = score
                best_j = j
        
        if best_j is not None:  # 记录当前traj1的最优匹配
            matched_pairs.append((i, best_j))
            used_indices.add(best_j)
    
    return matched_pairs

def pm_tsmd(FA, FB, base_penalty=1.0):
    """
    PM-TSMD（带惩罚机制的轨迹相似度距离）计算
    FA/FB：两个轨迹数据集
    base_penalty：未匹配轨迹的基础惩罚权重
    返回值：(pm_tsmd_distance, duration_time)，即最终距离和运行时间
    """
    start_time = time.perf_counter()  # 记录开始时间
    
    # 1. 按类别分组轨迹
    groups_A = group_class(FA)
    groups_B = group_class(FB)
    
    result_distance = 0.0  # 总距离
    total_pairs = 0        # 匹配成功的轨迹对数量
    
    # 2. 遍历FA的所有类别，计算跨数据集的相似度
    for class_id in groups_A:
        # 情况1：FA的类别在FB中不存在 → 所有轨迹均未匹配，直接加惩罚
        if class_id not in groups_B:
            result_distance += base_penalty * len(groups_A[class_id])
            continue
        
        # 情况2：FA和FB均有该类别 → 先匹配轨迹，再计算距离和惩罚
        group_A = groups_A[class_id]  # FA中当前类别的所有轨迹
        group_B = groups_B[class_id]  # FB中当前类别的所有轨迹
        
        # 2.1 一对一匹配轨迹
        matched_pairs = match_traj(group_A, group_B)
        
        # 2.2 计算匹配率（用于调整未匹配轨迹的惩罚权重）
        max_traj_num = max(len(group_A), len(group_B))
        match_rate = len(matched_pairs) / max_traj_num if max_traj_num > 0 else 0.0
        penalty_weight = 1.0 - match_rate  # 匹配率越低，惩罚权重越高
        
        # 2.3 计算已匹配轨迹对的TMD距离
        for i, j in matched_pairs:
            traj_dist = TMD(group_A[i], group_B[j])
            result_distance += traj_dist
            total_pairs += 1
        
        # 2.4 计算未匹配轨迹的惩罚（FA中未匹配数 + FB中未匹配数）
        unmatched_A = len(group_A) - len(matched_pairs)
        unmatched_B = len(group_B) - len(matched_pairs)
        result_distance += base_penalty * penalty_weight * (unmatched_A + unmatched_B)
    
    # 3. 计算最终距离（避免除以零）
    final_distance = result_distance / max(total_pairs, 1)
    
    # 4. 计算运行时间
    duration_time = time.perf_counter() - start_time
    
    return final_distance


if __name__ == "__main__":
    # 示例输入：FA和FB为轨迹数据集，每行格式[class_id, x1, y1, x2, y2, ...]
    FA = np.array([
        [0, 1, 9, 2, 11],          # class 0：轨迹[[1,9], [2,11]]
        [1, 3, 6, 4, 9],           # class 1：轨迹[[3,6], [4,9]]
        [2, 0, 3, 2, 9],           # class 2：轨迹[[0,3], [2,9]]
        [0, 4, 6, 5, 8]            # class 0：轨迹[[4,6], [5,8]]
    ])

    FB = np.array([
        [2, 1, 9, 5, 3, 7, 6],     # class 2：轨迹[[1,9], [5,3], [7,6]]（原长度7，截断为6）
        [0, 3, 5, 4, 9, 7, 5],     # class 0：轨迹[[3,5], [4,9], [7,5]]
        [1, 3, 1, 6, 5, 9, 5],     # class 1：轨迹[[3,1], [6,5], [9,5]]
        [3, 7, 7, 5, 3, 1, 5],     # class 3：轨迹[[7,7], [5,3], [1,5]]（FA中无class 3）
        [0, 8, 5, 3, 2, 1, 1]      # class 0：轨迹[[8,5], [3,2], [1,1]]
    ])

    # 计算PM-TSMD距离和运行时间
    pm_tsmd_dist, run_time = pm_tsmd(FA, FB, base_penalty=1.0)
    
    # 输出结果
    print(f"PM-TSMD距离（带相对匹配率惩罚）: {pm_tsmd_dist:.4f}")
    print(f"PM-TSMD计算时间: {run_time:.6f} seconds")