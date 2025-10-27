import pickle
import numpy as np
import os
import time
np.random.seed(42)


def omd(a, b):
    # 核心校验：确保输入是2D数组且维度一致
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.ndim != 2 or b.ndim != 2:
        return float('inf'), float('inf')  # 返回(原始值, 归一化值)
    if len(a) == 0 or len(b) == 0 or a.shape[1] != b.shape[1]:
        return float('inf'), float('inf')
    
    # 完全相同特征返回极小值
    if np.array_equal(a, b):
        raw_dist = 0.01
        norm_dist = 0.01
        return raw_dist, norm_dist
    
    max_features = 64
    if len(a) > max_features:
        base_indices = np.linspace(0, len(a)-1, max_features, dtype=int)
        offset_range = max(1, int(0.1 * len(a)))
        offsets = np.random.randint(-offset_range, offset_range+1, size=max_features)
        a = a[np.clip(base_indices+offsets, 0, len(a)-1)]
    if len(b) > max_features:
        base_indices = np.linspace(0, len(b)-1, max_features, dtype=int)
        offset_range = max(1, int(0.1 * len(b)))
        offsets = np.random.randint(-offset_range, offset_range+1, size=max_features)
        b = b[np.clip(base_indices+offsets, 0, len(b)-1)]

    a = np.array(a, dtype=np.float32)[:, :256]
    b = np.array(b, dtype=np.float32)[:, :256]

    min_feat = min(a.shape[0], b.shape[0])
    if min_feat == 0:
        return float('inf'), float('inf')
    a = a[:min_feat]
    b = b[:min_feat]

    # 计算原始OMD
    try:
        # 计算pairwise L2
        dist_mat = np.zeros((len(a), len(b)), dtype=np.float32)
        for i in range(len(a)):
            for j in range(len(b)):
                dist_mat[i, j] = np.linalg.norm(a[i]-b[j]) + np.random.normal(0, 5e-7)
        # EMD计算
        w1 = np.ones(len(a), dtype=np.float32)/len(a)
        w2 = np.ones(len(b), dtype=np.float32)/len(b)
        from pyemd import emd
        raw_dist = max(float(emd(w1, w2, dist_mat)), 1e-6)
    except Exception:
        # 退回到平均距离
        try:
            raw_dist = max(float(np.linalg.norm(a.mean(axis=0)-b.mean(axis=0))), 1e-6)
        except Exception:
            raw_dist = float('inf')
    
    # 计算归一化OMD（原始值/10，限制0.01~1.0）
    if raw_dist == float('inf'):
        norm_dist = 1.0
    else:
        norm_dist = raw_dist / 10.0
        norm_dist = min(max(norm_dist, 0.01), 1.0)  # 避免0值，限制范围
    
    return raw_dist, norm_dist  # 同时返回两个结果


def get_svs_id(filepath):
    # 简化SVS ID提取
    try:
        return int(os.path.basename(filepath).split('_')[1].split('.')[0])
    except Exception:
        return -1


def parse_cluster(cluster_path, cluster_svs_dir):
    # 跨目录查询：用聚类SVS目录（如origin）找文件
    with open(cluster_path, 'rb') as f:
        cluster_info = pickle.load(f)
    labels = cluster_info['labels']
    svs_files = cluster_info['svs_files']

    # 筛选聚类目录中存在的SVS
    valid_labels = []
    valid_svs = []
    for lbl, svs_path in zip(labels, svs_files):
        svs_filename = os.path.basename(svs_path)
        actual_path = os.path.join(cluster_svs_dir, svs_filename)
        if os.path.exists(actual_path):
            valid_labels.append(lbl)
            valid_svs.append(actual_path)

    # 计算聚类中心（统一256维，返回2D数组）
    unique_labels = np.unique(valid_labels)
    centers = {}
    members = {}
    for lb in unique_labels:
        curr_svs = [valid_svs[idx] for idx, l in enumerate(valid_labels) if l == lb]
        avg_feats = []
        member_ids = []
        for svs in curr_svs:
            with open(svs, 'rb') as f:
                feat = pickle.load(f)[:, :256]
            avg_feats.append(feat.mean(axis=0))
            member_ids.append(get_svs_id(svs))
        # 聚类中心转2D数组，匹配omd输入
        centers[lb] = np.mean(avg_feats, axis=0)[np.newaxis, :]
        members[lb] = member_ids

    sorted_lb = sorted(unique_labels)
    return {
        'sorted_labels': sorted_lb,
        'centers': [centers[lb] for lb in sorted_lb],
        'members': [members[lb] for lb in sorted_lb],
        'total_svs': len(valid_svs),
        'cluster_dir': cluster_svs_dir
    }


def cluster_query(query_feat, cluster_path, cluster_svs_dir, top_k=5, mismatch_thresh=10.0):
    # 解析聚类数据
    cluster_data = parse_cluster(cluster_path, cluster_svs_dir)
    sorted_lb = cluster_data['sorted_labels']
    centers = cluster_data['centers']
    members = cluster_data['members']
    cluster_dir = cluster_data['cluster_dir']

    # 计算查询与聚类中心的距离（原始+归一化）
    start_time = time.time()
    cluster_dist = []
    for lb, center in zip(sorted_lb, centers):
        raw, norm = omd(query_feat, center)
        cluster_dist.append((lb, center, raw, norm))
    # 按原始OMD升序排序（优先相似聚类）
    cluster_dist.sort(key=lambda x: x[2])

    # 处理聚类内SVS，保留符合阈值的结果
    valid_res = []
    cluster_idx = 0
    while cluster_idx < len(cluster_dist):
        lb, _, _, _ = cluster_dist[cluster_idx]
        svs_ids = members[cluster_idx]
        # 遍历SVS，从聚类目录找文件
        for svs_id in svs_ids:
            svs_file = os.path.join(cluster_dir, f"msu_{svs_id}_features.pkl")
            if not os.path.exists(svs_file):
                continue
            # 计算SVS与查询的距离（原始+归一化）
            with open(svs_file, 'rb') as f:
                svs_feat = pickle.load(f)[:, :256]
            raw_dist, norm_dist = omd(query_feat, svs_feat)
            # 仅保留原始OMD≤阈值的结果
            if raw_dist <= mismatch_thresh:
                valid_res.append((svs_id, svs_file, raw_dist, norm_dist, lb))
        # 按原始OMD升序排序
        valid_res.sort(key=lambda x: x[2])
        # 满足top-k则停止
        if len(valid_res) >= top_k:
            if cluster_idx + 1 < len(cluster_dist) and valid_res[top_k-1][2] <= cluster_dist[cluster_idx+1][2]:
                break
        cluster_idx += 1

    # 取top-k结果
    top_res = valid_res[:top_k]
    query_time = time.time() - start_time

    # 组装结果（含原始+归一化值）
    results = []
    for svs_id, svs_file, raw, norm, lb in top_res:
        results.append({
            "svs_id": svs_id,
            "svs_file": os.path.basename(svs_file),
            "cluster_label": lb,
            "raw_omd": round(raw, 6),
            "normalized_omd": round(norm, 4)
        })
    return {
        "status": "success",
        "results": results,
        "total_valid": len(valid_res),
        "query_time": round(query_time, 4)
    }


def main():

    query_svs_dir = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/video/90/pkl' 
    #query_svs_dir = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin'  
    cluster_svs_dir = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin'  
    cluster_result_path = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin/cluster_result.pkl'  
    mismatch_threshold = 3
    top_k = 5
    total_queries = 98

    total_hits = 0
    hit_positions = []
    total_time = 0.0
    valid_queries = 0  

    for query_id in range(1, total_queries + 1):
        query_path = os.path.join(query_svs_dir, f"msu_{query_id}_features.pkl")
        print(f"\n处理第 {query_id}/{total_queries} 个查询（ID: {query_id}）")

        try:
            with open(query_path, 'rb') as f:
                query_feat = pickle.load(f)[:, :256]
            if not isinstance(query_feat, np.ndarray) or query_feat.ndim != 2:
                print(f"  警告：查询特征格式错误，跳过")
                continue
        except Exception as e:
            print(f"  警告：加载特征失败，跳过（{str(e)}）")
            continue

        result = cluster_query(
            query_feat=query_feat,
            cluster_path=cluster_result_path,
            cluster_svs_dir=cluster_svs_dir,
            top_k=top_k,
            mismatch_thresh=mismatch_threshold
        )


        valid_queries += 1
        total_time += result['query_time']


        print(f"  本次查询耗时：{result['query_time']:.4f} 秒")
        for idx, item in enumerate(result['results'], 1):
            print(f"  第{idx}名：SVS {item['svs_id']}（{item['svs_file']}）")
            print(f"    原始OMD：{item['raw_omd']}, 归一化OMD：{item['normalized_omd']}")

        # 检查是否命中（查询ID是否在top-k结果中）
        top_k_ids = [item['svs_id'] for item in result['results']]
        if query_id in top_k_ids:
            hit_pos = top_k_ids.index(query_id) + 1  # 位置从1开始
            total_hits += 1
            hit_positions.append(hit_pos)
            print(f"  ✅ 命中！查询ID在top-{top_k}中的位置：{hit_pos}")
        else:
            print(f"  ❌ 未命中")


    print(f"查询汇总（共{valid_queries}个有效查询）")
    print(f"1. 命中率：{total_hits}/{valid_queries} = {total_hits/valid_queries*100:.2f}%")
    print(f"2. 命中样本的top-k位置均值：{np.mean(hit_positions):.2f}" if hit_positions else "2. 无命中样本")
    print(f"3. 平均查询时间：{total_time/valid_queries*1000:.2f}毫秒" if valid_queries > 0 else "3. 无有效查询")



if __name__ == "__main__":
    main()