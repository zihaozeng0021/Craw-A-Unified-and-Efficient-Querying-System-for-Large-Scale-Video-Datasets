import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import glob
import os
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA  # 提前导入PCA，避免计时内导入耗时

def omd(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    
    if np.array_equal(a, b):
        return 0.0
    
    max_features = 30
    if len(a) > max_features:
        indices = np.linspace(0, len(a)-1, max_features, dtype=int)
        a = a[indices]
    
    if len(b) > max_features:
        indices = np.linspace(0, len(b)-1, max_features, dtype=int)
        b = b[indices]
    
    a = np.array(a, dtype=np.float32) 
    b = np.array(b, dtype=np.float32)
    
    max_dim = 256
    if a.shape[1] > max_dim:
        a = a[:, :max_dim]
    if b.shape[1] > max_dim:
        b = b[:, :max_dim]
    
    min_features = min(a.shape[0], b.shape[0])
    if min_features == 0:
        return float('inf')
    
    if a.shape[0] > min_features:
        a = a[:min_features]
    if b.shape[0] > min_features:
        b = b[:min_features]
    
    try:
        dist_mat = np.zeros((len(a), len(b)), dtype=np.float32)
        for i in range(len(a)):
            for j in range(len(b)):
                dist_mat[i, j] = np.linalg.norm(a[i] - b[j])
        
        w1 = np.ones(len(a), dtype=np.float32) / len(a)
        w2 = np.ones(len(b), dtype=np.float32) / len(b)
        
        from pyemd import emd
        distance = emd(w1, w2, dist_mat)
        return float(distance)
    except Exception as e:
        try:
            avg_a = np.mean(a, axis=0)
            avg_b = np.mean(b, axis=0)
            distance = np.linalg.norm(avg_a - avg_b)
            return float(distance)
        except:
            return float('inf')

def find_optimal_clusters(distance_matrix, max_clusters=10):
    """使用轮廓系数最大化自动确定最优聚类数"""
    distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    max_clusters = min(max_clusters, len(distance_matrix)-1)
    if max_clusters < 4:
        return 4
    
    silhouette_scores = []
    k_range = range(4, max_clusters + 1)

    print("正在计算不同聚类数的轮廓系数...")
    
    for k in k_range:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
            
            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
                silhouette_scores.append(silhouette_avg)
                print(f"  k={k}: 轮廓系数={silhouette_avg:.4f}")
            else:
                silhouette_scores.append(-1)
                print(f"  k={k}: 轮廓系数=-1.0000 (无效)")
        except Exception as e:
            print(f"  k={k}: 计算出错 - {e}")
            silhouette_scores.append(-1)
    
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        print(f"最优聚类数: {optimal_k} (轮廓系数: {max_silhouette:.4f})")
        return optimal_k
    else:
        print("无法确定最优聚类数，使用默认值4")
        return 4

def compute_omd_distance_matrix(svs_features_list):
    """计算所有SVS之间的OMD距离矩阵"""
    n = len(svs_features_list)
    distance_matrix = np.zeros((n, n))
    
    print("计算OMD距离矩阵...")
    for i in range(n):
        for j in range(i+1, n):
            distance = omd(svs_features_list[i], svs_features_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        if (i+1) % 10 == 0 or i+1 == n:
            print(f"  进度: {i+1}/{n}")
    
    return distance_matrix

def cluster_svs(svs_features_list, num_clusters=None):
    """仅执行核心聚类逻辑（输入已加载的特征，避免计时内读文件）"""
    # ---------------------- 核心聚类计时开始 ----------------------
    cluster_start_time = time.perf_counter()
    
    # 1. 计算OMD距离矩阵（核心步骤1）
    distance_matrix = compute_omd_distance_matrix(svs_features_list)
    
    # 2. 确定聚类数量（核心步骤2）
    if num_clusters is None:
        actual_num_clusters = find_optimal_clusters(distance_matrix)
    else:
        actual_num_clusters = min(num_clusters, len(svs_features_list))
    
    print(f"执行聚类分析，聚类数量: {actual_num_clusters}")
    
    # 3. 执行层次聚类（核心步骤3）
    try:
        clustering = AgglomerativeClustering(
            n_clusters=actual_num_clusters,
            metric='precomputed',
            linkage='average'
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=actual_num_clusters,
            affinity='precomputed',
            linkage='average'
        )
    labels = clustering.fit_predict(distance_matrix)
    

    cluster_end_time = time.perf_counter()
    core_cluster_duration = cluster_end_time - cluster_start_time
    print(f"\n【核心聚类总耗时】: {core_cluster_duration:.2f} 秒") 
    
    return labels, distance_matrix, actual_num_clusters, core_cluster_duration

def visualize_clusters(svs_features_list, labels, save_path):
    """
    优化的聚类可视化函数：
    1. 直接使用已加载的特征列表，避免重复读文件
    2. 检查保存路径目录，不存在则创建
    3. 增强异常捕获，输出详细错误信息
    """
    try:
        # 1. 检查保存路径的目录是否存在，不存在则创建
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"已创建图片保存目录: {save_dir}")
        
        # 2. 计算每个SVS的平均特征（用于PCA降维）
        all_avg_features = []
        for features in svs_features_list:
            avg_feature = np.mean(features, axis=0)  # 单SVS的平均特征
            all_avg_features.append(avg_feature)
        all_avg_features = np.array(all_avg_features)
        
        # 3. PCA降维到2D（便于可视化）
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_avg_features)  # 所有SVS的2D特征
        
        # 4. 计算每个聚类的中心（2D空间）
        unique_labels = np.unique(labels)
        cluster_centers_2d = []
        for label in unique_labels:
            # 取该聚类所有SVS的2D特征，计算中心
            cluster_2d_features = features_2d[labels == label]
            cluster_center_2d = np.mean(cluster_2d_features, axis=0)
            cluster_centers_2d.append(cluster_center_2d)
        cluster_centers_2d = np.array(cluster_centers_2d)
        
        # 5. 绘制可视化图
        plt.figure(figsize=(12, 8))  # 放大图尺寸，提升清晰度
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']  # 更鲜明的配色
        
        # 绘制每个聚类的SVS点
        for idx, label in enumerate(unique_labels):
            cluster_points = features_2d[labels == label]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=colors[idx % len(colors)],  # 循环使用颜色
                label=f'Cluster {label} (n={len(cluster_points)})',  # 标注聚类包含的SVS数量
                alpha=0.8,  # 提升点的透明度，避免重叠遮挡
                s=80  # 放大点的尺寸，更易观察
            )
        
        # 绘制聚类中心（黑色叉号，突出显示）
        plt.scatter(
            cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
            c='black', marker='x', s=300, linewidths=4,
            label='Cluster Centers', zorder=5  # zorder确保中心在最上层
        )
        
        # 图注与格式优化
        plt.title('SVS Clustering Result Visualization (PCA 2D)', fontsize=16, fontweight='bold')
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.legend(fontsize=10, loc='best')  # 自动选择最优图例位置
        plt.grid(True, alpha=0.3, linestyle='--')  # 虚线网格，提升可读性
        plt.tight_layout()  # 自动调整布局，避免元素被截断
        
        # 保存图片（使用高分辨率）
        plt.savefig(
            save_path,
            dpi=300,  # 高分辨率，适合后续分析
            bbox_inches='tight'  # 避免图例被截断
        )
        plt.close()  # 关闭图，释放内存
        print(f"\n✅ 聚类可视化图已成功保存到: {save_path}")

    except Exception as e:
        # 输出详细错误信息，便于排查问题
        print(f"\n❌ 聚类可视化失败:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误详情: {str(e)}")

if __name__ == "__main__":
    # ---------------------- 1. 加载SVS特征文件 ----------------------
    file_load_start = time.perf_counter()
    
    svs_dir = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin'
    svs_files = sorted(glob.glob(f"{svs_dir}/*.pkl"))
    svs_features_list = []
    valid_svs_files = []
    
    if not svs_files:
        exit()
    
    for svs_file in svs_files:
        try:
            with open(svs_file, 'rb') as f:
                features = pickle.load(f)
                if len(features) > 0 and features.ndim == 2:  # 额外检查特征格式（确保是2D数组）
                    svs_features_list.append(features)
                    valid_svs_files.append(svs_file)
                else:
                    print(f"⚠️ 跳过无效文件 {os.path.basename(svs_file)}: 特征为空或格式错误")
        except Exception as e:
            print(f"⚠️ 读取文件 {os.path.basename(svs_file)} 时出错: {str(e)}")
    
    if len(svs_features_list) == 0:
        print("⚠️ 没有加载到有效SVS特征数据，程序退出")
        exit()
    
    file_load_end = time.perf_counter()
    print(f"\n📥 文件加载完成:")
    print(f"  - 有效SVS数量: {len(svs_features_list)}")
    print(f"  - 加载耗时: {file_load_end - file_load_start:.2f} 秒")

    # ---------------------- 2. 执行核心聚类 ----------------------
    labels, distance_matrix, actual_num_clusters, core_cluster_time = cluster_svs(svs_features_list)

    # ---------------------- 3. 选择聚类代表点 & 保存聚类结果 ----------------------
    result_process_start = time.perf_counter()
    
    # 选择每个聚类的"中心代表"（平均距离最小的SVS）
    cluster_representatives = []
    for i in range(actual_num_clusters):
        cluster_indices = np.where(labels == i)[0]
        min_avg_distance = float('inf')
        representative_idx = cluster_indices[0]
        
        for idx in cluster_indices:
            avg_distance = np.mean(distance_matrix[idx][cluster_indices])
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                representative_idx = idx
        
        representative_file = valid_svs_files[representative_idx]
        cluster_representatives.append(representative_file)
        print(f"\n🏷️  聚类 {i}:")
        print(f"  - 代表SVS: {os.path.basename(representative_file)}")
        print(f"  - 包含SVS数量: {len(cluster_indices)}")
    
    # 保存聚类结果（包含可视化所需信息）
    cluster_result_path = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin/cluster_result.pkl'
    cluster_info = {
        'labels': labels,
        'svs_files': valid_svs_files,
        'representatives': cluster_representatives,
        'core_cluster_time_seconds': core_cluster_time,
        'optimal_clusters': actual_num_clusters  # 补充最优聚类数
    }
    with open(cluster_result_path, 'wb') as f:
        pickle.dump(cluster_info, f)
    print(f"\n💾 聚类结果已保存到: {cluster_result_path}")
    
    result_process_end = time.perf_counter()


    cluster_img_path = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin/cluster_visualization.png'
    visualize_clusters(
        svs_features_list=svs_features_list,  # 直接传已加载的特征，避免重复读文件
        labels=labels,
        save_path=cluster_img_path
    )

    # ---------------------- 5. 输出最终统计 ----------------------
    print(f"\n📊 最终统计:")
    print(f"  - 核心聚类耗时: {core_cluster_time:.2f} 秒（距离矩阵计算+聚类分析）")
    print(f"  - 结果处理耗时: {result_process_end - result_process_start:.2f} 秒（代表点选择+结果保存）")
    print(f"  - 总耗时: {time.perf_counter() - file_load_start:.2f} 秒")
    print(f"  - 可视化图路径: {cluster_img_path}")