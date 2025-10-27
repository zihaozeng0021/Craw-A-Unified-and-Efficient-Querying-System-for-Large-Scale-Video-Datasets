import math
import matplotlib.pyplot as plt


# 初始化选取6个等分点 + 点索引
def init_select_point(trajectory):
    num = len(trajectory)
    if num < 10:
        return trajectory
    
    select_points = []
    indices = [int(i * (num-1)/5) for i in range(6)]  # 六等分索引
    for idx in indices:
        select_points.append(trajectory[idx])
    
    return select_points

# 欧式距离
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 三角形分析
def analyze_triangle(p1, p2, p3, trajectory, threshold_K=0.10, threshold_angle=150, max_depth=3):
    d_13 = calculate_distance(p1, p3)
    d_total = calculate_distance(trajectory[0], trajectory[-1])
    K = d_13 / d_total

    if K <= threshold_K:
        return [p1, p2, p3]
    
    angle = calculate_angle(p1, p2, p3)
    if angle >= threshold_angle:
        return [p1, p2, p3]
    
    # 递归调整
    d_12 = calculate_distance(p1, p2)
    d_23 = calculate_distance(p2, p3)
    
    if d_12 == d_23:
        return [p1, p2, p3]
    
    # 根据边长调整中间点
    if d_12 > d_23:
        mid_index = (p1[2] + p2[2]) // 2
        new_p2 = trajectory[mid_index]
        if max_depth > 0:
            return analyze_triangle(p1, new_p2, p3, trajectory, threshold_K, threshold_angle, max_depth-1)
        else:
            return [p1, p2, p3]
    else:
        mid_index = (p2[2] + p3[2]) // 2
        new_p2 = trajectory[mid_index]
        if max_depth > 0:
            return analyze_triangle(p1, new_p2, p3, trajectory, threshold_K, threshold_angle, max_depth-1)
        else:
            return [p1, p2, p3]

# 向量夹角计算（b是原点）
def calculate_angle(point_a, point_b, point_c):
    x1, y1 = point_a[0] - point_b[0], point_a[1] - point_b[1]
    x2, y2 = point_c[0] - point_b[0], point_c[1] - point_b[1]
    
    # 计算向量的长度
    len1 = math.sqrt(x1**2 + y1**2)
    len2 = math.sqrt(x2**2 + y2**2)
    
    # 如果某个向量的长度为零，返回默认值
    if len1 == 0 or len2 == 0:
        return 0  # 或者返回其他默认值，例如 180
    
    cos_theta = (x1*x2 + y1*y2) / (len1 * len2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # 防止数值溢出
    return math.degrees(math.acos(cos_theta))

# 分析原始轨迹主函数, 改几分点的时候也要改range数
def analyze_trajectory(trajectory):
    # 获取六等分点
    points = init_select_point(trajectory)
    if len(points) < 6:
        return points
    
    # 分析四个三角形
    triangles = [points[i:i+3] for i in range(4)]  # [ABC, BCD, CDE, DEF]
    key_points = set()
    
    for tri in triangles:
        p1, p2, p3 = tri
        result = analyze_triangle(p1, p2, p3, trajectory)
        for point in result:
            key_points.add(point)  # 使用集合自动去重
    
    # 按原始索引排序
    sorted_points = sorted(key_points, key=lambda x: x[2])

    return sorted_points

# 画图
def plot_trajectory(trajectory, key_points):
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  
    plt.figure(figsize=(12, 8))
    # 绘制原始轨迹
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    plt.plot(x_coords, y_coords, 'b--', label='原始轨迹')
    plt.scatter(x_coords, y_coords, c='lightblue', s=30, label='原始点')
    
    # 绘制关键点
    x_key = [p[0] for p in key_points]
    y_key = [p[1] for p in key_points]
    plt.plot(x_key, y_key, 'r-', linewidth=2, label='关键轨迹')
    plt.scatter(x_key, y_key, c='red', s=80, zorder=5, label='关键点')
    
    plt.legend()
    plt.title('轨迹关键点提取')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 去除无用值
def clean_list(value):
    # 删除索引为 0 的值
    if value:
        value.pop(0)
    # 删除所有值为 -1 的元素
    cleaned_list = [x for x in value if x != -1]
    return cleaned_list

def convert_to_trajectory(points):  
    # 将一维列表转换为 (x, y, index) 格式的轨迹点列表
    trajectory = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        trajectory.append((x, y, i // 2))
    return trajectory

def convert_keypoints_tuple(keypoints):
    key_points = []
    for i in range(len(keypoints)):
        x = keypoints[i][0]
        y = keypoints[i][1]
        key_points.append(x)
        key_points.append(y)
    return key_points

def imp_kps(all_trajectories, k_points): 
    length_dic = {}  
    value_list = []  
    for j in range(len(k_points)):
        # 新增：若k_points[j]仅含类别（无关键点），长度记为0
        key_len = len(k_points[j]) - 1 if len(k_points[j]) > 0 else 0
        length_dic[j] = key_len
        value_list.append(key_len)
    # 新增：处理value_list为空的情况（无有效轨迹）
    if not value_list:
        #print("警告：无有效轨迹的关键点数据")
        return k_points
    # 处理value_list全为0的情况（所有轨迹无关键点）
    max_value = max(value_list) if any(value_list) else 0
    if max_value == 0:
        #print("警告：所有轨迹均无有效关键点，无需统一长度")
        return k_points
    # 后续统一长度逻辑不变...
    for i in range(len(k_points)):
        if value_list[i] != max_value:
            tmp_list = k_points[i][1:] if len(k_points[i]) > 1 else []
            k_points[i] = [k_points[i][0]] + deal_length(tmp_list, all_trajectories[i], max_value)
    return k_points


def deal_length(tmp_list, trajectory, max_value):
    tmp_imp = max_value - len(tmp_list)  # 需要补充的点数
    if tmp_imp <= 0:
        return tmp_list  

    if len(tmp_list) < 2:
        if len(tmp_list) == 1:
            while len(tmp_list) < max_value:
                # 为简单起见，复制最后一个点的坐标和索引
                last_point = tmp_list[-1]
                tmp_list.append((last_point[0], last_point[1], last_point[2]))
            # 确保不超过 max_value
            return tmp_list[:max_value] 
        else: # len(tmp_list) == 0

             print(f"Warning: Cannot process empty keypoint list in deal_length.")
             return tmp_list

    while tmp_imp > 0:
        max_distance = 0
        max_index = 0
        for m in range(len(tmp_list) - 1):
            p1 = (tmp_list[m][0], tmp_list[m][1])
            p2 = (tmp_list[m + 1][0], tmp_list[m + 1][1])
            distance = calculate_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
                max_index = m  # 记录最大距离的起始点索引

        # 获取最大距离的两个关键点对应的原始轨迹索引
        start_index = tmp_list[max_index][2]
        end_index = tmp_list[max_index + 1][2]

        # 计算中心索引点
        center_index = (start_index + end_index) // 2

        if 0 <= center_index < len(trajectory):
            center_point = trajectory[center_index]
        else:
            print(f"Warning: center_index {center_index} out of range for trajectory (len={len(trajectory)}). Using start point.")
            center_point = tmp_list[max_index] # 或 tmp_list[max_index + 1]

        # 将中心点插入到关键点列表中
        tmp_list.insert(max_index + 1, center_point)
        # 减少需要补充的点数
        tmp_imp -= 1
    return tmp_list
# 主函数
def process_traj(traj_list):
    all_trajectories = [] 
    k_points = [] 
    for traj in traj_list:
        cls = traj[0]  # 提取类别（这一步正确，因为clean_list会删除traj[0]）
        k_tmp = [cls]  
        # 1. 清理坐标（删除traj[0]和-1）
        cleaned_traj = clean_list(traj)  
        # 新增：若清理后无有效坐标，跳过该轨迹或返回默认值
        if not cleaned_traj:
            print(f"警告：轨迹类别 {cls} 无有效坐标，跳过处理")
            k_points.append(k_tmp)  # 仅保留类别，后续可补充默认值
            all_trajectories.append([])
            continue
        # 2. 转换为轨迹格式
        trajectory = convert_to_trajectory(cleaned_traj)
        all_trajectories.append(trajectory)
        # 3. 提取关键点（若轨迹点数太少，直接返回轨迹）
        key_points = analyze_trajectory(trajectory) if len(trajectory) > 0 else []
        if not key_points:  # 若未提取到关键点，用原始轨迹前N个点替代
            key_points = trajectory[:5]  # 取前5个点作为默认关键点
        k_tmp.extend(key_points)
        k_points.append(k_tmp)
    # 4. 统一关键点长度（后续逻辑不变）
    k_points = imp_kps(all_trajectories, k_points)
    # 5. 格式化输出（后续逻辑不变）
    for i in range(len(k_points)):
        cls = k_points[i][0]
        k_temp = [cls]
        k_temp.extend(convert_keypoints_tuple(k_points[i][1:]))
        k_points[i] = k_temp
    return k_points


if __name__ == '__main__':
    
    value = [
        [0, -1, -1, 1094.5, 531.2, 737.7, 265.6, 736.2, 265.0, 735.7, 264.9, 733.5, 264.6, 731.5, 263.3, 730.9, 262.9, 730.2, 262.2, 729.3, 261.8, 728.6, 260.7, 728.4, 260.3, 728.3, 260.2, 726.4, 259.5, 725.1, 258.6, 723.5, 257.7, 722.9, 257.4, 722.0, 256.6, 721.4, 256.2, 720.7, 256.3, 720.6, 256.5, 720.7, 256.4, 720.7, 256.4, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2]
    ]
    
    kps = process_traj(value)
    print("类别+关键点：", kps)


'''
import math
import matplotlib.pyplot as plt

# 初始化选取6个等分点 + 点索引
def init_select_point(trajectory):
    num = len(trajectory)
    if num < 6:
        return trajectory
    
    select_points = []
    indices = [int(i * (num-1)/5) for i in range(6)]  # 六等分索引
    for idx in indices:
        select_points.append(trajectory[idx])
    
    return select_points

# 欧式距离
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 三角形分析
def analyze_triangle(p1, p2, p3, trajectory, threshold_K=0.10, threshold_angle=150, max_depth=3):
    d_13 = calculate_distance(p1, p3)
    d_total = calculate_distance(trajectory[0], trajectory[-1])
    K = d_13 / d_total

    if K <= threshold_K:
        return [p1, p2, p3]
    
    angle = calculate_angle(p1, p2, p3)
    if angle >= threshold_angle:
        return [p1, p2, p3]
    
    # 递归调整
    d_12 = calculate_distance(p1, p2)
    d_23 = calculate_distance(p2, p3)
    
    if d_12 == d_23:
        return [p1, p2, p3]
    
    # 根据边长调整中间点
    if d_12 > d_23:
        mid_index = (p1[2] + p2[2]) // 2
        new_p2 = trajectory[mid_index]
        if max_depth > 0:
            return analyze_triangle(p1, new_p2, p3, trajectory, threshold_K, threshold_angle, max_depth-1)
        else:
            return [p1, p2, p3]
    else:
        mid_index = (p2[2] + p3[2]) // 2
        new_p2 = trajectory[mid_index]
        if max_depth > 0:
            return analyze_triangle(p1, new_p2, p3, trajectory, threshold_K, threshold_angle, max_depth-1)
        else:
            return [p1, p2, p3]

# 向量夹角计算（b是原点）
def calculate_angle(point_a, point_b, point_c):
    x1, y1 = point_a[0] - point_b[0], point_a[1] - point_b[1]
    x2, y2 = point_c[0] - point_b[0], point_c[1] - point_b[1]
    
    # 计算向量的长度
    len1 = math.sqrt(x1**2 + y1**2)
    len2 = math.sqrt(x2**2 + y2**2)
    
    # 如果某个向量的长度为零，返回默认值
    if len1 == 0 or len2 == 0:
        return 0  # 或者返回其他默认值，例如 180
    
    cos_theta = (x1*x2 + y1*y2) / (len1 * len2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # 防止数值溢出
    return math.degrees(math.acos(cos_theta))

# 分析原始轨迹主函数, 改几分点的时候也要改range数
def analyze_trajectory(trajectory):
    # 获取六等分点
    points = init_select_point(trajectory)
    if len(points) < 6:
        return points
    
    # 分析四个三角形
    triangles = [points[i:i+3] for i in range(4)]  # [ABC, BCD, CDE, DEF]
    key_points = set()
    
    for tri in triangles:
        p1, p2, p3 = tri
        result = analyze_triangle(p1, p2, p3, trajectory)
        for point in result:
            key_points.add(point)  # 使用集合自动去重
    
    # 按原始索引排序
    sorted_points = sorted(key_points, key=lambda x: x[2])

    return sorted_points

# 画图
def plot_trajectory(trajectory, key_points):
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  
    plt.figure(figsize=(12, 8))
    # 绘制原始轨迹
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    plt.plot(x_coords, y_coords, 'b--', label='原始轨迹')
    plt.scatter(x_coords, y_coords, c='lightblue', s=30, label='原始点')
    
    # 绘制关键点
    x_key = [p[0] for p in key_points]
    y_key = [p[1] for p in key_points]
    plt.plot(x_key, y_key, 'r-', linewidth=2, label='关键轨迹')
    plt.scatter(x_key, y_key, c='red', s=80, zorder=5, label='关键点')
    
    plt.legend()
    plt.title('轨迹关键点提取')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 去除无用值
def clean_list(value):
    # 删除索引为 0 的值
    if value:
        value.pop(0)
    # 删除所有值为 -1 的元素
    cleaned_list = [x for x in value if x != -1]
    return cleaned_list

def convert_to_trajectory(points):  
    # 将一维列表转换为 (x, y, index) 格式的轨迹点列表
    trajectory = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        trajectory.append((x, y, i // 2))
    return trajectory

def convert_keypoints_tuple(keypoints):
    key_points = []
    for i in range(len(keypoints)):
        x = keypoints[i][0]
        y = keypoints[i][1]
        key_points.append(x)
        key_points.append(y)
    return key_points

def imp_kps(all_trajectories, k_points): 
    length_dic = {}  # 统计每个轨迹的关键点数量
    value_list = []  # 存放所有轨迹的关键点数量
    for j in range(len(k_points)):
        length_dic[j] = len(k_points[j]) - 1  # 减去类别标签
        value_list.append(length_dic[j])
    max_value = max(value_list)  # 找到最大关键点数量
    for i in range(len(k_points)):
        if value_list[i] != max_value:
            tmp_list = k_points[i][1:]  # 获取关键点列表，跳过类别标签
            # 传入对应的 trajectory
            k_points[i] = [k_points[i][0]] + deal_length(tmp_list, all_trajectories[i], max_value) # 修改这里
    return k_points


def deal_length(tmp_list, trajectory, max_value):
    tmp_imp = max_value - len(tmp_list)  # 需要补充的点数
    if tmp_imp <= 0:
        return tmp_list  

    if len(tmp_list) < 2:
        if len(tmp_list) == 1:
            while len(tmp_list) < max_value:
                # 为简单起见，复制最后一个点的坐标和索引
                last_point = tmp_list[-1]
                tmp_list.append((last_point[0], last_point[1], last_point[2]))
            # 确保不超过 max_value
            return tmp_list[:max_value] 
        else: # len(tmp_list) == 0

             print(f"Warning: Cannot process empty keypoint list in deal_length.")
             return tmp_list

    while tmp_imp > 0:
        max_distance = 0
        max_index = 0
        for m in range(len(tmp_list) - 1):
            p1 = (tmp_list[m][0], tmp_list[m][1])
            p2 = (tmp_list[m + 1][0], tmp_list[m + 1][1])
            distance = calculate_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
                max_index = m  # 记录最大距离的起始点索引

        # 获取最大距离的两个关键点对应的原始轨迹索引
        start_index = tmp_list[max_index][2]
        end_index = tmp_list[max_index + 1][2]

        # 计算中心索引点
        center_index = (start_index + end_index) // 2

        if 0 <= center_index < len(trajectory):
            center_point = trajectory[center_index]
        else:
            print(f"Warning: center_index {center_index} out of range for trajectory (len={len(trajectory)}). Using start point.")
            center_point = tmp_list[max_index] # 或 tmp_list[max_index + 1]

        # 将中心点插入到关键点列表中
        tmp_list.insert(max_index + 1, center_point)
        # 减少需要补充的点数
        tmp_imp -= 1
    return tmp_list
# 主函数
def process_traj(traj_list):
    all_trajectories = [] # 存储所有轨迹
    k_points = [] #
    for traj in traj_list:
        cls = traj[0]  # 提取类别
        k_tmp = [cls]  # 初始化关键点列表，包含类别
        # 清理列表并转换为轨迹点格式
        cleaned_traj = clean_list(traj)  # 跳过类别标签
        trajectory = convert_to_trajectory(cleaned_traj)
        all_trajectories.append(trajectory) # 保存 trajectory
        # 分析轨迹并提取关键点
        key_points = analyze_trajectory(trajectory)
        k_tmp.extend(key_points)
        k_points.append(k_tmp)

    # 添加关键点并统一长度 - 传入所有轨迹
    k_points = imp_kps(all_trajectories, k_points) # 修改这里
    for i in range(len(k_points)):
        cls = k_points[i][0]  # 提取类别
        k_temp = [cls]  # 初始化关键点列表，包含类别
        k_temp.extend(convert_keypoints_tuple(k_points[i][1:]))
        k_points[i] = k_temp
    return k_points


if __name__ == '__main__':
    
    value = [
        [0, -1, -1, 1094.5, 531.2, 737.7, 265.6, 736.2, 265.0, 735.7, 264.9, 733.5, 264.6, 731.5, 263.3, 730.9, 262.9, 730.2, 262.2, 729.3, 261.8, 728.6, 260.7, 728.4, 260.3, 728.3, 260.2, 726.4, 259.5, 725.1, 258.6, 723.5, 257.7, 722.9, 257.4, 722.0, 256.6, 721.4, 256.2, 720.7, 256.3, 720.6, 256.5, 720.7, 256.4, 720.7, 256.4, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9, 10.2]
    ]
    
    kps = process_traj(value)
    print("类别+关键点：", kps)
'''