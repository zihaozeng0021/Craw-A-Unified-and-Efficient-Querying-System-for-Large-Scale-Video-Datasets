# 构建I矩阵
# 该矩阵包含每个MSU对应的交互矩阵
from F_build import F_matrix
import json
import math

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def I_matrix(F, threshold_close=300, threshold_far=1000, consecutive_points=45):
    I = []
    interaction_start_points = []  # 用于记录首次满足连续重合条件的起始点索引

    for msu in F:  # 遍历每个MSU
        num_objects = len(msu)  # 当前MSU中的对象数量
        msu_result = [[0] * num_objects for _ in range(num_objects)]  # 初始化结果矩阵
        msu_interaction_start = [[-1] * num_objects for _ in range(num_objects)]  # 初始化起始点索引矩阵

        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue  # 跳过对角线

                obj1 = msu[i]
                obj2 = msu[j]

                # 检查是否有连续的重合关系
                intersect = False
                consecutive_count = 0
                start_index = -1  # 用于记录首次满足连续重合条件的起始点索引
                for k in range(0, len(obj1), 2):  # 遍历每个坐标点
                    if obj1[k] != -1 and obj1[k + 1] != -1 and obj2[k] != -1 and obj2[k + 1] != -1:
                        if consecutive_count == 0:  # 记录首次满足条件的起始点索引
                            start_index = k // 2  # 记录第几个cx1
                        consecutive_count += 1
                        if consecutive_count >= consecutive_points:
                            intersect = True
                            break
                    else:
                        consecutive_count = 0  # 重置连续计数

                if intersect:
                    msu_result[i][j] = 1 #相交
                    msu_interaction_start[i][j] = start_index  # 记录起始点索引
                    continue

                # 如果没有相交，计算首尾坐标值的距离
                start_distance = calculate_distance((obj1[0], obj1[1]), (obj2[0], obj2[1]))
                end_distance = calculate_distance((obj1[-2], obj1[-1]), (obj2[-2], obj2[-1]))

                # 判断运动关系
                if abs(start_distance - end_distance) > threshold_close and abs(start_distance - end_distance) < threshold_far: 
                    msu_result[i][j] = 2 #跟随
                elif abs(start_distance - end_distance) > threshold_far:
                    msu_result[i][j] = 3 #远离
                elif abs(start_distance - end_distance) < threshold_close:
                    msu_result[i][j] = 4 #靠近
                else:
                    msu_result[i][j] = 0  # 无显著变化（即自己跟自己）

        I.append(msu_result)
        interaction_start_points.append(msu_interaction_start)

    return I, interaction_start_points


if __name__ == '__main__':
    json_path = r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/frame_info.json'
    with open(json_path, 'r') as f:
        frame_data = json.load(f)
    
    # 调用F_matrix函数
    F, value = F_matrix(frame_data)
    I,inter = I_matrix(value)
    
    # 打印交互矩阵I（与F矩阵格式一致）
    for idx, i_mat in enumerate(I, start=1):
        print(f"=== MSU {idx} 交互矩阵I ===")
        for row in i_mat:
            print(f"  {row}")
        print("\n")
