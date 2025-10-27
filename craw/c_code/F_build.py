# 构建F矩阵
# 该矩阵包含每个MSU对应的轨迹矩阵
import json
from video_split import main  
from extract_traj import process_traj

def extract_cls_cx_cy_matrix(frame_data, frame_range):
    all_ids = set()
    cls_dict = {}  # 单独存储每个id的cls值
    trajectory_dict = {}  # 存储每个id的轨迹信息
    
    # 首先收集所有id和它们的cls值
    for frame_id in range(frame_range[0], frame_range[1] + 1):
        frame_key = str(frame_id)
        if frame_key not in frame_data:
            continue
            
        for detection in frame_data[frame_key]:
            obj_id = detection['id']
            all_ids.add(obj_id)
            if obj_id not in cls_dict:  # 只记录第一次出现的cls
                cls_dict[obj_id] = detection['cls']
    
    # 初始化轨迹字典
    all_ids = sorted(all_ids)
    for obj_id in all_ids:
        trajectory_dict[obj_id] = [cls_dict[obj_id]]  # 第一个元素是cls
    
    # 收集每个帧的cx, cy数据
    for frame_id in range(frame_range[0], frame_range[1] + 1):
        frame_key = str(frame_id)
        frame_detections = {}
        
        if frame_key in frame_data:
            for detection in frame_data[frame_key]:
                frame_detections[detection['id']] = detection
        
        for obj_id in all_ids:
            if obj_id in frame_detections:
                detection = frame_detections[obj_id]
                cx = round(detection['cx'], 2)
                cy = round(detection['cy'], 2)
                trajectory_dict[obj_id].extend([cx, cy])
            else:
                trajectory_dict[obj_id].extend([-1, -1])  # 缺失数据用-1填充

    return all_ids, trajectory_dict

def F_matrix(frame_data):
    info = {}
    msu_list, durations = main()  # 假设 main() 返回一个包含多个帧区间的列表

    for (s, e) in msu_list:
        all_ids, cls_cx_cy_matrix = extract_cls_cx_cy_matrix(frame_data, (s, e))
        info[tuple(all_ids)] = cls_cx_cy_matrix  # 使用元组作为键

    # 提取每个MSU的轨迹数据，去掉id信息，但保留cls信息
    trajectories = []  # 存储每个MSU的轨迹数据
    for msu_ids in info:
        msu_data = info[msu_ids]  # 获取当前MSU的轨迹信息
        msu_trajectories = []  # 存储当前MSU的所有轨迹
        for obj_id in msu_data:
            trajectory = msu_data[obj_id]  # 获取当前对象的轨迹
            msu_trajectories.append(trajectory)  # 直接添加整个轨迹，包括cls
        trajectories.append(msu_trajectories)  # 添加到总轨迹列表中

    # 构建F矩阵
    F = []
    for msu_trajectories in trajectories:
        #F_tmp = process_traj(msu_trajectories)
        #F.append(F_tmp)
        F.append(process_traj(msu_trajectories))

    return F, trajectories
if __name__ == '__main__':
    json_path = r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/frame_info1.json'
    with open(json_path, 'r') as f:
        frame_data = json.load(f)

    F,value = F_matrix(frame_data)

    for j in range(len(value)):
        print("MSU {}: {}".format(j + 1, value[j]))
        print('\n')
    