#光流
import cv2
import numpy as np

VIDEO_PATH = "/home/nanchang/ZZQ/yolov13/main/video_Bellevue/Bellevue_116th_NE12th_2.mp4"

def calculate_optical_flow_mean(baseline_ratio=0.2):
    # 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算需要处理的基线帧数
    baseline_frames = max(10, int(total_frames * baseline_ratio))
    
    # 读取第一帧并检测初始特征点
    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("无法读取视频帧")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # 存储光流幅值（运动强度）
    mag_values = []
    frame_count = 1  # 已处理1帧
    
    try:
        while frame_count < baseline_frames:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                p0,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # 筛选跟踪成功的特征点
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # 计算光流幅值
                dx = good_new[:, 0] - good_old[:, 0]
                dy = good_new[:, 1] - good_old[:, 1]
                mag = np.sqrt(dx**2 + dy**2)
                mag_values.extend(mag)
            
            # 更新特征点和前一帧
            p0 = good_new.reshape(-1, 1, 2) if p1 is not None else p0
            prev_gray = curr_gray.copy()
            frame_count += 1
            
            # 打印进度
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧")
                
    finally:
        cap.release()
    
    # 仅计算并返回均值
    if not mag_values:
        return 0.0
    
    mean_mag = np.mean(mag_values)
    print(f"平均幅值: {mean_mag:.2f}")
    
    return mean_mag

if __name__ == "__main__":
    calculated_mean = calculate_optical_flow_mean(baseline_ratio=0.2)

    