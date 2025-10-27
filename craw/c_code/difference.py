#帧间差分阈值
import cv2
import numpy as np


VIDEO_PATH = "/home/nanchang/ZZQ/yolov13/main/video_youtobe/youtobe.mp4"

def calculate_exact_average_diff():
    # 打开视频文件（与原代码处理方式一致）
    cap = cv2.VideoCapture(VIDEO_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    process_frames = int(total_frames * 0.2)  
    print(f"视频总帧数: {total_frames}")
    print(f"处理前20%帧: {process_frames} 帧")
    print(f"视频分辨率: {width}x{height}")
    
    total_diff = 0  
    diff_count = 0   
    prev_frame = None  
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取结束
            
            # 获取当前帧号（与原代码计算方式完全一致）
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            # 帧预处理（与原代码完全相同的步骤和参数）
            # 1. 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. 高斯模糊（5x5核，sigma=0，与原代码参数完全一致）
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 计算帧间差异（与原代码计算逻辑完全一致）
            if prev_frame is not None:
                # 计算绝对差异并求和（与原代码diff_val计算方式完全相同）
                diff_val = int(np.sum(cv2.absdiff(gray, prev_frame)))
                total_diff += diff_val
                diff_count += 1
                
                # 打印进度
                if diff_count % 10 == 0:
                    print(f"已处理 {diff_count} 对帧")
            
            # 更新前一帧
            prev_frame = gray
            
            # 达到前20%帧范围后停止处理
            if current_frame >= process_frames - 1:
                break
                
    finally:
        cap.release()  # 确保资源释放，与原代码一致
    
    # 计算平均差异（以帧为单位，总差异/帧对数量）
    if diff_count == 0:
        print("无法计算差异（帧数不足）")
        return 0.0
    
    average_diff = total_diff / diff_count
    
    print(f"\n前20%帧的平均帧间差异: {average_diff:.2f}")
    print(f"总差异值: {total_diff}")
    print(f"计算帧对数量: {diff_count}")
    
    return average_diff

if __name__ == "__main__":
    calculate_exact_average_diff()
    