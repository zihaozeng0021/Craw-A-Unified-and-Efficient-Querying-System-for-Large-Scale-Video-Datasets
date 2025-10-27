#背景差
import cv2
import numpy as np


VIDEO_PATH = "/home/nanchang/ZZQ/yolov13/main/video_Bellevue/Bellevue_116th_NE12th_2.mp4"

def calculate_bg_sub_threshold(baseline_ratio=0.2):

    # 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    baseline_frames = max(10, int(total_frames * baseline_ratio))
    print(f"处理{baseline_ratio*100}%帧: {baseline_frames} 帧")
    
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=baseline_frames,
        detectShadows=False  
    )
    
    residual_values = []
    
    # 形态学操作核（用于去除小面积噪声前景）
    kernel = np.ones((1, 1), np.uint8)
    
    try:
        for frame_idx in range(baseline_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
            
            fg_mask = back_sub.apply(frame_blur)
            
            fg_mask_eroded = cv2.erode(fg_mask, kernel, iterations=1)
            
            bg_model = back_sub.getBackgroundImage()
            if bg_model is None:
                continue  
            
            # 5. 计算当前帧与背景模型的残差
            frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)
            residual = cv2.absdiff(frame_gray, bg_gray)
            
            bg_region = (fg_mask_eroded == 0)  # 仅保留背景区域
            min_residual = 0.25  
            valid_bg_region = bg_region & (residual > min_residual)
            
            # 7. 统计有效背景区域的残差
            valid_residual = residual[valid_bg_region]
            if len(valid_residual) > 0:
                residual_values.append(np.mean(valid_residual))
            
            # 打印进度
            if (frame_idx + 1) % 100 == 0:
                print(f"已处理 {frame_idx + 1}/{baseline_frames} 帧")
                
    finally:
        cap.release()
    
    # 计算均值（仅使用有效残差）
    if not residual_values:
        print("警告：未获取到有效残差数据")
        return 0.0
        
    mean_residual = np.mean(residual_values)
    threshold = mean_residual  # 阈值仍为均值，但均值已提高
    
    print(f"\n背景残差均值: {mean_residual:.2f}")

    
    return threshold

if __name__ == "__main__":
    calculated_threshold = calculate_bg_sub_threshold(baseline_ratio=0.2)
    