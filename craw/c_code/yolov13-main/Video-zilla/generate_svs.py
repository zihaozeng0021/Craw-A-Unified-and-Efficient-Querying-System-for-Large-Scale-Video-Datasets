import cv2, torch, pickle, json, os, numpy as np
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
import time
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
resnet.eval()          
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

yolo = YOLO(r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/runs/ytb4/weights/best.pt')

def extract_object_features(frame):
    objs = yolo(frame, conf=0.6)
    feats = []
    for o in objs:
        for box in o.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if y2 <= y1 or x2 <= x1:
                continue  
                
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue  
                
            patch = torch.from_numpy(patch).to(device).permute(2,0,1).unsqueeze(0)/255.0
            
            with torch.no_grad():
                feat = resnet(patch).squeeze().cpu().numpy()  
            feats.append(feat)
    return feats

def extract_video_features(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}"); return False
    
    # 存储整个视频的所有物体特征
    all_features = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
        frame = cv2.resize(frame, (960, 540))
        feats = extract_object_features(frame)
        if feats:  # 只添加非空特征
            all_features.extend(feats)
            
        # 打印进度
        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"  已处理 {frame_idx} 帧，累计 {len(all_features)} 个物体特征")
            
        frame_idx += 1
    
    # 保存所有特征
    if all_features:
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, 'wb') as f:
            pickle.dump(np.array(all_features), f)
        print(f"  保存特征到 {out_path}，共 {len(all_features)} 个物体特征")
        cap.release()
        return True
    else:
        print(f"  视频 {video_path} 未提取到任何特征")
        cap.release()
        return False

def batch_extract_features(input_dir, output_dir, video_extensions=['.mp4', '.avi', '.mov', '.mkv']):


    # 获取所有视频文件路径
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
    
    if not video_files:
        print(f"在 {input_dir} 中未找到任何视频文件")
        return 0
    
    print(f"找到 {len(video_files)} 个视频文件，开始处理...")
    total_start_time = time.perf_counter()
    success_count = 0
    
    for i, video_path in enumerate(video_files):
        print(f"\n处理第 {i+1}/{len(video_files)} 个视频: {os.path.basename(video_path)}")
        start_time = time.perf_counter()
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_features.pkl")
        
        if extract_video_features(video_path, output_path):
            success_count += 1
        
        elapsed_time = time.perf_counter() - start_time
        print(f"  处理耗时: {elapsed_time:.2f} 秒")
    
    total_elapsed_time = time.perf_counter() - total_start_time
    print(f"\n所有视频处理完成，总耗时: {total_elapsed_time:.2f} 秒")
    print(f"成功处理 {success_count}/{len(video_files)} 个视频")
    return success_count


if __name__ == "__main__":

    input_video_dir = r'/home/nanchang/ZZQ/yolov13/main/warsaw_msu'
    output_feature_dir = r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin'
    

    batch_extract_features(input_video_dir, output_feature_dir)
    