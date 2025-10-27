
import cv2

img = cv2.imread(r'/home/nanchang/ZZQ/yolov13/main/xxx/frame_0001.png')

x1, y1 = 80, 120
x2, y2 = 120,220

crop = img[y1:y2, x1:x2]   
cv2.imwrite('crop.png', crop)

'''
import cv2
import os

src_file = r'/home/nanchang/ZZQ/yolov13/main/video_Bellevue/Bellevue_116th_NE12th_2.mp4'  # 原视频
dst_file = r'/home/nanchang/ZZQ/yolov13/main/video_Bellevue/Bellevue_116th_NE12th_2_test.mp4'  # 输出文件

# 1. 裁剪区域（与你给图片用的一致）
x1, y1 = 0, 140
x2, y2 = 1280, 600
W = x2 - x1  
H = y2 - y1  

cap = cv2.VideoCapture(src_file)
if not cap.isOpened():
    raise IOError('打不开原视频')

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(dst_file, fourcc, fps, (W, H))

count = 0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[y1:y2, x1:x2]
    out.write(crop)

cap.release()
out.release()
print(f'完成！裁剪后视频已保存为 {dst_file}')

'''