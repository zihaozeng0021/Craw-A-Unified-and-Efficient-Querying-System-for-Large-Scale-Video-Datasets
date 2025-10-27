# 构建MSU
from F_build import F_matrix
from T_build import T_matrix
from I_build import I_matrix
from video_split import main
import json

def build_msu():
    msu_list, durations = main()
    json_path = r'/home/nanchang/XW2/yolov13/main/yolov13-main/frame_info1.json'
    with open(json_path, 'r') as f:
        frame_data = json.load(f)
    F,value = F_matrix(frame_data)
    I,inter = I_matrix(value)
    T = T_matrix(msu_list, durations)
    return F,I,T


if __name__ == '__main__':
    MSU = []
    F,I,T = build_msu()
    print(F)
    print(I)
    print(T)


    




