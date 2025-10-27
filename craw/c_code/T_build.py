# 构建T矩阵
# 该矩阵包含每个MSU对应的时序矩阵
from video_split import main

def T_matrix(msu_list, durations):
    T = []
    for i in range(len(msu_list)):
        T_tmp = []
        s, e = msu_list[i]  # 当前MSU的起始和结束帧
        total = e - s + 1   # 当前MSU的总帧数
        
        # 获取当前MSU中所有目标的持续时间信息
        value_list = list(durations[i].values())
        
        for start_abs, end_abs in value_list:
            # 转换为相对于当前MSU的相对帧号
            start_relative = start_abs - s
            end_relative = end_abs - s
            
            # 归一化到[0,1]区间
            start_norm = start_relative / total
            end_norm = end_relative / total
            
            # 确保值在[0,1]范围内
            start_norm = round(max(0, min(1, start_relative / total)), 2)
            end_norm = round(max(0, min(1, end_relative / total)), 2)
            
        
            T_tmp.append([start_norm, end_norm])

        if T_tmp == []:
            T_tmp = [0,1]
                
            
        T.append(T_tmp)
    return T


if __name__ == '__main__':
    msu_list, durations = main()  
    T = T_matrix(msu_list, durations)
    for i in range(len(T)):
        print("MSU {}: {}".format(i + 1, T[i]))
        print('\n')
    
