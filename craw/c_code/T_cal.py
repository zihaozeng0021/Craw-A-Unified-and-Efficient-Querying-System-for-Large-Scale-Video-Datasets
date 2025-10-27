# T时序矩阵的相似度计算
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_ctmd(TA, TB):
    import time
    cA, _ = TA.shape  
    cB, _ = TB.shape  
    c_max = max(cA, cB)  
    start_time = time.perf_counter()
    # 填充矩阵
    def pad_matrix(T, target_rows):
        padded = np.zeros((target_rows, 2))  
        padded[:T.shape[0], :] = T  
        if T.shape[0] < target_rows:
            padded[T.shape[0]:, 0] = 0  
            padded[T.shape[0]:, 1] = 1  
        return padded


    TA_padded = pad_matrix(TA, c_max)
    TB_padded = pad_matrix(TB, c_max)

    # 提取填充后的第一列和第二列，分别作为两个分布
    Du1 = TA_padded[:, 0]  
    Dv1 = TA_padded[:, 1]  
    Du2 = TB_padded[:, 0]  
    Dv2 = TB_padded[:, 1]  

    # 构建成本矩阵
    def construct_cost_matrix(Du, Dv):
        return np.abs(Du[:, None] - Dv[None, :])  # 使用绝对差值作为成本


    C1 = construct_cost_matrix(Du1, Du2) 
    C2 = construct_cost_matrix(Dv1, Dv2)  

    # 计算EMD
    def compute_emd(cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  
        return cost_matrix[row_ind, col_ind].sum()  

    emd_u = compute_emd(C1) 
    emd_v = compute_emd(C2)  

    L = 1 / c_max  # 重合分布的权重

    # 计算CTMD
    ctmd = L * (emd_u + emd_v) / 2  
    end_time = time.perf_counter()
    time = end_time - start_time
    return ctmd, time 

if __name__ == "__main__":
    TA = np.array([[0.28,0.82],[0.13,0.65]])
    TB = np.array([[0.25,0.34],[0.66,0.89],[0.45,0.78]])
    ctmd_value,time = compute_ctmd(TA, TB)
    print(f"CTMD计算时间: {time:.6f} seconds")
    print("两个视频片段之间的CTMD距离为:", ctmd_value)
