# I交互矩阵的相似度计算
import numpy as np
def matrix_analysis(IA,IB):
    V1 = count_matrix(IA)
    V2 = count_matrix(IB)
    sum1 = sum(V1)
    sum2 = sum(V2)
    v1 = np.round(V1 / sum1,2) 
    v2 = np.round(V2 / sum2,2)
    result = np.round(compute_manhattan(v1, v2), 3)
    return result

def count_matrix(I):
    V = np.zeros((5,),dtype= int)
    rows, cols = np.triu_indices(I.shape[0])
    for r, c in zip(rows, cols):
        temp = I[r][c]
        if temp == 0:
            V[0] += 1
        elif temp == 1:
            V[1] += 1
        elif temp == 2:
            V[2] += 1
        elif temp == 3:
            V[3] += 1
        elif temp == 4:
            V[4] += 1
    return V

def compute_manhattan(V1,V2):
    return np.sum(np.abs(V1-V2))

if __name__ == "__main__":
    IA = np.array([[0,2,3],[2,0,1],[3,1,0]])
    IB = np.array([[0,4],[4,0]])
    result = matrix_analysis(IA,IB)
    print("结果:", result)
