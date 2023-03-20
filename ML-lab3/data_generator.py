import numpy as np
import matplotlib.pyplot as plt
from config import *


# 单类点生成器
def generatePoints(center, rand_normal_x, rand_normal_y, nums, naive_rate):
    # 中心
    x = center[0]
    y = center[1]
    # 协方差矩阵
    cov = [[rand_normal_x, naive_rate], [naive_rate, rand_normal_y]]
    # x各维生成
    xs = np.random.multivariate_normal((x, y), cov, nums)
    return xs


matrix = None
# 生成点并画出点图
for a in range(-10, 11,20):
    for b in range(-10, 11,20):
        xs = generatePoints([a, b], rand_normal, rand_normal, nums, naive_rate)
        x = [xs[i][0] for i in range(xs.shape[0])]
        y = [xs[i][1] for i in range(xs.shape[0])]
        plt.scatter(x, y)
        if matrix is None:
            matrix = xs
        else:
            matrix = np.append(matrix,xs,axis=0)

plt.show()
np.savetxt(fname=data_file, X=matrix)