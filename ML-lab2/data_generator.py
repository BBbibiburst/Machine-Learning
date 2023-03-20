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

# 生成点并画出点图
xs = generatePoints([center_x, center_y], rand_normal_x, rand_normal_y, nums, naive_rate_xy)
x = [xs[i][0] for i in range(xs.shape[0])]
y = [xs[i][1] for i in range(xs.shape[0])]
xs = generatePoints([center_a, center_b], rand_normal_a, rand_normal_b, nums, naive_rate_ab)
a = [xs[i][0] for i in range(xs.shape[0])]
b = [xs[i][1] for i in range(xs.shape[0])]
plt.scatter(x, y)
plt.scatter(a, b)
plt.legend(['0', '1'])
plt.show()
# 保存点
matrix = np.append(x, a)
matrix = np.append(matrix, y)
matrix = np.append(matrix, b)
matrix = np.append(matrix, [0 for i in range(len(x))])
matrix = np.append(matrix, [1 for j in range(len(a))])
matrix.resize(3, 2 * nums)
np.savetxt(fname=data_file, X=matrix)
