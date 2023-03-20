import numpy as np
import matplotlib.pyplot as plt
from config import *


# 单类点生成器
def generatePoints(center, rand_normal_x, rand_normal_y, rand_normal_z, nums_p):
    # 中心
    x_p = center[0]
    y_p = center[1]
    z_p = center[2]
    # 协方差矩阵
    cov = [[rand_normal_x, naive_rate_xy, naive_rate_xz], [naive_rate_xy, rand_normal_y, naive_rate_yz],
           [naive_rate_xz, naive_rate_yz, rand_normal_z]]
    # x各维生成
    ps = np.random.multivariate_normal((x_p, y_p, z_p), cov, nums_p)
    return ps


ax3d = plt.subplot(projection="3d")
# 生成点并画出点图
matrix = generatePoints([0, 0, 0], rand_normal_x, rand_normal_y, rand_normal_z, nums)
x = [matrix[i][0] for i in range(matrix.shape[0])]
y = [matrix[i][1] for i in range(matrix.shape[0])]
z = [matrix[i][2] for i in range(matrix.shape[0])]
ax3d.scatter(x, y, z)
ax3d.set_xlabel("X Axis")
ax3d.set_ylabel("Y Axis")
ax3d.set_zlabel("Z Axis")
plt.legend(['data'])
plt.show()
np.savetxt(fname=data_file, X=matrix)
