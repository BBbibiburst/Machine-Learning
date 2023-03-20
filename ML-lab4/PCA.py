import numpy as np
import matplotlib.pyplot as plt
from config import *


# 获取数据
def get_data():
    matrix = np.loadtxt(data_file)
    return matrix


# 零均值化
def all_zero_mean(data):
    # 计算均值
    mean = np.array([np.mean(data[:, index]) for index in range(data.shape[1])])
    data = data - mean
    return mean, data


# 计算协方差矩阵
def get_cov_matrix(data):
    return data.T @ data / data.shape[0]


# PCA实现，返回降维后的数据，特征向量基和降维后重构的数据
def PCA(data, dimension_to_cut):
    # 零均值化
    all_zero_mean(data)
    # 计算协方差矩阵
    Cov = get_cov_matrix(data)
    # svd计算特征值和特征向量
    u, s, vT = np.linalg.svd(Cov)
    # 取前k个向量
    vT = vT[:][:dimension_to_cut]
    # 压缩数据
    data = data @ vT.T
    # 返回降维后的数据，特征向量基和降维后重构的数据
    return data, vT, data @ vT


def show2D(data):
    # 平面图
    x = [data[i][0] for i in range(data.shape[0])]
    y = [data[i][1] for i in range(data.shape[0])]
    plt.scatter(x, y)
    plt.legend(['data'])
    ax = plt.subplot()
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    plt.show()


def show3D(data, fvs, data_o):
    # 散点图
    ax3d = plt.subplot(projection="3d")
    x = [data[i][0] for i in range(data.shape[0])]
    y = [data[i][1] for i in range(data.shape[0])]
    z = [data[i][2] for i in range(data.shape[0])]
    ax3d.scatter(x, y, z)
    x = [data_o[i][0] for i in range(data_o.shape[0])]
    y = [data_o[i][1] for i in range(data_o.shape[0])]
    z = [data_o[i][2] for i in range(data_o.shape[0])]
    ax3d.scatter(x, y, z)
    ax3d.set_xlabel("X Axis")
    ax3d.set_ylabel("Y Axis")
    ax3d.set_zlabel("Z Axis")
    # 直线
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-5, 5, 20)
    colors = ['r', 'g', 'b']
    for i in range(fvs.shape[0]):
        z = -(fvs[i][0] / fvs[i][2]) * x + -(fvs[i][1] / fvs[i][2]) * y
        ax3d.plot(x, y, z, colors[i])
    plt.legend(['data', 'pca_data', 'feature0', 'feature1'])
    plt.show()


def main():
    data_set = get_data()
    print('原数据为')
    print(data_set)
    pca_data_set, fvs, data_o = PCA(data_set, dimension)
    print('经过降维后')
    print(pca_data_set)
    print('特征向量为')
    print(fvs)
    show3D(data_set, fvs, data_o)
    show2D(pca_data_set)


if __name__ == '__main__':
    main()
