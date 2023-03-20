import numpy as np
import matplotlib.pyplot as plt
from config import *


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 读取x，y
def getXY():
    xy = np.loadtxt(data_file)
    x = xy[:2]
    y = xy[2]
    x = np.concatenate([x, np.array([[1 for i in range(2 * nums)]])], axis=0)
    return x.T, y.T


# 计算loss函数
def getLoss(x, y, w):
    loss = 0
    for i in range(x.shape[1]):
        wx = w @ x[i]
        loss_single = y[i] * np.log(sigmoid(wx)) + (1 - y[i]) * np.log(1 - sigmoid(wx))
        loss -= loss_single
    return loss


# 梯度下降计算
def getW():
    x, y = getXY()
    # 随机生成初始矩阵
    w = np.array([1.0 for i in range(3)])
    # 计算初始loss
    old_loss = loss = getLoss(x, y, w)
    times = 0  # 迭代次数
    # 正则化参数
    lambda_analytical = 0.1
    delta = 1
    # 梯度下降主循环
    while True:
        delta_w = np.array([0.0 for i in range(3)])
        for i in range(x.shape[0]):
            wx = w @ x[i]
            d = x[i] * (y[i] - sigmoid(wx))
            delta_w += d
        w += (learning_rate * delta_w / x.shape[1] - lambda_analytical * w) * delta
        loss = getLoss(x, y, w)
        times += 1
        # 递减学习率
        if times % 1000 == 0:
            delta *= 0.96
        print('epoch {}: Loss = {}'.format(times, loss))
        if abs(old_loss - loss) < 10 ** -8:
            break
        old_loss = loss
    print('Training completed.\nFinal Loss = {}\nw = {}'.format(loss, w))
    return w


# 计算准确率
def getAccuracy(w, x, y):
    total = y.shape[0]
    correct = 0
    for i in range(x.shape[0]):
        if w @ x[i] < 0:
            correct += 1 - y[i]
        else:
            correct += y[i]
    return correct / total


# 画图
w = getW()
xx, yy = getXY()
print(getAccuracy(w, xx, yy))
points = getXY()
points_xs, _ = getXY()
points_xs = points_xs[:nums]
x = [points_xs[i][0] for i in range(nums)]
y = [points_xs[i][1] for i in range(nums)]
points_as, _ = getXY()
points_as = points_as[nums:]
a = [points_as[i][0] for i in range(nums)]
b = [points_as[i][1] for i in range(nums)]
line_x = np.linspace(-2, 5, 300)
line_y = (-w[2] - w[0] * line_x) / w[1]
plt.plot(line_x, line_y)
plt.scatter(x, y)
plt.scatter(a, b)
plt.legend(['Discriminant function', '0', '1'])
plt.show()
