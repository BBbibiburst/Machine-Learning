import numpy as np
import matplotlib.pyplot as plt
from config import *


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 读取x，y
def getXY():
    x = [[], [], [], []]
    y = []
    with open(uci_file, 'r') as uci:
        for line in uci:
            if line == '\n':
                continue
            line_split = line.rstrip().split('\t')
            for i in range(3):
                x[i].append(float(line_split[i]))
            x[3].append(1)
            y.append(float(line_split[3]) - 1)
    x = np.array(x)
    for i in range(3):
        mean = x[i].mean()
        for j in range(len(x[i])):
            x[i][j] /= mean
    y = np.array(y)
    return x.T, y.T


# 计算loss函数
def getLoss(x, y, w):
    loss = 0
    for i in range(x.shape[0]):
        wx = w @ x[i]
        loss_single = y[i] * np.log(sigmoid(wx)) + (1 - y[i]) * np.log(1 - sigmoid(wx))
        loss -= loss_single
    loss /= x.shape[0]
    return loss


# 梯度下降计算
def getW():
    x, y = getXY()
    # 随机生成初始矩阵
    w = np.array([1.0 for i in range(4)])
    # 计算初始loss
    old_loss = loss = getLoss(x, y, w)
    times = 0  # 迭代次数
    # 正则化参数
    lambda_analytical = 0
    # 梯度下降主循环
    while True:
        delta_w = np.array([0.0 for i in range(4)])
        for i in range(x.shape[0]):
            wx = w @ x[i]
            d = x[i] * (y[i] - sigmoid(wx))
            delta_w += d
        delta_w /= np.linalg.norm(delta_w)
        w += learning_rate * delta_w / x.shape[1] - lambda_analytical * w
        loss = getLoss(x, y, w)
        times += 1
        print('epoch {}: Loss = {}'.format(times, loss))
        if abs(old_loss - loss) < 10 ** -7:
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
x, y = getXY()
xs = [[], [], [], []]
ys = [[], [], [], []]
for i in range(y.shape[0]):
    if y[i] == 0:
        for j in range(4):
            xs[j].append(x[i][j])
    else:
        for j in range(4):
            ys[j].append(x[i][j])
ax = plt.axes(projection='3d')
ax.view_init(10, -70)
x = np.linspace(0, 2, 9)
y = np.linspace(0, 2, 9)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, Z=-(w[0] * X + w[1] * Y + w[3]) / w[2], color='g', alpha=0.6)
ax.scatter3D(xs[0], xs[1], xs[2], 'gray')
ax.scatter3D(ys[0], ys[1], ys[2], 'gray')
plt.show()
