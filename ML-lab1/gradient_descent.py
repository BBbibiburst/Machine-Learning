import numpy as np
from matplotlib.pyplot import *
from config import *

# 设置字形
rc('font', family='SimHei')
rc('axes', unicode_minus=False)

# 读取数据
data = np.loadtxt(fname="data.csv", dtype=float, delimiter=",")
x = data[1, :]
y = data[0, :]
polyFeat_x = np.array([x ** i for i in range(dimension + 1)])


# 定义loss函数
def getLoss(yo, t, polyFeat):
    return 0.5 * np.linalg.norm(t @ polyFeat - yo)


times = 0  # 迭代次数
# 随机生成初始矩阵
theta = np.array([np.random.random() for i in range(polyFeat_x.shape[0])])
# 计算初始loss
loss = getLoss(y, theta, polyFeat_x)
# 初始化loss变化
delta_loss = 1
# 梯度下降迭代循环
while loss > 1 or delta_loss > 10 ** -8:
    derivative = (theta @ polyFeat_x - y) @ polyFeat_x.T + lambda_analytical * theta
    theta -= derivative * learning_rate / np.linalg.norm(derivative)
    loss_0 = loss
    loss = getLoss(y, theta, polyFeat_x)
    delta_loss = abs(loss_0 - loss)
    print('Turn {} Loss = {}'.format(times, loss))
    times += 1
    if times % 5000 == 0:
        learning_rate *= 0.96

print('Training completed.\nFinal Loss = {}'.format(loss))

# 画图
a = np.linspace(left, right, 2000)
polyFeat_a = np.array([a ** i for i in range(dimension + 1)])
b1 = theta @ polyFeat_a
scatter(x, y)
plot(a, np.sin(a))
plot(a, b1)
legend(['data', 'sin(x)', 'f(x)'])
xlabel('$x$')
ylabel('$y$')
title('梯度下降loss最优解')
show()
