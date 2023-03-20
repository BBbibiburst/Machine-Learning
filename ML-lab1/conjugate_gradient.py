import numpy as np
from matplotlib.pyplot import *
from config import *

# 设置字形
rc('font', family='SimHei')
rc('axes', unicode_minus=False)
title('n = {}'.format(dimension))

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
lambda_analytical = 0.01
# 初始化共轭矩阵计算
A = polyFeat_x @ polyFeat_x.T + lambda_analytical * np.eye(polyFeat_x.shape[0])
b = polyFeat_x @ y
r_0 = b - theta @ A
p = r_0
# 共轭矩阵迭代循环
while True:
    alpha = (r_0.T @ r_0) / (p.T @ A @ p)
    theta = theta + p * alpha
    r = r_0 - alpha * A @ p
    if r_0 @ r_0 < 10 ** -8:
        break
    beta = (r.T @ r) / (r_0.T @ r_0)
    p = r + beta * p
    r_0 = r
    loss = getLoss(y, theta, polyFeat_x)
    times += 1
    print('Turn {} Loss = {}'.format(times, loss))

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
title('共轭梯度下降loss最优解')
show()
