import numpy as np
from matplotlib.pyplot import *
from config import *

# 设置字形
rc('font', family='SimHei')
rc('axes', unicode_minus=False)
z
# 读取数据
data = np.loadtxt(fname="data.csv", dtype=float, delimiter=",")
x = data[1, :]
y = data[0, :]

# 计算无正则项解析解
polyFeat_x = np.array([x ** i for i in range(dimension + 1)])
theta1 = np.linalg.pinv(polyFeat_x @ polyFeat_x.T) @ polyFeat_x @ y
# 画图
a = np.linspace(left, right, 2000)
polyFeat_a = np.array([a ** i for i in range(dimension + 1)])
b1 = theta1 @ polyFeat_a
subplot(1, 2, 1)
scatter(x, y)
plot(a, np.sin(a))
plot(a, b1)
legend(['data', 'sin(x)', 'f(x)'])
xlabel('$x$')
ylabel('$y$')
title('解析法无正则项loss最优解')
# 计算有正则项解析解
theta2 = np.linalg.pinv(
    np.eye(polyFeat_x.shape[0], dtype=float) * lambda_analytical + polyFeat_x @ polyFeat_x.T) @ polyFeat_x @ y
b2 = theta2 @ polyFeat_a
# 画图
subplot(1, 2, 2)
scatter(x, y)
plot(a, np.sin(a))
plot(a, b2)
xlabel('$x$')
ylabel('$y$')
legend(['data', 'sin(x)', 'f(x)'])
title('解析法有正则项loss最优解，λ={}'.format(lambda_analytical))
show()
