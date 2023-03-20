import numpy as np
from config import *
from matplotlib.pyplot import *

# 生成x
x = np.linspace(left, right, points)
y = np.sin(x)  # 计算y
# 加入高斯噪声
yg = y + np.array([np.random.normal(0, theta_normal) for i in range(0, points)])
data = np.append(yg, x).reshape(2, points)
# 画图
scatter(x, yg)
plot(np.linspace(left, right, 2000), np.sin(np.linspace(left, right, 2000)))
legend(['data', 'sin(x)'])
xlabel('$x$')
ylabel('$y$')
show()
# 保存
np.savetxt(fname="data.csv", X=data, fmt="%f", delimiter=",")
