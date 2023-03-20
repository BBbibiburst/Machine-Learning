import cv2
import matplotlib.pyplot as plt
from numpy import log10
from faces_pca import get_pic

img_ori = r'faces/newpic.jpg'
img = r'faces/img{}.jpg'

# 设置字形
plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)


def get_SNR(img_ori, img):
    img_ori = cv2.imread(img_ori, cv2.IMREAD_GRAYSCALE)
    c = img - img_ori
    c = c.reshape(c.shape[0] * c.shape[1], )
    var_c = sum(c ** 2)
    img = img.reshape(img.shape[0] * img.shape[1], )
    var_img = sum(img ** 2)
    return 10 * log10(var_img / var_c)


xs = [i for i in range(1, 256,1)]
ys = [get_SNR(img_ori, get_pic(x)) for x in xs]
plt.plot(xs, ys, 'r')
ax = plt.subplot()
ax.set_xlabel("主成分维度")
ax.set_ylabel("信噪比/dB")
plt.title('图像信噪比随降维数变化折线图')
plt.show()
