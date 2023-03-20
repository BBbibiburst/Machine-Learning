import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import *
from PCA import all_zero_mean, get_cov_matrix


# PCA实现，返回降维后重构的数据，特征向量基
def PCA(data, dimension_to_cut):
    # 零均值化
    mean, data = all_zero_mean(data)
    # 计算协方差矩阵
    Cov = get_cov_matrix(data)
    # svd计算特征值和特征向量
    u, s, vT = np.linalg.svd(Cov)
    # 取前k个向量
    vT = vT[:][:dimension_to_cut]
    # 压缩数据
    data = data @ vT.T
    # 还原数据
    data = data @ vT
    data += mean
    # 返回降维后重构的数据，特征向量基
    return data, vT


# 从图像读取人脸数据集
def get_faces():
    img = cv2.imread(faces_file, cv2.IMREAD_GRAYSCALE)
    face_list = []
    for i in range(col):
        for j in range(col):
            picture = img[i * single_pic_size:i * single_pic_size + single_pic_size, j * single_pic_size:j * single_pic_size + single_pic_size]
            picture = picture.reshape(single_pic_size*single_pic_size, )
            face_list.append(picture)
    return np.array(face_list)


# 将人脸数据集恢复为图像
def go_back(pca_faces):
    result = []
    for face in pca_faces:
        face = face.reshape(single_pic_size,single_pic_size)
        result.append(face)
    picture = np.array([0.0 for i in range(single_pic_size*col*single_pic_size*col)])
    picture = picture.reshape(single_pic_size*col,single_pic_size*col)
    for i in range(col):
        for j in range(col):
            picture[i * single_pic_size:i * single_pic_size + single_pic_size, j * single_pic_size:j * single_pic_size + single_pic_size]+=result[col*i+j]
    return picture


def get_pic(i):
    faces = get_faces()
    pca_faces = PCA(faces, i)[0]
    solved_faces = go_back(pca_faces)
    solved_faces = solved_faces.astype('uint8')
    return solved_faces


# 主程序
def main():
    faces = get_faces()
    pca_faces = PCA(faces, face_dimension)[0]
    solved_faces = go_back(pca_faces)
    solved_faces = solved_faces.astype('uint8')
    cv2.imshow('img.jpg', solved_faces)
    cv2.waitKey(0)
    cv2.imwrite('faces/img{}.jpg'.format(face_dimension), solved_faces)


if __name__ == '__main__':
    main()
