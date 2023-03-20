#
import math
import random

import numpy as np
from matplotlib import pyplot as plt

from config import *


def get_data_set():
    matrix = np.loadtxt(data_file)
    return matrix


def get_data_dict(data_set):
    data_dict = {}
    for data in data_set:
        data_dict[(data[0], data[1])] = {}
    return data_dict


def get_k_center():
    xs = set()
    while len(xs) < k_num:
        xs.add((2 * random.randint(-15, 15), 2 * random.randint(-15, 15)))
    k_num_list = []
    for x in xs:
        k_num_list.append(x)
    return k_num_list


def get_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def mean(points_list):
    x_mean = 0
    y_mean = 0
    sum = 0
    for point in points_list:
        x_mean += point[0]
        y_mean += point[1]
        sum += 1
    if sum == 0:
        return (0, 0)
    return (x_mean / sum, y_mean / sum)


data_set = get_data_set()
data_dict = get_data_dict(data_set)
k_num_list = [(16, 20), (28, 20), (-20, -4)]#get_k_center()
print(k_num_list)
k_points_dict = {}
for i in range(times):
    for key, value in data_dict.items():
        for k_seg in range(k_num):
            data_dict[key][k_seg] = get_length(key, k_num_list[k_seg])
    for k_seg in range(k_num):
        k_points_dict[k_seg] = []
        for key, value in data_dict.items():
            minflag = True
            min_value = value[k_seg]
            for i in range(k_num):
                if min_value > value[i]:
                    minflag = False
            if minflag:
                k_points_dict[k_seg].append(key)
        k_num_list[k_seg] = mean(k_points_dict[k_seg])

for k_seg in range(k_num):
    x = [k_points_dict[k_seg][i][0] for i in range(len(k_points_dict[k_seg]))]
    y = [k_points_dict[k_seg][i][1] for i in range(len(k_points_dict[k_seg]))]
    plt.scatter(x, y)
#plt.scatter(2.29354,1.55941,marker='*',s=300)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(['1','2','3','EERIE'])
plt.show()
