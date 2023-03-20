#
import math
import random

import numpy as np
from matplotlib import pyplot as plt

from config import *


def get_data_set():
    matrix = np.loadtxt('Skin_NonSkin.txt')
    return matrix


def get_data_dict(data_set):
    data_dict = {}
    for data in data_set:
        data_dict[(data[0], data[1], data[2])] = {}
    return data_dict


def get_k_center():
    xs = set()
    while len(xs) < uci_k_num:
        xs.add((2 * random.randint(-15, 15), 2 * random.randint(-15, 15), 2 * random.randint(-15, 15)))
    k_num_list = []
    for x in xs:
        k_num_list.append(x)
    return k_num_list


def get_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(point1[2] - point2[2], 2))


def mean(points_list):
    x_mean = 0
    y_mean = 0
    z_mean = 0
    sum = 0
    for point in points_list:
        x_mean += point[0]
        y_mean += point[1]
        z_mean += point[2]
        sum += 1
    if sum == 0:
        return 0, 0, 0
    return x_mean / sum, y_mean / sum, z_mean / sum


def main():
    data_set = get_data_set()
    data_dict = get_data_dict(data_set)
    k_num_list = get_k_center()
    k_points_dict = {}
    for i in range(times):
        for key, value in data_dict.items():
            for k_seg in range(uci_k_num):
                data_dict[key][k_seg] = get_length(key, k_num_list[k_seg])
        for k_seg in range(uci_k_num):
            k_points_dict[k_seg] = []
            for key, value in data_dict.items():
                minflag = True
                min_value = value[k_seg]
                for i in range(uci_k_num):
                    if min_value > value[i]:
                        minflag = False
                if minflag:
                    k_points_dict[k_seg].append(key)
            k_num_list[k_seg] = mean(k_points_dict[k_seg])
    ax3d = plt.gca(projection="3d")
    for k_seg in range(uci_k_num):
        x = [k_points_dict[k_seg][i][0] for i in range(len(k_points_dict[k_seg]))]
        y = [k_points_dict[k_seg][i][1] for i in range(len(k_points_dict[k_seg]))]
        z = [k_points_dict[k_seg][i][2] for i in range(len(k_points_dict[k_seg]))]
        ax3d.scatter(x, y, z)
    plt.show()


# main()
data_set = get_data_set()
data_list = {1: [], 2: []}
for data in data_set:
    data_list[data[3]].append((data[0], data[1], data[2]))
ax3d = plt.gca(projection="3d")
for k_seg in range(1, 3):
    x = [data_list[k_seg][i][0] for i in range(len(data_list[k_seg]))]
    y = [data_list[k_seg][i][1] for i in range(len(data_list[k_seg]))]
    z = [data_list[k_seg][i][2] for i in range(len(data_list[k_seg]))]
    ax3d.scatter(x, y, z)
plt.show()
