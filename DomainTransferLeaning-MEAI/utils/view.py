# coding=utf-8

from sklearn.decomposition import pca
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


# 降维
def view_on_two_dim(points=[], proto_points=[]):
    # list: points, proto_points
    # points = points.reshape(-1, 768).tolist()
    # proto_points = proto_points.reshape(-1, 768).tolist()
    matrix = []
    if len(points) > 0:
        matrix.extend(points)
    if len(proto_points) > 0:
        matrix.extend(proto_points)

    K = 2
    model = pca.PCA(n_components=K).fit(matrix)
    points_d = model.transform(points)
    proto_points_d = model.transform(proto_points)

    return points_d, proto_points_d


# 画图
def scalar_point(points_d=[], proto_points_d=[], pic_name=None, radius=None, rels=None, is_circle=False):
    # colors = ["mediumseagreen", "blueviolet"]
    fig = plt.figure()
    ax = plt.subplot()
    plt.axis([-3, 4, -3, 3])
    for i, mat in enumerate(points_d):
        xs = []
        ys = []
        for (x, y) in mat:
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, alpha=0.9, marker="o", label=rels[i])

    if len(proto_points_d) > 0:
        for i in range(len(proto_points_d)):
            ax.scatter([proto_points_d[i][0]], [proto_points_d[i][1]], alpha=0.9, color="black")

    if is_circle:
        # 画圆
        for i in range(len(proto_points_d)):
            # circle
            theta = np.arange(0, 2*np.pi, 0.01)
            x = proto_points_d[i][0] + radius[i] * np.cos(theta)
            y = proto_points_d[i][1] + radius[i] * np.sin(theta)
        #
            plt.plot(x, y, linestyle="--")#, color=colors[i])

    plt.legend(loc="upper right")

    plt.savefig(pic_name)


def cal_mean(mat, proto):
    sum = 0
    for v in mat:
        sum += np.sum(np.square(proto - v))
    mean = sum / len(mat)
    print("mean:{}".format(mean))
    return mean


def calc_radius(point_d, proto_points_d, scale):
    # 这是所用的类别一起做吗？
    R = []
    mean_R = []
    for i, mat in enumerate(point_d):
        proto = proto_points_d[i]
        mean = cal_mean(mat, proto)
        R.append(np.abs(scale[i]) * mean)
        mean_R.append(mean)
    print("R:{}".format(R))
    return R, mean_R

