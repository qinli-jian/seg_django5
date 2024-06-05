# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: cal_break_w_l
Description:
Author: qinlinjian
Datetime: 4/29/2024 12:35 AM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import os

import cv2
import numpy as np
import pandas as pd
from seg_app.utils.crack_analysis.measurement_methods import crack_analysis
# from crack_analysis.measurement_methods import crack_analysis
from skimage import io

def read_matrix_from_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 将所有非零值转换为1，0值保持不变
    matrix = np.where(df != 0, 1, 0)
    return matrix

from skimage.morphology import skeletonize
'''
传入逻辑矩阵、骨架图保存的路径
返回总长度
'''
def save_break_tal_skeletonimg(bin_matrix,save_path):
    # 需要传入二值矩阵，即01逻辑矩阵
    skeleton = skeletonize(bin_matrix)
    io.imsave(save_path,skeleton.astype(np.uint8) * 255)


from skimage.measure import label
import statistics
'''
# 计算每个病害的每个对象的长度
返回最大,最小,平均,中位数,总和
'''
def break_lengths(bin_matrix):
    lengths = []
    skeletons = []
    labeled_image = label(bin_matrix)
    for i in range(1, labeled_image.max() + 1):
        component = (labeled_image == i)
        skeleton = skeletonize(component)
        skeletons.append(skeleton)
        lengths.append(np.sum(skeleton))

    min_len = min(lengths)
    max_len = max(lengths)
    aver_len = sum(lengths)/len(lengths)
    median_len = statistics.median(lengths)
    tal_len = sum(lengths)

    return min_len,max_len,aver_len,median_len,tal_len

def get_path_id(path):
    path = path.split('\\')[-1]
    file_name = path.split('/')[-1]
    pathid = file_name.split('.')[0]
    return pathid


def break_l_w_analysis(mask_csvpath,save_dir):
    bin_matrix = read_matrix_from_csv(mask_csvpath)

    # 获取csv文件的id，不要后缀
    file_id = get_path_id(mask_csvpath)

    # 钢筋暴露/裂缝病害宽度分析
    matrix = bin_matrix * 255.0
    aver_width, min_width, max_width, median_width, distance_img,pt_arr = crack_analysis(matrix)

    pt_arr = [[arr.astype(int).tolist() for arr in tup] for tup in pt_arr]
    distance_save_path = os.path.join(save_dir,file_id+"_distance.jpg")
    cv2.imwrite(
        distance_save_path,
        distance_img)
    print("平均宽度：", aver_width)
    print("最小宽度：", min_width)
    print("最大宽度：", max_width)
    print("中位数：", median_width)


    # 钢筋暴露/裂缝病害长度分析
    skeleton_save_path = os.path.join(save_dir, file_id + "_skeleton.jpg")
    save_break_tal_skeletonimg(bin_matrix,
                               skeleton_save_path)
    min_len, max_len, aver_len, median_len, tal_len = break_lengths(bin_matrix)
    print("最小长度：", min_len)
    print("最大长度：", max_len)
    print("平均长度：", aver_len)
    print("长度中位数：", median_len)
    print("总长度：", tal_len)
    measurements = {
        "pixel_min_len": int(min_len),
        "pixel_max_len": int(max_len),
        "pixel_aver_len": float(aver_len),
        "pixel_median_len": int(median_len),
        "pixel_tal_len": int(tal_len),
        "pixel_aver_width": float(aver_width),
        "pixel_min_width": int(min_width),
        "pixel_max_width": int(max_width),
        "pixel_median_width": int(median_width),
        "distance_img": file_id+"_distance.jpg",
        "skeleton_img": file_id + "_skeleton.jpg",
        "width_points_array": pt_arr
    }
    print(pt_arr)
    return measurements

from scipy.spatial.distance import euclidean
import networkx as nx
def get_skeleton_pixelpaths(mask_csvpath):

    bin_matrix = read_matrix_from_csv(mask_csvpath)

    labeled_image = label(bin_matrix)
    longest_paths = []

    for i in range(1, labeled_image.max() + 1):
        component = (labeled_image == i)
        skeleton = skeletonize(component)
        skeleton_points = np.column_stack(np.nonzero(skeleton))

        G = nx.Graph()
        for index, p in enumerate(skeleton_points):
            G.add_node(index, pos=p)

        for index1, p1 in enumerate(skeleton_points):
            for index2, p2 in enumerate(skeleton_points):
                distance = euclidean(p1, p2)
                if index1 != index2 and (distance == 1.0 or distance == np.sqrt(2)):
                    G.add_edge(index1, index2)

        endpoints = [node for node, degree in G.degree() if degree == 1]

        if len(endpoints) < 2:
            continue

        longest_path = []
        max_length = 0

        for j in range(len(endpoints)):
            for k in range(j + 1, len(endpoints)):
                u, v = endpoints[j], endpoints[k]
                if nx.has_path(G, u, v):
                    path = nx.shortest_path(G, u, v)
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path

        longest_paths.append([G.nodes[index]["pos"] for index in longest_path])

    return longest_paths

def get_real_width_mm(mask_csvpath,save_dir):
    bin_matrix = read_matrix_from_csv(mask_csvpath)
    # 获取csv文件的id，不要后缀
    file_id = get_path_id(mask_csvpath)


    skeleton_save_path = os.path.join(save_dir, file_id + "_skeleton.jpg")
    save_break_tal_skeletonimg(bin_matrix,
                               skeleton_save_path)

    # 钢筋暴露/裂缝病害宽度分析
    matrix = bin_matrix * 255.0
    aver_width, min_width, max_width, median_width, distance_img, pt_arr = crack_analysis(matrix)

    distance_save_path = os.path.join(save_dir, file_id + "_distance.jpg")
    cv2.imwrite(
        distance_save_path,
        distance_img)

    return pt_arr,file_id + "_skeleton.jpg",file_id + "_distance.jpg"

# 检查是否存在重复路径
def has_duplicates(paths):
    seen = set()
    for path in paths:
        path_tuple = tuple(map(tuple, path))  # 转换为不可变的形式
        if path_tuple in seen:
            return True
        seen.add(path_tuple)
    return False
if __name__ == '__main__':
    file_path = r'E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output\1783149870589284352_4.csv'
    save_dir = r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output"
    # break_l_w_analysis(file_path,save_dir)

    longest_paths = get_skeleton_pixelpaths(file_path)
    print("所有的路：",longest_paths)
    print("所有的路：",len(longest_paths))

    # 验证路径
    if has_duplicates(longest_paths):
        print("There are duplicate paths.")
    else:
        print("No duplicate paths found.")
    for p in longest_paths:
        print(len(p))
    # bin_matrix = read_matrix_from_csv(file_path)
    #
    # # 钢筋暴露/裂缝病害宽度分析
    # matrix = bin_matrix*255.0
    # aver_width,min_width,max_width,median_width,distance_img = crack_analysis(matrix)
    # cv2.imwrite(r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output\1783150706212081664_1_distance.jpg", distance_img)
    # print("平均宽度：", aver_width)
    # print("最小宽度：", min_width)
    # print("最大宽度：", max_width)
    # print("中位数：",median_width)
    #
    # # 钢筋暴露/裂缝病害长度分析
    # save_break_tal_skeletonimg(bin_matrix,r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output\1783150706212081664_1.jpg")
    # min_len,max_len,aver_len,median_len,tal_len = break_lengths(bin_matrix)
    # print("最小长度：", min_len)
    # print("最大长度：", max_len)
    # print("平均长度：", aver_len)
    # print("长度中位数：", median_len)
    # print("总长度：", tal_len)

    # plt.imshow(matrix, cmap='Greys', interpolation='none')
    # plt.colorbar()  # 可以添加颜色条来指示True和False
    # plt.show()


    pass
