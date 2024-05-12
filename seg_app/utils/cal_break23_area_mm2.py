# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: cal_break23_area_mm2
Description:
Author: qinlinjian
Datetime: 5/2/2024 5:39 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from seg_app.utils.AstraCam import AstraCam
from seg_django_5 import settings


def read_matrix_from_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 将所有非零值转换为1，0值保持不变
    matrix = np.where(df != 0, 1, 0)
    return matrix

import statistics
"""
传入逻辑矩阵
计算每个对象区域的面积
返回最大、最小、平均和中位面积
"""
def pixel_area(logic_matrix,pixel_size_mm):
    # 使用 label 函数标记连通区域
    labeled_matrix = label(logic_matrix, connectivity=1)
    # 使用 regionprops 计算每个区域的面积
    properties = regionprops(labeled_matrix)
    for i,p in enumerate(properties):
        properties[i]=pixel_to_real_area_mm2(p,pixel_size_mm)

    min_pixel_area = min(properties)
    max_pixel_area = max(properties)
    aver_pixel_area = sum(properties)/len(properties)
    median_pixel_area = statistics.median(properties)
    return min_pixel_area,max_pixel_area,aver_pixel_area,median_pixel_area

"""
传入像素面积和一个像素边长代表显示多少mm
将像素面积转换成真实面积
返回这个区域的真实面积（单位：平方毫米）
"""
def pixel_to_real_area_mm2(pixel_area,pixel_size_mm):
    return pixel_area * (pixel_size_mm ** 2)

"""
返回：min_pixel_area,max_pixel_area,aver_pixel_area,median_pixel_area
"""
def get_areas(csv_path,pixel_size_mm):
    logic_matri = read_matrix_from_csv(pixel_size_mm)
    min_area, max_area, aver_area, median_area = pixel_area(logic_matri,pixel_size_mm)
    measurements = {}
    measurements["min_area"] = min_area
    measurements["max_area"] = max_area
    measurements["aver_area"] = aver_area
    measurements["median_area"] = median_area
    return measurements

import cv2
# 主要调用计算病害的掩码中多个区域在实际中的真实面积
def cal_astraimg_areas_mm2(csv_path,depth_img_path,img_path,save_rgb_path):
    shape = settings.ASTRA_CAM_SHAPE
    cam = AstraCam(
        depth_img_path=depth_img_path,
        img_path=img_path,
        save_rgb_path=save_rgb_path,
        shape=shape)

    # 读取掩码矩阵
    bin_matrix = read_matrix_from_csv(csv_path)

    # 使用OpenCV检测轮廓
    contours, _ = cv2.findContours(bin_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []

    for vertices in contours:
        epsilon = 0.01 * cv2.arcLength(vertices, True)
        approx_vertices = cv2.approxPolyDP(vertices, epsilon, True)
        approx_vertices = get_real_world_xyz(cam,approx_vertices.reshape(-1, 2)).reshape(-1, 3)
        areas.append(calculate_area(approx_vertices))

    return areas # 返回的是所有多边形的面积


def get_real_world_xyz(cam,approx_vertices):
    real_xyzs = []
    for p in approx_vertices:
        print(p)
        xyz = cam.real_word_xyz_matrix[int(p[1])][int(p[0])] # 传入需要时y,x y表示行
        real_xyzs.append(xyz)
    return np.array(real_xyzs)

from sklearn.decomposition import PCA
def calculate_area(points):
    print("投影平面前:",points)
    if points.shape[1] == 3:  # 如果是三维坐标
        print("三维")
        pca = PCA(n_components=2)
        points = pca.fit_transform(points)

    x = points[:, 0]
    y = points[:, 1]
    print("投影平面:",points)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


# 下面是测试
def cal_astraimg_areas_mm2_test(depth_img_path, img_path, save_rgb_path):
    shape = settings.ASTRA_CAM_SHAPE
    cam = AstraCam(
        depth_img_path=depth_img_path,
        img_path=img_path,
        save_rgb_path=save_rgb_path,
        shape=shape)

    # 读取掩码矩阵
    # bin_matrix = read_matrix_from_csv(csv_path)
    #
    # # 使用OpenCV检测轮廓
    # contours, _ = cv2.findContours(bin_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    """
Clicked at: (193, 270)
Clicked at: (378, 231)
Clicked at: (393, 288)
Clicked at: (201, 331)
        x,y:x表示列
    """
    contours = np.array([[[[193, 270]],[[378, 231]],[[393, 288]],[[201, 331]]]])

    areas = []

    for vertices in contours:
        epsilon = 0.01 * cv2.arcLength(vertices, True)
        approx_vertices = cv2.approxPolyDP(vertices, epsilon, True)
        # approx_vertices = get_real_world_xyz(cam, approx_vertices.reshape(-1, 2)).reshape(-1, 3)
        approx_vertices = get_real_world_xyz(cam, approx_vertices.reshape(-1, 2)).reshape(-1, 3)
        areas.append(calculate_area(approx_vertices))

    return areas  # 返回的是所有多边形的面积

if __name__ == '__main__':
    # logic_matri= read_matrix_from_csv(r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output\1785966766862766080_2.csv")
    # pixel_area(logic_matri)
    areas = cal_astraimg_areas_mm2_test("raw/Depth_66.raw","raw/Color_66.raw","raw/Color_66.jpg")
    print(areas)
