# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: instance_count
Description:
Author: qinlinjian
Datetime: 5/19/2024 3:25 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"
import os
import cv2
import numpy as np

# 定义病害类型与灰度值的对应关系
disease_colors = {
    14: '裂缝',
    38: '露筋',
    75: '剥落',
    113: '侵蚀'
}

def count_diseases_in_image(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image {image_path}")
        return None

    # 初始化计数器
    counts = {key: 0 for key in disease_colors.keys()}

    # 统计每种病害的像素数量
    unique, counts_in_img = np.unique(img, return_counts=True)
    # print(unique,counts_in_img)
    count_dict = dict(zip(unique, counts_in_img))

    # 将统计结果保存到 counts 中
    for color, count in count_dict.items():
        if color in disease_colors:
            counts[color] = count

    return counts

def count_diseases_in_directory(directory_path):
    # 初始化总计数器
    total_counts = {key: 0 for key in disease_colors.keys()}

    # 遍历目录中的所有图片文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            counts = count_diseases_in_image(image_path)
            if counts:
                for key in total_counts:
                    total_counts[key] += counts[key]

    return total_counts

# 示例用法
# directory_path = r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\seg_app\utils\tmpimg"
directory_path = r'E:\homework_file\bishe\bs_paper_code_datas\mmsegmentation\data\multi_defect_new\gtFine\val'  # 替换为你的掩码图像目录路径
total_counts = count_diseases_in_directory(directory_path)

# 打印统计结果
for color, count in total_counts.items():
    print(f"{disease_colors[color]}: {count} instances")

# 训练集
# 裂缝: 21211373
# 露筋: 71534228
# 剥落: 652435527
# 侵蚀: 225432689

# 测试集
# 裂缝: 6337502 instances
# 露筋: 12010511 instances
# 剥落: 154473063 instances
# 侵蚀: 79758497 instances

if __name__ == '__main__':
    pass
