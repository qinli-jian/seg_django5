# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: instance_count
Description:
Author: qinlinjian
Datetime: 5/19/2024 3:42 PM
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

def count_disease_instances_in_image(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image {image_path}")
        return None

    # 初始化计数器
    counts = {key: 0 for key in disease_colors.keys()}

    # 统计每种病害的实例数量
    for color in disease_colors.keys():
        # 创建一个二值掩码，表示当前颜色的区域
        binary_mask = (img == color).astype(np.uint8)
        # 使用连通组件分析统计区域数量
        num_labels, labels = cv2.connectedComponents(binary_mask)
        # 减去背景标签
        counts[color] = num_labels - 1

    return counts

def count_diseases_in_directory(directory_path):
    # 初始化总计数器
    total_counts = {key: 0 for key in disease_colors.keys()}

    # 遍历目录中的所有图片文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            counts = count_disease_instances_in_image(image_path)
            if counts:
                for key in total_counts:
                    total_counts[key] += counts[key]

    return total_counts

# 示例用法
directory_path = r'E:\homework_file\bishe\bs_paper_code_datas\mmsegmentation\data\multi_defect_new\gtFine\val'  # 替换为你的掩码图像目录路径
total_counts = count_diseases_in_directory(directory_path)

# 打印统计结果
for color, count in total_counts.items():
    print(f"{disease_colors[color]}: {count} instances")

# 训练集
#     14: 3675,
#     38: 2112,
#     75: 1637,
#     113: 1329

# 测试集

if __name__ == '__main__':
    pass
