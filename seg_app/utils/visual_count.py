# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: visual_count
Description:
Author: qinlinjian
Datetime: 5/19/2024 3:48 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import matplotlib.pyplot as plt

# 总共
from matplotlib import rcParams

pixel_count_results = {
    14: 6337502+21211373,
    38: 12010511+71534228,
    75: 154473063+652435527,
    113: 79758497+225432689
}

instance_count_results = {
    14: 936+3675,
    38: 422+2112,
    75: 493+1637,
    113: 277+1329
}

# # 训练集数据
# pixel_count_results = {
#     14: 21211373,
#     38: 71534228,
#     75: 652435527,
#     113: 225432689
# }
#
# instance_count_results = {
#     14: 3675,
#     38: 2112,
#     75: 1637,
#     113: 1329
# }
# 裂缝: 3675 instances
# 露筋: 2112 instances
# 剥落: 1637 instances
# 侵蚀: 1329 instances

# 测试集
# pixel_count_results = {
#     14: 6337502,
#     38: 12010511,
#     75: 154473063,
#     113: 79758497
# }
# # 裂缝: 6337502 instances
# # 露筋: 12010511 instances
# # 剥落: 154473063 instances
# # 侵蚀: 79758497 instances
#
# instance_count_results = {
#     14: 936,
#     38: 422,
#     75: 493,
#     113: 277
# }
# # 裂缝: 936 instances
# # 露筋: 422 instances
# # 剥落: 493 instances
# # 侵蚀: 277 instances


# 定义病害类型与名称的对应关系
disease_names = {
    14: '裂缝',
    38: '露筋',
    75: '剥落',
    113: '侵蚀'
}
# 设置字体为宋体以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 提取数据用于绘图
diseases = list(disease_names.values())
pixel_counts = [pixel_count_results[key] for key in disease_names.keys()]
instance_counts = [instance_count_results[key] for key in disease_names.keys()]

# 设置图表的大小
plt.figure(figsize=(12, 6))

# 绘制像素数量统计结果
plt.subplot(1, 2, 1)
plt.bar(diseases, pixel_counts, color='skyblue')
plt.title('病害像素数量统计', fontsize=18)
plt.xlabel('病害类型', fontsize=18)
plt.ylabel('像素数量', fontsize=18)
plt.xticks(fontsize=16)

# 绘制实例数量统计结果
plt.subplot(1, 2, 2)
plt.bar(diseases, instance_counts, color='lightgreen')
plt.title('病害实例数量统计', fontsize=18)
plt.xlabel('病害类型', fontsize=18)
plt.ylabel('实例数量', fontsize=18)
plt.xticks(fontsize=16)

# 调整布局
plt.tight_layout()
# Save the figure to a specified directory
output_path = r'E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\seg_app\utils\tmpimg/statistics.png'
plt.savefig(output_path)

# 显示图表
plt.show()


if __name__ == '__main__':
    pass
