# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: dfs_account_dui
Description:
Author: qinlinjian
Datetime: 4/27/2024 11:36 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import pandas as pd
import numpy as np

def analyze_clusters_with_positions(matrix):
    if matrix.size == 0:
        return []

    rows, cols = matrix.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    cluster_sizes = []
    cluster_starts = []

    def dfs(r, c):
        stack = [(r, c)]
        size = 0
        start_position = (r, c)  # 记录小堆的起始位置
        while stack:
            sr, sc = stack.pop()
            if visited[sr][sc]:
                continue
            visited[sr][sc] = True
            size += 1
            for dr, dc in directions:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and matrix[nr][nc] == 1:
                    stack.append((nr, nc))
        return size, start_position

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1 and not visited[r][c]:
                size, start_pos = dfs(r, c)
                if size > 0:
                    cluster_sizes.append(size)
                    cluster_starts.append(start_pos)

    return cluster_sizes, cluster_starts

# 过滤小堆的
# def filter_clusters(matrix):
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
#     cluster_sizes = analyze_clusters(matrix)
#     print("size:",cluster_sizes)
#     if not cluster_sizes:
#         return matrix
#
#     mean_size = np.mean(cluster_sizes)
#     std_dev = np.std(cluster_sizes)
#     threshold = mean_size - std_dev  # One standard deviation below the mean
#
#     print(f"Mean size: {mean_size}, Standard deviation: {std_dev}, Threshold: {threshold}")
#
#     rows, cols = matrix.shape
#     visited = np.zeros((rows, cols), dtype=bool)
#     new_matrix = np.zeros((rows, cols), dtype=int)
#
#     def dfs(r, c, mark):
#         stack = [(r, c)]
#         size = 0
#         cells = []
#         while stack:
#             sr, sc = stack.pop()
#             if visited[sr][sc]:
#                 continue
#             visited[sr][sc] = True
#             cells.append((sr, sc))
#             size += 1
#             for dr, dc in directions:
#                 nr, nc = sr + dr, sc + dc
#                 if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and matrix[nr][nc] == 1:
#                     stack.append((nr, nc))
#         if size >= threshold or not mark:
#             for (sr, sc) in cells:
#                 new_matrix[sr][sc] = 1
#
#     for r in range(rows):
#         for c in range(cols):
#             if matrix[r][c] == 1 and not visited[r][c]:
#                 dfs(r, c, True)
#
#     return new_matrix

def read_matrix_from_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 将所有非零值转换为1，0值保持不变
    matrix = np.where(df != 0, 1, 0)
    return matrix


import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = r'E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\media\output\1783544751958659072_1.csv'
    matrix = read_matrix_from_csv(file_path)
    num_elements = matrix.size
    # 运行函数
    sizes, starts = analyze_clusters_with_positions(matrix)
    for size, start in zip(sizes, starts):
        print(f"Cluster size: {size}, proportion: {size/num_elements*100}%, Start position: {start}")

    plt.imshow(matrix, cmap='Greys', interpolation='none')
    plt.colorbar()  # 可以添加颜色条来指示True和False
    plt.show()
