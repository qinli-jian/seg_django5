# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: AstraCam
Description:
Author: qinlinjian
Datetime: 5/3/2024 5:43 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import cv2
import numpy as np
from seg_django_5 import settings

class AstraCam():

    def __init__(self,depth_img_path,img_path,save_rgb_path,shape):
        self.R = np.array([[0.99998,-0.00233813,-0.00589042],
                           [0.00233627,0.999997,-0.000322957],
                           [0.00589116,0.000309189,0.999983]])
        self.T = np.array([-25.1456,0.151228,-0.717819])
        self.fx = settings.fx
        self.fy = settings.fy
        self.cx = settings.cx
        self.cy = settings.cy

        self.fx_rgb = 513.607
        self.fy_rgb = 513.607
        self.cx_rgb = 334.983
        self.cy_rgb = 249.776

        depth_img = np.fromfile(depth_img_path, dtype='uint16')
        depth_img = depth_img.reshape((shape[0], shape[1], 1)) # 使用接口上传的时候不用重塑深度图片的形状
        self.depth_img = depth_img
        self.shape = shape

        # 传入需要时y,x y表示行
        self.real_word_xyz_matrix = self.get_real_word_xyz_matrix(shape)
        self.raw_2_rbgimg(img_path,save_rgb_path,shape)
        pass

    """
    传入像素坐标(u, v)，传入完整的深度图路径
    """
    def real_word_xyz(self,u,v):
        Z_ir = float(self.depth_img[u][v])
        X_ir = float((u-self.cx_rgb)*Z_ir/self.fx_rgb)
        Y_ir = float((v-self.cy_rgb)*Z_ir/self.fy_rgb)
        # print("红外：",[X_ir, Y_ir, Z_ir])
        # X_rgb, Y_rgb, Z_rgb = np.dot(self.R, np.array([X_ir, Y_ir, Z_ir])) + self.T
        # print([X_rgb, Y_rgb, Z_rgb])
        # return np.array([X_rgb, Y_rgb, Z_rgb]).reshape((3,))
        return np.array([X_ir, Y_ir, Z_ir]).reshape((3,))

    def get_real_word_xyz_matrix(self,shape):
        real_word_mat = np.zeros((shape[0],shape[1],3))
        h = shape[0]
        w = shape[1]
        for u in range(h):
            for v in range(w):
                real_word_mat[u,v] = self.real_word_xyz(u,v)
        return real_word_mat

    def get_depth_from_depthimg(self,depth_img_path,shape):
        img = np.fromfile(depth_img_path, dtype='uint16')
        img = img.reshape((shape[0], shape[1], 1))
        return img

    def real_distance_mm(self,u1,v1,u2,v2):
        p1 = self.real_word_xyz_matrix[u1][v1]
        p2 = self.real_word_xyz_matrix[u2][v2]
        distance = np.linalg.norm(p2 - p1)
        return distance

    def raw_2_rbgimg(self,raw_path,save_path,shape):
        img = np.fromfile(raw_path, dtype='uint8')
        img = img.reshape((shape[0], shape[1], 3))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, img)
        self.rgb_img = img



# 测试的时候记得解开注释重塑形状
if __name__ == '__main__':
    cam = AstraCam(depth_img_path=r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\seg_app\utils\raw\Depth_2.raw",
                   img_path=r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\seg_app\utils\raw\Color_2.raw",
                   save_rgb_path=r"E:\homework_file\bishe\bs_paper_code_datas\seg_django_5\seg_app\utils\raw\Color_2.jpg",
             shape = [480,640])
    # Clicked
    # at: (303, 165)
    # Clicked
    # at: (276, 218)
    # Clicked
    # at: (268, 263)
    # Clicked
    # at: (260, 285)
    print("结构光相机测量的距离(mm):",cam.real_distance_mm(235,254,237,403))
    # print(cam.real_distance_mm(218,276,263,268))
    # print(cam.real_distance_mm(263,268,285,260))

    # l1 = cam.real_distance_mm(165,303,218,276)
    # l2 = cam.real_distance_mm(218,276,263,268)
    # l3 = cam.real_distance_mm(263,268,285,260)
    # print(l1+l2+l3)
    # cv2.imshow("img", cam.rgb_img)
    # cv2.waitKey(0)
    # real_mat = cam.real_word_xyz_matrix
    # p1 = real_mat[237][256]
    # p2 = real_mat[241][404]
    # print(p1)
    # print(p2)
    # distance = np.linalg.norm(p2 - p1)
    # print(distance)# 206.0674668841952
