# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: read_raw_rgbd
Description:
Author: qinlinjian
Datetime: 5/3/2024 3:26 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import numpy as np
import cv2


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
if __name__ == '__main__':
    img=np.fromfile(r'raw/Color_66.raw', dtype='uint8')
    # img = np.fromfile(r'raw/Depth_1.raw', dtype='uint16')
    # img = cv2.imread(r'raw/Color_1714722545676_0.raw')

    print(img)
    # print("d:",img[181602])
    print(len(img))
    img = img.reshape((480, 640, 3))
    print(img.shape)
    # for i in range(len(img)):
    #     for j in img[i]:
    #         img[i][j] = np.append(img[i][j],0)

    # print(img[116][412])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # print(img)
    # print(img.shape)
    # print(len(img))
    # print(len(img[0]))
    # 设置窗口
    cv2.namedWindow("img")

    # 设置鼠标回调函数
    cv2.setMouseCallback("img", mouse_callback)
    while True:
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按下ESC键退出
            break
