# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: detect_crack
Name: crack_analysis
Description:
Author: qinlinjian
Datetime: 2023-04-05 13:38
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"
from skimage.morphology import skeletonize
import numpy as np
def crack_length(image):
    skeleton = skeletonize(image)
    print("像素总长度", np.sum(skeleton))
    crack_l = np.sum(skeleton)
    return crack_l



if __name__ == '__main__':
    pass
