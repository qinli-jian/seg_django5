# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: predict_img
Description:
Author: qinlinjian
Datetime: 4/21/2024 11:32 AM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import copy
import os
from datetime import datetime
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

from seg_app.utils import dfs_account_dui
from seg_django_5 import settings


class SegImg:
    def __init__(self,config,checkpoint):
        # config = 'configs/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py'
        self.config = config
        # 模型 checkpoint 权重文件'weight/best_mIoU_iter_40000.pth'
        self.checkpoint = checkpoint
        # device = 'cpu'
        # self.device = 'cuda:0'
        self.device = 'cpu'
        print("===初始化模型===")
        self.model = init_model(config, checkpoint, device=self.device)

        self.palette = [
            ['background', [127,127,127]],
            ['red', [0,0,200]],
            ['green', [0,200,0]],
            ['yellow', [0,238,238]],
            ['violet', [238,0,238]]
        ]
        self.palette_dict = {}
        for idx, each in enumerate(self.palette):
            self.palette_dict[idx] = each[1]
        self.opacity = 0.5 # 透明度
        print("===初始化结束===")

    '''
    统计病害数量（小堆的数量）
    '''
    def count_dui(self,mask):
        mask = np.where(mask != 0, 1, 0)
        num_el = mask.size
        sizes, starts = dfs_account_dui.analyze_clusters_with_positions(mask)
        label_attribute = {}
        for size, start in zip(sizes, starts):
            label_attribute["pixel_size"] = size
            label_attribute["attribute"] = size/num_el
            label_attribute["position"] = start
        return label_attribute

    def prefict(self,img_path,new_image_name):
        img_bgr = cv2.imread(img_path)
        result = inference_model(self.model, img_bgr)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        labels = np.unique(pred_mask)

        segments = []
        # 循环labels拿到对应的mask
        for label in labels:
            print(label)
            if label==0:
                continue
            mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1]), dtype=int)
            # color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            item = {}
            item['label'] = int(label)
            mask[np.where(pred_mask == label)] = int(label)
            # mask_gpu = torch.tensor(mask, device='cuda')
            csv_name = new_image_name.split('.')[0]+'_'+str(label)+'.csv'
            item['label_mask_csv'] = csv_name
            np.savetxt(os.path.join(settings.CSV_OUT_PATH,csv_name), mask, delimiter=",")

            label_attribute = self.count_dui(mask)
            item["label_attribute"] = label_attribute

            # color_mask[np.where(pred_mask == label)] = self.palette_dict[label]
            # color_mask_gpu = torch.tensor(color_mask, device='cuda')
            # item['color_mask'] = copy.deepcopy(color_mask.tolist())
            segments.append(copy.deepcopy(item))

        pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
        for idx in self.palette_dict.keys():
            pred_mask_bgr[np.where(pred_mask == idx)] = self.palette_dict[idx]
        pred_mask_bgr = pred_mask_bgr.astype('uint8')

        # 将语义分割预测图和原图叠加显示
        pred_viz = cv2.addWeighted(img_bgr, self.opacity, pred_mask_bgr, 1 - self.opacity, 0)

        # 保存语义分割之后的图片
        cv2.imwrite(os.path.join(settings.IMG_SEG_SAVE_PATH,new_image_name), pred_viz)
        return segments,self.palette_dict

if __name__ == '__main__':
    pass
