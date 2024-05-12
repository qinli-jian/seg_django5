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

import statistics
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

from seg_app.utils import dfs_account_dui, cal_break_w_l,cal_break23_area_mm2
from seg_app.utils.AstraCam import AstraCam
from seg_app.utils.cal_break23_area_mm2 import cal_astraimg_areas_mm2
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

    def prefict(self,img_path,new_image_name,pixel_size_mm=1):
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
            measurements = {}
            csv_path = os.path.join(settings.CSV_OUT_PATH,csv_name)
            if label==1 or label ==4:
                measurements = cal_break_w_l.break_l_w_analysis(csv_path,os.path.join(settings.IMG_SEG_SAVE_PATH))
            else:
                measurements = cal_break23_area_mm2.get_areas(csv_path,pixel_size_mm)
            item["label_attribute"] = {**label_attribute,**measurements}

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

    def predict_AstraCamImg(self,img_path,new_image_name,depth_img_path):
        shape = settings.ASTRA_CAM_SHAPE
        cam = AstraCam(
            depth_img_path=depth_img_path,
            img_path=img_path,
            save_rgb_path=os.path.join(settings.IMG_SEG_SAVE_PATH,new_image_name),
            shape=shape)
        img_bgr = cam.rgb_img
        result = inference_model(self.model, cam.rgb_img)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        labels = np.unique(pred_mask)
        segments = []
        # 循环labels拿到对应的mask
        print("*****astra cam*****")
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
            csv_name = new_image_name.split('.')[0] + '_' + str(label) + '.csv'
            item['label_mask_csv'] = csv_name
            np.savetxt(os.path.join(settings.CSV_OUT_PATH, csv_name), mask, delimiter=",")

            label_attribute = self.count_dui(mask)
            measurements = {}
            csv_path = os.path.join(settings.CSV_OUT_PATH, csv_name)

            if label==1 or label==4:
                longest_paths = cal_break_w_l.get_skeleton_pixelpaths(csv_path)
                real_distance_mm = []
                for path in longest_paths:
                    dis_mm = []
                    for idx in range(1,len(path)):
                        u1 = path[idx-1][0]
                        v1 = path[idx-1][1]
                        u2 = path[idx][0]
                        v2 = path[idx][1]
                        dis = cam.real_distance_mm(u1,v1,u2,v2)
                        dis_mm.append(dis)
                    real_distance_mm.append(sum(dis_mm))
                min_length_mm = min(real_distance_mm)
                max_length_mm = max(real_distance_mm)
                aver_length_mm = sum(real_distance_mm)/len(real_distance_mm)
                median_length_mm = statistics.median(real_distance_mm)
                tal_length_mm = sum(real_distance_mm)

                # 宽度
                pt_arr,skeleton_imgname,distance_imgname = cal_break_w_l.get_real_width_mm(csv_path,os.path.join(settings.IMG_SEG_SAVE_PATH))
                real_width_mm = []
                for pos in pt_arr:
                    u1 = pos[0][0]
                    v1 = pos[0][1]
                    u2 = pos[1][0]
                    v2 = pos[1][1]
                    width_mm = cam.real_distance_mm(u1,v1,u2,v2)
                    real_width_mm.append(width_mm)
                min_width_mm = min(real_width_mm)
                max_width_mm = max(real_width_mm)
                aver_width_mm = sum(real_width_mm)/len(real_distance_mm)
                median_width_mm = statistics.median(real_width_mm)

                measurements = {
                    "real_min_len": int(min_length_mm),
                    "real_max_len": int(max_length_mm),
                    "real_aver_len": float(aver_length_mm),
                    "real_median_len": int(median_length_mm),
                    "real_tal_len": int(tal_length_mm),
                    "real_aver_width": float(aver_width_mm),
                    "real_min_width": int(min_width_mm),
                    "real_max_width": int(max_width_mm),
                    "real_median_width": int(median_width_mm),
                    "distance_img": distance_imgname,
                    "skeleton_img": skeleton_imgname,
                    "width_points_array": pt_arr
                }
            else:
                # TODO 计算面积
                areas = cal_astraimg_areas_mm2(csv_path,depth_img_path,img_path,save_rgb_path=settings.IMG_SEG_SAVE_PATH)
                min_area_mm2 = min(areas)
                max_area_mm2 = max(areas)
                aver_area_mm2 = sum(areas)/len(areas)
                median_area_mm2 = statistics.median(areas)
                measurements = {
                    "min_area_mm2":min_area_mm2,
                    "max_area_mm2":max_area_mm2,
                    "aver_area_mm2":aver_area_mm2,
                    "median_area_mm2":median_area_mm2
                }
                pass
            item["label_attribute"] = {**label_attribute, **measurements}

            segments.append(copy.deepcopy(item))

            pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            for idx in self.palette_dict.keys():
                pred_mask_bgr[np.where(pred_mask == idx)] = self.palette_dict[idx]
            pred_mask_bgr = pred_mask_bgr.astype('uint8')

            # 将语义分割预测图和原图叠加显示
            pred_viz = cv2.addWeighted(img_bgr, self.opacity, pred_mask_bgr, 1 - self.opacity, 0)

            # 保存语义分割之后的图片
            cv2.imwrite(os.path.join(settings.IMG_SEG_SAVE_PATH, new_image_name), pred_viz)
            return segments, self.palette_dict

if __name__ == '__main__':
    pass
