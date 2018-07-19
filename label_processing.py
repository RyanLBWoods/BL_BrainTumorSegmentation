# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/19
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_scans_list(folder):
    seg_list = []
    grades = ['HGG', 'LGG']
    for grade in grades:
        files = os.listdir(os.path.join(folder, grade))
        for file in files:
            volume = os.listdir(os.path.join(folder, grade, file))
            for data in volume:
                data_path = os.path.join(folder, grade, file, data)
                if 'seg.nii' in data_path:
                    seg_list.append(data_path)
    return seg_list


data_folder = 'MICCAI_BraTS_2018_Data_Training'

gt_path = load_scans_list(data_folder)
print(len(gt_path))


def whole_tumor_label(gt_path_list):
    whole_tumor_label_list = []
    tumor_core_label_list = []
    cystic_label_list = []
    for path in gt_path_list:
        print("processing", path)
        seg_data = nib.load(path).get_data()
        shape = seg_data.shape
        whole_tumor_gt = []
        tumor_core_gt = []
        cystic_gt = []
        for slice in seg_data:
            whole_tumor_data = np.zeros(shape=(shape[1], shape[2]))
            tumor_core_data = np.zeros(shape=(shape[1], shape[2]))
            # enhancing_tumor = np.zeros(shape=(shape[1], shape[2]))
            cystic_data = np.zeros(shape=(shape[1], shape[2]))
            for row in range(len(slice)):
                for col in range(len(slice[row])):
                    if slice[row][col] != 0:
                        whole_tumor_data[row][col] = 1
                        if slice[row][col] != 2:
                            tumor_core_data[row][col] = 1
                            if slice[row][col] != 4:
                                cystic_data[row][col] = 1

            whole_tumor_gt.append(whole_tumor_data)
            tumor_core_gt.append(tumor_core_data)
            cystic_gt.append(cystic_data)

        whole_tumor_label_list.append(whole_tumor_gt)
        tumor_core_label_list.append(tumor_core_gt)
        cystic_label_list.append(cystic_gt)
        break
    whole_tumor_label_list = np.array(whole_tumor_label_list)
    tumor_core_label_list = np.array(tumor_core_label_list)
    cystic_label_list = np.array(cystic_label_list)
    return whole_tumor_label_list, tumor_core_label_list, cystic_label_list


whole_tumor, tumor_core, cystic = whole_tumor_label(gt_path)
