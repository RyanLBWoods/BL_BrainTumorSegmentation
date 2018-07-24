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
        files.sort()
        for file in files:
            if 'DS_Store' not in file:
                volume = os.listdir(os.path.join(folder, grade, file))
                for data in volume:
                    data_path = os.path.join(folder, grade, file, data)
                    if 'seg.nii' in data_path:
                        seg_list.append(data_path)
    return seg_list


def tumor_label(gt_path_list):
    whole_tumor_label_list = []
    tumor_core_label_list = []
    cystic_label_list = []
    whole_tumor_label_dict = {}
    tumor_core_label_dict = {}
    cystic_label_dict = {}
    for path in gt_path_list:
        print("processing", path)
        seg_data = nib.load(path).get_data()
        shape = seg_data.shape
        whole_tumor_gt = []
        tumor_core_gt = []
        cystic_gt = []
        tumor_num = 0
        core_num = 0
        cystic_num = 0
        for slice in seg_data:
            have_tumor = False
            have_core = False
            have_cystic = False
            whole_tumor_data = np.zeros(shape=(shape[1], shape[2]))
            tumor_core_data = np.zeros(shape=(shape[1], shape[2]))
            # enhancing_tumor = np.zeros(shape=(shape[1], shape[2]))
            cystic_data = np.zeros(shape=(shape[1], shape[2]))
            for row in range(len(slice)):
                for col in range(len(slice[row])):
                    if slice[row][col] == 3:
                        print("find 3 in ", path)
                    if slice[row][col] != 0:
                        have_tumor = True
                        whole_tumor_data[row][col] = 1
                        if slice[row][col] != 2:
                            have_core = True
                            tumor_core_data[row][col] = 1
                            if slice[row][col] != 4:
                                have_cystic = True
                                cystic_data[row][col] = 1
            if have_tumor:
                tumor_num = tumor_num + 1
                whole_tumor_tuple = (whole_tumor_data, 1)
            else:
                whole_tumor_tuple = (whole_tumor_data, 0)

            if have_core:
                core_num = core_num + 1
                tumor_core_tuple = (tumor_core_data, 1)
            else:
                tumor_core_tuple = (tumor_core_data, 0)

            if have_cystic:
                cystic_num = cystic_num + 1
                cystic_tuple = (cystic_data, 1)
            else:
                cystic_tuple = (cystic_data, 0)

            whole_tumor_gt.append(whole_tumor_tuple)
            tumor_core_gt.append(tumor_core_tuple)
            cystic_gt.append(cystic_tuple)

        whole_tumor_label_dict[path] = whole_tumor_gt
        tumor_core_label_dict[path] = tumor_core_gt
        cystic_label_dict[path] = cystic_gt

        whole_tumor_gt = []
        tumor_core_gt = []
        cystic_gt = []
        print(tumor_num)
        print(core_num)
        print(cystic_num)
        break
    #     whole_tumor_label_list.append(whole_tumor_gt)
    #     tumor_core_label_list.append(tumor_core_gt)
    #     cystic_label_list.append(cystic_gt)
    #
    # whole_tumor_label_list = np.array(whole_tumor_label_list)
    # tumor_core_label_list = np.array(tumor_core_label_list)
    # cystic_label_list = np.array(cystic_label_list)
    # return whole_tumor_label_list, tumor_core_label_list, cystic_label_list
    return whole_tumor_label_dict, tumor_core_label_dict, cystic_label_dict


data_folder = 'MICCAI_BraTS_2018_Data_Training'
gt_path = load_scans_list(data_folder)
whole_tumor, tumor_core, cystic = tumor_label(gt_path)
exit(0)
print(tumor_core)
print(cystic)

np.save("whole_tumor.npy", whole_tumor)
np.save("tumor_core.npy", whole_tumor)
np.save("cystic.npy", whole_tumor)
