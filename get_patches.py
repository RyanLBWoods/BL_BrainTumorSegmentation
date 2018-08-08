# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/11
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import numpy as np
import json
from utils import *


def load_scans_list(folder):
    wt_list = []
    grades = ['HGG', 'LGG']
    for grade in grades:
        files = os.listdir(os.path.join(folder, grade))
        files.sort()
        for file in files:
            if 'DS_Store' not in file:
                volume = os.listdir(os.path.join(folder, grade, file))
                for data in volume:
                    data_path = os.path.join(folder, grade, file, data)
                    if 'whole_tumor_label' in data_path:
                        wt_list.append(data_path)
    return wt_list


def tumor_label(gt_path_list):
    for path in gt_path_list:
        print("processing", path)
        # Create data dictionary
        whole_tumor_label_dict = {}
        tumor_core_label_dict = {}
        cystic_label_dict = {}

        # Load nii data
        seg_data = nib.load(path).get_data()
        shape = seg_data.shape
        whole_tumor_gt = []
        tumor_core_gt = []
        cystic_gt = []
        for slice in seg_data:
            have_tumor = False
            have_core = False
            have_cystic = False
            whole_tumor_data = np.zeros(shape=(shape[1], shape[2]))
            tumor_core_data = np.zeros(shape=(shape[1], shape[2]))
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
                whole_tumor_tuple = (whole_tumor_data.tolist(), 1)
            else:
                whole_tumor_tuple = (whole_tumor_data.tolist(), 0)

            if have_core:
                tumor_core_tuple = (tumor_core_data.tolist(), 1)
            else:
                tumor_core_tuple = (tumor_core_data.tolist(), 0)

            if have_cystic:
                cystic_tuple = (cystic_data.tolist(), 1)
            else:
                cystic_tuple = (cystic_data.tolist(), 0)

            whole_tumor_gt.append(whole_tumor_tuple)
            tumor_core_gt.append(tumor_core_tuple)
            cystic_gt.append(cystic_tuple)

        whole_tumor_label_dict[path[:path.rfind('/')]] = whole_tumor_gt
        tumor_core_label_dict[path[:path.rfind('/')]] = tumor_core_gt
        cystic_label_dict[path[:path.rfind('/')]] = cystic_gt

        # Create file to store processed label
        whole_tumor_label_file = open(path[:path.rfind('/')] + '/whole_tumor_label.json', 'w')
        tumor_core_label_file = open(path[:path.rfind('/')] + '/tumor_core_label.json', 'w')
        cystic_label_file = open(path[:path.rfind('/')] + '/cystic_label.json', 'w')

        # Save processed data
        print("Saving processed label data")
        wtl_obj = json.dumps(whole_tumor_label_dict)
        whole_tumor_label_file.write(wtl_obj)

        tc_obj = json.dumps(tumor_core_label_dict)
        tumor_core_label_file.write(tc_obj)

        cystic_obj = json.dumps(cystic_label_dict)
        cystic_label_file.write(cystic_obj)

        whole_tumor_label_file.close()
        tumor_core_label_file.close()
        cystic_label_file.close()


data_folder = 'MICCAI_BraTS_2018_Data_Training'
wt_path = load_scans_list(data_folder)
print(wt_path)
# tumor_label(gt_path)
