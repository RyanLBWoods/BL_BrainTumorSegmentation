# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

data_folder = ['MICCAI_BraTS_2018_Data_Training/HGG', 'MICCAI_BraTS_2018_Data_Training/LGG']


def load_all_volumes(folder):
    volumes = []
    for grade in folder:
        files = os.listdir(grade)
        for file in files:
            volume = os.listdir(os.path.join(grade, file))
            for data in volume:
                data_path = os.path.join(grade, file, data)
                volumes.append(data_path)
    return volumes


whole_volumes = load_all_volumes(data_folder)

seg = []
flair = []
t1 = []
t1ce = []
t2 = []
scans = []
scans_dic = {}
for v in whole_volumes:
    if 'seg.nii' not in v:
        scans.append(v)
    if 'seg.nii' in v:
        seg.append(v)
    elif 'flair.nii' in v:
        flair.append(v)
    elif 't1.nii' in v:
        t1.append(v)
    elif 't1ce.nii' in v:
        t1ce.append(v)
    elif 't2.nii' in v:
        t2.append(v)

# Build a dictionary in the format of {seg:[flair, t1, t1ce, t2]}
for s in seg:
    if s not in scans_dic:
        scans_dic[s] = []
        for scan in scans:
            if s[:-11] in scan:
                scans_dic[s].append(scan)

print(scans_dic)
ground_truth = []
modals = []
segment_dic = {}
for key in scans_dic:
    gt = nib.load(key).get_data()
    ground_truth.append(gt)

