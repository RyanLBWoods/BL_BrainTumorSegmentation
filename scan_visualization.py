# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/6
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import tensorflow as tf
import numpy as np


# import matplotlib.pyplot as plt

def load_scans_list(folder):
    all_list = []
    grades = ['HGG', 'LGG']
    for grade in grades:
        files = os.listdir(os.path.join(folder, grade))
        for file in files:
            volume = os.listdir(os.path.join(folder, grade, file))
            for data in volume:
                data_path = os.path.join(folder, grade, file, data)
                all_list.append(data_path)
    return all_list


def load_scans_dic():
    data_folder = 'MICCAI_BraTS_2018_Data_Training'

    scans_list = load_scans_list(data_folder)

    seg = []
    flair = []
    t1 = []
    t1ce = []
    t2 = []
    scans = []
    scans_dic = {}
    for vol in scans_list:
        if 'seg.nii' not in vol:
            scans.append(vol)
        if 'seg.nii' in vol:
            seg.append(vol)
        elif 'flair.nii' in vol:
            flair.append(vol)
        elif 't1.nii' in vol:
            t1.append(vol)
        elif 't1ce.nii' in vol:
            t1ce.append(vol)
        elif 't2.nii' in vol:
            t2.append(vol)

    # Build a dictionary in the format of {seg:[flair, t1, t1ce, t2]}
    for s in seg:
        if s not in scans_dic:
            scans_dic[s] = []
            for scan in scans:
                if s[:-11] in scan:
                    scans_dic[s].append(scan)
    return scans_dic


dic = load_scans_dic()


def load_scan_data(input_queue):
    scans_list = input_queue[0]
    label_list = input_queue[1]
    scan_imgs = []
    label_imgs = []
    for scan in scans_list:
        for sc in scan:
            scan_imgs.append(tf.convert_to_tensor(nib.load(sc).get_data(), dtype=tf.float32))
    for label in label_list:
        label_imgs.append(tf.convert_to_tensor(nib.load(label).get_data(), dtype=tf.float32))
    scan_img = tf.cast(tf.convert_to_tensor(scan_imgs), dtype=tf.float32)
    label_img = tf.cast(tf.convert_to_tensor(label_imgs), dtype=tf.float32)

    print(scan_img)
    print(label_img)


label = list(dic.keys())
scans = list(dic.values())
queue = [scans, label]

# queue = tf.train.slice_input_producer([scans, label])

load_scan_data(queue)
