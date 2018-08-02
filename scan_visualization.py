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
import keras

import matplotlib.pyplot as plt
from matplotlib import colors


def train_validation_split(data_dict):
    test_size = 0.3
    volume_num = len(data_dict)
    train_index = list(data_dict.keys())
    print(train_index)
    np.random.shuffle(train_index)
    print(train_index)
    test_index = []
    test_num = int(volume_num * test_size)
    train = {}
    test = {}
    for i in range(test_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        test_index.append(train_index[random_index])
        del train_index[random_index]
    for index in train_index:
        train[index] = data_dict[index]
    for index in test_index:
        test[index] = data_dict[index]
    return train, test


def load_scans_list(folder):
    all_list = []
    grades = ['HGG', 'LGG']
    for grade in grades:
        files = os.listdir(os.path.join(folder, grade))
        for file in files:
            if '.DS_Store' not in file:
                volume = os.listdir(os.path.join(folder, grade, file))
                for data in volume:
                    data_path = os.path.join(folder, grade, file, data)
                    all_list.append(data_path)
    return all_list


def load_scans_dic():
    data_folder = 'MICCAI_BraTS_2018_Data_Training'

    scans_list = load_scans_list(data_folder)

    seg = []
    scans = []
    scans_dic = {}
    for vol in scans_list:
        if 'seg.nii' not in vol:
            scans.append(vol)
        else:
            seg.append(vol)

    # Build a dictionary in the format of {seg:[flair, t1, t1ce, t2]}
    for s in seg:
        if s not in scans_dic:
            scans_dic[s] = []
            for scan in scans:
                if s[:-11] in scan:
                    scans_dic[s].append(scan)
    return scans_dic


def colormap():
    map_list = ['#000000', '#FF0000', '#008B00', '#B0E2FF', '#FFFF00']
    return colors.ListedColormap(map_list, 'indexed')


def load_scan_data(dict):
    scan_data = []
    data = []
    for label in dict:
        label_data = nib.load(label).get_data()
        count = 0
        for scan in dict[label]:
            if '.json' not in scan:
                scan_data = np.expand_dims(nib.load(scan).get_data(), axis=-1)
                if count == 0:
                    data = scan_data
                else:
                    data = np.concatenate((data, scan_data), axis=-1)
                count = count + 1
        break
    print(data.shape)
    print(data[0].shape)
    exit(0)
    # for i in range(len(label_data)):
    #     plt.imshow(scan_data[i], cmap='gray')
    #     plt.imshow(label_data[i], cmap=colormap(), alpha=0.3)
    #     plt.show()
    # break


dic = load_scans_dic()
train_dic, test_dic = train_validation_split(dic)
load_scan_data(train_dic)
