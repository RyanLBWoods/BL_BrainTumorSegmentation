# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/16
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import numpy as np
import tensorflow as tf


# import matplotlib.pyplot as plt


def train_validation_split(data_dict):
    test_size = 0.3
    volume_num = len(data_dict)
    train_index = list(data_dict.keys())
    np.random.shuffle(train_index)
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
    grades = ['HGG', 'LGG']
    all_list = []
    for grade in grades:
        files = os.listdir(os.path.join(folder, grade))
        files.sort()
        for file in files:
            if ".DS_Store" not in file:
                volume = os.listdir(os.path.join(folder, grade, file))
                for data in volume:
                    if ".DS_Store" not in data:
                        data_path = os.path.join(folder, grade, file, data)
                        all_list.append(data_path)
    return all_list


def load_scans_dic(data_dir):
    scans_list = load_scans_list(data_dir)

    seg = []
    scans = []
    scans_dic = {}
    for vol in scans_list:
        if 'seg.nii' not in vol:
            scans.append(vol)
        else:
            seg.append(vol)

    for s in seg:
        if s not in scans_dic:
            scans_dic[s] = []
            for scan in scans:
                if s[:s.rfind('/')] in scan:
                    scans_dic[s].append(scan)
    return scans_dic


def load_scan_data(mri_dict):
    scan_imgs = []
    label_imgs = []

    for label in mri_dict:
        print(label)
        for scan in mri_dict[label]:
            scan_data = nib.load(scan).get_data()
            for data in scan_data:
                scan_imgs.append(data)
    return scan_imgs, label_imgs


class ScanReader(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.input_size = input_size
        # self.coord = coord

        self.mri_dic = load_scans_dic(self.data_dir)
        # self.scans_list = tf.convert_to_tensor(list(self.scans_dic.values()), dtype=tf.string)
        self.train_dic, self.validation_dic = train_validation_split(self.mri_dic)
        # self.scans_list = list(self.train_dic.values())
        # self.label_list = list(self.train_dic.keys())
        # self.label_list = tf.convert_to_tensor(list(self.scans_dic.keys()), dtype=tf.string)
        # self.queue = tf.train.slice_input_producer([self.scans_list, self.label_list], shuffle=input_size is not None)
        # self.queue = [self.scans_list, self.label_list]
        # self.scan_img, self.label_img = load_scan_data(self.train_dic)
        # self.validation_scans, self.validation_labels = load_scan_data(self.validation_dic)
