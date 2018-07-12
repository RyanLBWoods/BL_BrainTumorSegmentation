# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
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
        for file in files:
            volume = os.listdir(os.path.join(folder, grade, file))
            for data in volume:
                data_path = os.path.join(folder, grade, file, data)
                all_list.append(data_path)
    return all_list


def load_scans_dic(data_dir):

    scans_list = load_scans_list(data_dir)

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
        else:
            seg.append(vol)
        # elif 'flair.nii' in vol:
        #     flair.append(vol)
        # elif 't1.nii' in vol:
        #     t1.append(vol)
        # elif 't1ce.nii' in vol:
        #     t1ce.append(vol)
        # elif 't2.nii' in vol:
        #     t2.append(vol)

    # Build a dictionary in the format of {seg:[flair, t1, t1ce, t2]}
    for s in seg:
        if s not in scans_dic:
            scans_dic[s] = []
            for scan in scans:
                if s[:-11] in scan:
                    scans_dic[s].append(scan)

    # print(scans_dic)
    return scans_dic


def load_scan_data(mri_dict, input_size, random_scale):
    # scans_list = list(mri_dict.values())
    # label_list = list(mri_dict.keys())
    scan_imgs = []
    label_imgs = []
    scan_img_list = []
    for label in mri_dict:
        # label_imgs.append(nib.load(label).get_data())
        label_imgs.append(tf.convert_to_tensor(nib.load(label).get_data(), dtype=tf.float32))
        for scan in mri_dict[label]:
            # scan_imgs.append(nib.load(scan).get_data())
            scan_imgs.append(tf.convert_to_tensor(nib.load(scan).get_data(), dtype=tf.float32))
        scan_img_list.append(scan_imgs)
        scan_imgs = []
    # for scan in scans_list:
    #     for sc in scan:
    #         scan_imgs.append(tf.convert_to_tensor(nib.load(sc).get_data(), dtype=tf.float32))
    #         scan_tensor.append(scan_imgs)
    #     scan_imgs = []
    # for label in label_list:
    #     label_imgs.append(tf.convert_to_tensor(nib.load(label).get_data(), dtype=tf.float32))

    # scan_img = tf.cast(tf.convert_to_tensor(scan_tensor), dtype=tf.float32)
    # label_img = tf.cast(tf.convert_to_tensor(label_imgs), dtype=tf.float32)

    scan_tensor = tf.convert_to_tensor(scan_img_list, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(label_imgs, dtype=tf.float32)
    # return scan_img, label_img
    return scan_tensor, label_tensor


class ScanReader(object):
    def __init__(self, data_dir, input_size, random_scale, coord):
        self.data_dir = data_dir
        self.input_size = input_size
        self.coord = coord

        self.mri_dic = load_scans_dic(self.data_dir)
        # self.scans_list = tf.convert_to_tensor(list(self.scans_dic.values()), dtype=tf.string)
        self.train_dic, self.validation_dic = train_validation_split(self.mri_dic)
        print("train dict", len(self.train_dic), self.train_dic)
        print("validation dict", len(self.validation_dic), self.validation_dic)
        # self.scans_list = list(self.train_dic.values())
        # self.label_list = list(self.train_dic.keys())
        # self.label_list = tf.convert_to_tensor(list(self.scans_dic.keys()), dtype=tf.string)
        # self.queue = tf.train.slice_input_producer([self.scans_list, self.label_list], shuffle=input_size is not None)
        # self.queue = [self.scans_list, self.label_list]
        self.scan_img, self.label_img = load_scan_data(self.train_dic, self.input_size, random_scale)

    def dequeue(self, num_elements):
        scan_batch, label_batch = tf.train.batch([self.scan_img, self.label_img], num_elements)
        return scan_batch, label_batch
