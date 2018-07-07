# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def load_scans_dic(data_dir):
    data_folder = ['MICCAI_BraTS_2018_Data_Training/HGG', 'MICCAI_BraTS_2018_Data_Training/LGG']

    def load_scans_list(folder):
        all_list = []
        for grade in folder:
            files = os.listdir(grade)
            for file in files:
                volume = os.listdir(os.path.join(grade, file))
                for data in volume:
                    data_path = os.path.join(grade, file, data)
                    all_list.append(data_path)
        return all_list

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

    # print(scans_dic)
    return scans_dic


def load_scan_data(input_queue, input_size, random_scale):
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
    
    return scan_img, label_img


class ScanReader(object):
    def __init__(self, data_dir, input_size, random_scale, coord):
        self.data_dir = data_dir
        self.input_size = input_size
        self.coord = coord

        self.scans_dic = load_scans_dic(self.data_dir)
        # self.scans_list = tf.convert_to_tensor(list(self.scans_dic.values()), dtype=tf.string)
        self.scans_list = list(self.scans_dic.values())
        self.label_list = list(self.scans_dic.keys())
        # self.label_list = tf.convert_to_tensor(list(self.scans_dic.keys()), dtype=tf.string)
        # self.queue = tf.train.slice_input_producer([self.scans_list, self.label_list], shuffle=input_size is not None)
        self.queue = [self.scans_list, self.label_list]
        self.scan_img, self.label_img = load_scan_data(self.queue, self.input_size, random_scale)

    def dequeue(self, num_elements):
        scan_batch, label_batch = tf.train.batch([self.scan_img, self.label_img], num_elements)
        return scan_batch, label_batch
