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


def load_testing_list(folder):
    all_list = []
    files = os.listdir(folder)
    files.sort()
    for file in files:
        if os.path.isdir(os.path.join(folder, file)):
            if ".DS_Store" not in file:
                volume = os.listdir(os.path.join(folder, file))
                for data in volume:
                    if ".DS_Store" not in data:
                        data_path = os.path.join(folder, file, data)
                        all_list.append(data_path)
    return all_list


def load_scans_dic(data_dir):
    scans_list = load_testing_list(data_dir)
    scans_dic = {}
    # Build a dictionary in the format of {seg:[flair, t1, t1ce, t2]}
    for scan in scans_list:
        # if s not in scans_dic:
        s = scan[:scan.rfind('/')]
        if s not in scans_dic:
            scans_dic[s] = []
            scans_dic[s].append(scan)
        else:
            scans_dic[s].append(scan)
    return scans_dic


class ScanReader(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mri_dic = load_scans_dic(self.data_dir)
