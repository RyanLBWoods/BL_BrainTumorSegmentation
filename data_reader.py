# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib

# training_folder = 'MICCAI_BraTS_2018_Data_Training'
hgg_folder = 'MICCAI_BraTS_2018_Data_Training/HGG'
lgg_folder = 'MICCAI_BraTS_2018_Data_Training/LGG'


def load_path(folder):
    volumes = []
    files = os.listdir(folder)
    for file in files:
        volume = os.listdir(os.path.join(folder, file))
        for data in volume:
            data_path = os.path.join(folder, file, data)
            volumes.append(data_path)
    return volumes


hgg_volumes = load_path(hgg_folder)
lgg_volumes = load_path(lgg_folder)
