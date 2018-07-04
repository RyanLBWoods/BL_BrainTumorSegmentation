# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import os
import nibabel as nib

hgg_folder = 'MICCAI_BraTS_2018_Data_Training/HGG'
lgg_folder = 'MICCAI_BraTS_2018_Data_Training/LGG'

hgg_files = os.listdir(hgg_folder)
hgg_volumes = []
for hgg_file in hgg_files:
    hgg_volume = os.listdir(os.path.join(hgg_folder, hgg_file))
    for hgg_data in hgg_volume:
        hgg = os.path.join(hgg_folder, hgg_file, hgg_data)
        hgg_volumes.append(hgg)

print(hgg_volumes)
