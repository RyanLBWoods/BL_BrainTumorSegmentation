# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import numpy as np
import json
from PIL import Image
import tensorflow as tf
# from keras.engine.topology import Layer
import pydensecrf.densecrf as dcrf
from matplotlib import colors
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras.backend as K
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn import preprocessing

n_classes = 2  # Whole tumor, tumor core, enhancing tumor, cystic/necrotic component, non-tumor part
label_colors = [(255, 255, 0), (255, 0, 0), (176, 1226, 255), (0, 255, 0)]
# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
min_max_scaler = preprocessing.MinMaxScaler()
normalizer = preprocessing.Normalizer()


def colormap():
    map_list = ['#000000', '#FF0000']  # , '#008B00', '#B0E2FF', '#FFFF00']
    return colors.ListedColormap(map_list, 'indexed')


def pred_colormap():
    map_list = ['#000000', '#008B00']
    return colors.ListedColormap(map_list, 'indexed')


def plot(scans, segs):
    for i in range(len(segs)):
        plt.imshow(scans[i], cmap='gray')
        plt.imshow(segs[i], cmap=colormap(), alpha=0.3)
        plt.show()


def extract_patches(img, patch_size):
    m, n = img.shape
    b0, b1 = patch_size
    return img.reshape(m // b0, b0, n // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)


def reconstruct_image(patch, img_size):
    p_n, p_h, p_w = patch.shape
    img_h, img_w = img_size
    return patch.reshape(img_h // p_h, -1, p_h, p_w).swapaxes(1, 2).reshape(img_h, img_w)


def dice_coef(y_true, y_pred, smooth=1):
    # intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    # union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def batch_generator(dict, batch_size, n_classes, label_class):
    while True:
        for key in dict:
            count = 0
            data = []
            x = []
            y = []
            ids = []
            # x_train, y_train = [], []
            for scan in dict[key]:
                if '.json' not in scan:
                    scan_data = nib.load(scan).get_data().astype(np.float32)
                    for s in range(0, len(scan_data)):
                        # scaled = min_max_scaler.fit_transform(scan_data[s])
                        scaled = normalizer.fit_transform(scan_data[s])
                        scan_data[s] = scaled
                    scan_data = np.expand_dims(scan_data, axis=-1)
                    if count == 0:
                        data = scan_data
                    else:
                        data = np.concatenate((data, scan_data), axis=-1)
                    count = count + 1
                if label_class in scan:
                    with open(scan, 'r') as f:
                        label_data = json.load(f)
                        for key in label_data:
                            slices = [s for (s, _) in label_data[key]]
                            labels = [l for (_, l) in label_data[key]]
                            for index in range(0, len(labels)):
                                if labels[index] == 1:
                                    ids.append(index)
                                    y.append(slices[index])
            for i in ids:
                x.append(data[i])
            x = np.array(x)
            print(x[30][120])
            exit(0)
            y = to_categorical(np.array(y), n_classes)
            # for i in range(0, len(y)):
            #     x_patches = extract_patches_2d(x[i], (48, 31))
            #     y_patches = extract_patches_2d(y[i], (48, 31))
            #     x_train, y_train = [], []
            #     for p in range(0, len(y_patches)):
            #         if 1 in y_patches[p]:
            #             x_train.append(x_patches[p])
            #             y_train.append(y_patches[p])
            #     x_train = np.array(x_train)
            #     y_train = to_categorical(y_train, n_classes)
            for j in range(0, len(x), batch_size):
                yield (x[j:j + batch_size], y[j:j + batch_size])


def test_batch_generator(dict, batch_size):
    while True:
        for key in dict:
            # seg_data = nib.load(key).get_data()
            # y = to_categorical(seg_data, n_classes)
            data = []
            count = 0
            for scan in dict[key]:
                # print(value)
                # if 'nii.gz' in value:
                #     slice_data = nib.load(value).get_data()
                #     x = np.expand_dims(slice_data, -1).astype(np.float32)
                #     for i in range(0, len(x), batch_size):
                #         yield (x[i:i + batch_size])
                patched_slices = []
                scan_data = nib.load(scan).get_data().astype(np.float32)
                # for s in range(0, len(scan_data)):
                #     scaled = normalizer.fit_transform(scan_data[s])
                #     scaled = min_max_scaler.fit_transform(scan_data[s])
                #     scan_data[s] = scaled
                scan_data = np.expand_dims(scan_data, -1)
                # for scan in scan_data:
                #     patched_slices.extend(extract_patches(scan, (48, 31)))
                # patched_slices = np.expand_dims(np.array(patched_slices), axis=-1)
                if count == 0:
                    # data = patched_slices
                    data = scan_data
                    count = count + 1
                else:
                    data = np.concatenate((data, scan_data), axis=-1)
                    # data = np.concatenate((data, patched_slices), axis=-1)
            for i in range(0, len(data), batch_size):
                yield data[i: i + batch_size]


def no_norm_evaluate_generator(dict, batch_size):
    while True:
        for key in dict:
            count = 0
            data = []
            y = []
            # yt = []
            for scan in dict[key]:
                if '.json' in scan:
                    with open(scan, 'r') as f:
                        label = json.load(f)
                        for k in label:
                            slices = [s for (s, _) in label[k]]
                            y = np.array(slices)
                if '.json' not in scan:
                    scan_data = nib.load(scan).get_data().astype(np.float32)
                    # for s in range(0, len(scan_data)):
                    #     scaled = normalizer.fit_transform(scan_data[s])
                        # scaled = preprocessing.MinMaxScaler().fit_transform(scan_data[s])
                        # scan_data[s] = scaled
                    scan_data = np.expand_dims(scan_data, -1)
                    if count == 0:
                        data = scan_data
                        count = count + 1
                    else:
                        data = np.concatenate((data, scan_data), axis=-1)
            y = to_categorical(np.array(y), 2)
            print(data.shape)
            print(y.shape)
            exit(0)
            for j in range(0, len(data), batch_size):
                yield (data[j:j + batch_size], y[j:j + batch_size])


def seg_patch_evaluate_batch_generator(dict, batch_size):
    while True:
        for key in dict:
            count = 0
            data = []
            y = []
            yt = []
            for scan in dict[key]:
                if '.json' in scan:
                    with open(scan, 'r') as f:
                        label = json.load(f)
                        for k in label:
                            slices = [s for (s, _) in label[k]]
                            y = np.array(slices)
                    for j in range(0, len(y)):
                        patched_y = extract_patches(y[j], (48, 31))
                        yt.extend(patched_y)
                if '.json' not in scan:
                    patched_slices = []
                    scan_data = nib.load(scan).get_data().astype(np.float32)
                    for s in range(0, len(scan_data)):
                        scaled = normalizer.fit_transform(scan_data[s])
                        scan_data[s] = scaled
                    for scan in scan_data:
                        patched_slices.extend(extract_patches(scan, (48, 31)))
                    patched_slices = np.expand_dims(np.array(patched_slices), axis=-1)
                    if count == 0:
                        data = patched_slices
                        count = count + 1
                    else:
                        data = np.concatenate((data, patched_slices), axis=-1)
            yt = to_categorical(np.array(yt), 2)
            for j in range(0, len(data), batch_size):
                yield (data[j:j + batch_size], yt[j: j + batch_size])


def seg_evaluate_batch_generator(dict, batch_size):
    while True:
        for key in dict:
            count = 0
            data = []
            y = []
            yt = []
            for scan in dict[key]:
                if '.json' in scan:
                    with open(scan, 'r') as f:
                        label = json.load(f)
                        for k in label:
                            slices = [s for (s, _) in label[k]]
                            y = np.array(slices)
                if '.json' not in scan:
                    scan_data = nib.load(scan).get_data().astype(np.float32)
                    for s in range(0, len(scan_data)):
                        scaled = normalizer.fit_transform(scan_data[s])
                        # scaled = preprocessing.MinMaxScaler().fit_transform(scan_data[s])
                        scan_data[s] = scaled
                    scan_data = np.expand_dims(scan_data, -1)
                    if count == 0:
                        data = scan_data
                        count = count + 1
                    else:
                        data = np.concatenate((data, scan_data), axis=-1)
            y = to_categorical(np.array(y), 2)
            for j in range(0, len(data), batch_size):
                yield (data[j:j + batch_size], y[j:j + batch_size])


def id_evaluate_batch_generator(dict, batch_size):
    while True:
        for key in dict:
            count = 0
            data = []
            y = []
            for scan in dict[key]:
                if '.json' in scan:
                    with open(scan, 'r') as f:
                        label = json.load(f)
                        for k in label:
                            category = [c for (_, c) in label[k]]
                            y = np.array(category)
                if '.json' not in scan:
                    scan_data = nib.load(scan).get_data().astype(np.float32)
                    for s in range(0, len(scan_data)):
                        scaled = normalizer.fit_transform(scan_data[s])
                        # scaled = min_max_scaler.fit_transform(scan_data[s])
                        scan_data[s] = scaled
                    scan_data = np.expand_dims(scan_data, -1)
                    if count == 0:
                        data = scan_data
                        count = count + 1
                    else:
                        data = np.concatenate((data, scan_data), axis=-1)
            y = to_categorical(np.array(y), 2)
            for j in range(0, len(data), batch_size):
                yield (data[j:j + batch_size], y[j:j + batch_size])


def dense_crf(probs, img=None, n_iters=10, sxy_gaussian=(1, 1), compat_gaussian=4, kernel_gaussian=dcrf.DIAG_KERNEL,
              norm_gaussian=dcrf.NORMALIZE_SYMMETRIC, sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13), kernel_bilateral=dcrf.DIAG_KERNEL, norm_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    _, h, w, n_classes = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')

    d = dcrf.DenseCRF2D(w, h, n_classes)
    unaries = -np.log(probs)
    unaries = unaries.reshape((n_classes, -1))
    d.setUnaryEnergy(unaries)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=kernel_gaussian, normalization=norm_gaussian)
    if img is not None:
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral, kernel=kernel_bilateral,
                               normalization=norm_bilateral, srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
