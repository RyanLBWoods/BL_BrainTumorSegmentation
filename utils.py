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

n_classes = 2  # Whole tumor, tumor core, enhancing tumor, cystic/necrotic component, non-tumor part
label_colors = [(255, 255, 0), (255, 0, 0), (176, 1226, 255), (0, 255, 0)]
# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def colormap():
    map_list = ['#000000', '#FF0000', '#008B00', '#B0E2FF', '#FFFF00']
    return colors.ListedColormap(map_list, 'indexed')


def pred_colormap():
    map_list = ['#000000', '#B0E2FF']
    return colors.ListedColormap(map_list, 'indexed')


def plot(scans, segs):
    for i in range(len(segs)):
        plt.imshow(scans[i], cmap='gray')
        plt.imshow(segs[i], cmap=colormap(), alpha=0.3)
        plt.show()


# def extract_patches(a, patch_size):
#     m, n = a.shape
#     b0, b1 = patch_size
#     return a.reshape(m // b0, b0, n // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)


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
            y = []
            ids = []
            for scan in dict[key]:
                if label_class in scan:
                    with open(scan, 'r') as f:
                        label_data = json.load(f)
                        for k in label_data:
                            slices = [s for (s, _) in label_data[k]]
                            labels = [l for (_, l) in label_data[k]]
                            for index in range(0, len(labels)):
                                if labels[index] == 1:
                                    ids.append(index)
                                    y.append(slices[index])
            y = np.array(y)
            x_train = []
            y_train = []
            c_count = 0
            print(len(ids))
            for scan in dict[key]:
                if '.json' not in scan:
                    scan_data = nib.load(scan).get_data()
                    x_c = []
                    x = []
                    for i in ids:
                        x.append(scan_data[i])
                    x = np.array(x)
                    y = np.array(y)
                    for j in range(0, len(y)):
                        x_patches = extract_patches_2d(x[j], (48, 31))
                        y_patches = extract_patches_2d(y[j], (48, 31))
                        for yi in range(0, len(y_patches)):
                            if 1 in y_patches[yi]:
                                x_c.append(x_patches[yi])
                                y_train.append(y_patches[yi])
                    x_c = np.expand_dims(np.array(x_c), axis=-1)
                    if c_count == 0:
                        x_train = x_c
                        c_count = c_count + 1
                    else:
                        x_train = np.concatenate((x_train, x_c), axis=-1)
            y_train = to_categorical(y_train[0:len(x_train)], n_classes)
            for i in range(0, len(x_train), batch_size):
                yield (x_train[i:i + batch_size], y_train[i:i + batch_size])


def test_batch_generator(dict, batch_size):
    while True:
        for key in dict:
            count = 0
            data = []
            patched_slices = []
            for scan in dict[key]:
                scan_data = nib.load(scan).get_data()
                for slice in scan_data:
                    scan_patches = extract_patches_2d(slice, (48, 31))
                    patched_slices.extend(scan_patches)
                patched_slices = np.expand_dims(patched_slices, axis=-1)
                if count == 0:
                    data = patched_slices
                    count = count + 1
                else:
                    data = np.concatenate((data, patched_slices), axis=-1)
                patched_slices = []
            x = data.astype(np.float32)
            for i in range(0, len(x), batch_size):
                yield (x[i:i + batch_size])


def decode_labels(label, num_images):
    n, h, w, c = label.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 1), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('L', (h, w))
        pixels = img.load()
        for j_, j in enumerate(label[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < n_classes:
                    pixels[k_, j_] = label_colors[k]
        outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch):
    """[285, 4, 240, 240, 155]
       [48000, 240, 155, 1]
    """

    with tf.name_scope('label_encode'):
        input_batch = tf.squeeze(input_batch)
        input_batch = tf.one_hot(input_batch, n_classes)

    return input_batch


def inverse_preprocess(imgs, num_images):
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 1), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
    return outputs


def dense_crf(probs, img=None, n_iters=10, sxy_gaussian=(1, 1), compat_gaussian=4, kernel_gaussian=dcrf.DIAG_KERNEL,
              norm_gaussian=dcrf.NORMALIZE_SYMMETRIC, sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13), kernel_bilateral=dcrf.DIAG_KERNEL, norm_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')

    d = dcrf.DenseCRF2D(w, h, n_classes)
    unaries = -np.log(probs)
    unaries = unaries.reshap((n_classes, -1))
    d.setUnaryEnergy(unaries)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=kernel_gaussian, normalization=norm_gaussian)
    if img is not None:
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral, kernel=kernel_bilateral,
                               normalization=norm_bilateral, srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
