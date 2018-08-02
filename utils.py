# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import numpy as np
from PIL import Image
import tensorflow as tf
# from keras.engine.topology import Layer
import pydensecrf.densecrf as dcrf
from matplotlib import colors
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import nibabel as nib

n_classes = 5  # Whole tumor, tumor core, enhancing tumor, cystic/necrotic component, non-tumor part
label_colors = [(255, 255, 0), (255, 0, 0), (176, 1226, 255), (0, 255, 0)]
# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def colormap():
    map_list = ['#000000', '#FF0000', '#008B00', '#B0E2FF', '#FFFF00']
    return colors.ListedColormap(map_list, 'indexed')


def plot(scans, segs):
    for i in range(len(segs)):
        plt.imshow(scans[i], cmap='gray')
        plt.imshow(segs[i], cmap=colormap(), alpha=0.3)
        plt.show()


def batch_generator(dict, batch_size, n_classes):
    while True:
        for key in dict:
            seg_data = nib.load(key).get_data()
            y = to_categorical(seg_data, n_classes)
            count = 0
            data = []
            for scan in dict[key]:
                if '.json' not in scan:
                    scan_data = np.expand_dims(nib.load(scan).get_data(), axis=-1)
                    if count == 0:
                        data = scan_data
                    else:
                        data = np.concatenate((data, scan_data), axis=-1)
                    count = count + 1
                # for value in dict[key]:
                #     if 'nii.gz' in value:
                #         slice_data = nib.load(value).get_data()
            x = data.astype(np.float32)
            for i in range(0, len(x), batch_size):
                yield (x[i:i + batch_size], y[i:i + batch_size])


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
