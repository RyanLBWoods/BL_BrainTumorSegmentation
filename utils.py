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

n_classes = 4
# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def decode_labels(label, num_images):
    n, h, w, c = label.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 1), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('L', (h, w))
        pixels = img.load()
        # for j_, j in enumerate()
        outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch, new_size):
    with tf.name_scope('label_encode'):
        input_batch = tf.squeeze(input_batch, squeeze_dims=[1])

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
