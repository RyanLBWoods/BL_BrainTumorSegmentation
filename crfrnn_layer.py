# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/4
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import numpy as np
import tensorflow as tf
# from keras.engine.topology import Layer
import pydensecrf.densecrf as dcrf

n_classes = 4

def dense_crf(probs, img=None, n_iters=10, sxy_gaussian=(1, 1), compat_gaussian=4, kernel_gaussian=dcrf.DIAG_KERNEL,
              norm_gaussian=dcrf.NORMALIZE_SYMMETRIC, sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13), kernel_bilateral=dcrf.DIAG_KERNEL, norm_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')

    d = dcrf.DenseCRF2D(w, h, n_classes)
    U = -np.log(probs)
    U = U.reshap((n_classes, -1))
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=kernel_gaussian, normalization=norm_gaussian)
    if img is not None:
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral, kernel=kernel_bilateral,
                               normalization=norm_bilateral, srgb=sxy_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
