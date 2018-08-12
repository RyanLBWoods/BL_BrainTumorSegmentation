# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/6
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

from keras.layers import Input, Conv2DTranspose, UpSampling2D, Dense
from keras.models import load_model, Model
from keras import optimizers
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn import preprocessing
import load_data_2d
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
from utils import *
from load_data_2d import ScanReader
from ResNet2D import ResnetBuilder
from BilinearUpSampling import BilinearUpSampling2D
import pydensecrf.densecrf as dcrf

n_classes = 2
BATCH_SIZE = 10
TESTING_DATA_DIRECTORY = 'MICCAI_BraTS_Testing'
STEPS = 14400 / BATCH_SIZE
LABEL_CLASS = 'whole_tumor_label'
LEARNING_RATE = 0.001


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TESTING_DATA_DIRECTORY, help='Path to BraTS testing set.')
    parser.add_argument("--steps", type=str, default=STEPS,
                        help='Number of batches of samples to yield from generator.')
    parser.add_argument("--label-class", type=str, default=LABEL_CLASS,
                        help="Which kind of classification. Whole tumor, tumor core or cystic")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for training.")

    return parser.parse_args()


def main():
    args = get_arguments()
    # Get input and output
    print("Reading data...")
    scan_reader = ScanReader(args.data_dir)
    data_dict = scan_reader.mri_dic
    adam = optimizers.Adam(lr=args.learning_rate, decay=0.0001)

    # model_path = 'wt_only_nb-01-0.10.hdf5'
    # model_path = 'wt_minmax-05-0.30.hdf5'
    # model_path = 'wt_l2-12-0.35.hdf5'
    # model_path = 'wt_skpatches-02-0.42.hdf5'
    # model_path = 'wt_id-13-0.76.hdf5'
    model_path = 'whole_tumor_label_only_adam.h5'
    print("Loading model...")
    trained_model = load_model(model_path,
                               custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D,
                                               'dice_coef_loss': dice_coef_loss,
                                               'dice_coef': dice_coef})
    weights = trained_model.get_weights()
    model = ResnetBuilder.build_resnet_101((240, 155, 4), 2)
    model.set_weights(weights)
    model.compile(loss=dice_coef_loss, optimizer=adam, metrics=[dice_coef])
    print("Predicting...")
    # probs = trained_model.predict_generator(generator=test_batch_generator(d, args.batch_size), steps=24)
    # probs = trained_model.evaluate_generator(generator=seg_patch_evaluate_batch_generator(data_dict, args.batch_size), steps=24)
    probs = model.evaluate_generator(generator=no_norm_evaluate_generator(data_dict, args.batch_size), steps=14400/10)
    with open('no_norm_eva.txt', 'w') as nf:
        nf.write(probs)
    # exit(0)
    print(probs.shape)
    exit(0)
    result = []
    for prob in range(0, len(probs)):
        probs[prob] = dense_crf(np.expand_dims(probs[prob], axis=0))
        result.append(probs[prob])
    result = np.argmax(np.array(result), axis=-1).astype(np.uint8)
    print(result.shape)
    if 1 in result:
        print('111111')
    img = []
    for i in range(0, len(result), 25):
        patches = result[i: i + 25]
        image = reconstruct_image(patches, (240, 155))
        img.append(image)
    img = np.array(img)
    print(img.shape)
    seg = nib.load(
        'MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_seg.nii.gz').get_data()
    # print(scan1.shape)
    scan1 = np.expand_dims(
        nib.load(
            'MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_t1ce.nii.gz').get_data(),
        axis=-1)
    scan1 = scan1.squeeze()
    for i in range(0, len(result)):
        plt.imshow(scan1[i], cmap='gray')
        plt.imshow(img[i], cmap=pred_colormap(), alpha=0.4)
        plt.imshow(seg[i], cmap=colormap(), alpha=0.5)
        plt.show()
    exit(0)
    # data = []
    # scan = np.expand_dims(
    #     nib.load('MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_t1ce.nii.gz').get_data(),
    #     axis=-1)
    # data = scan
    scan1 = np.expand_dims(
        nib.load('MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_flair.nii.gz').get_data(),
        axis=-1)
    # data = np.concatenate((data, scan1), axis=-1)
    # scan = np.expand_dims(
    #     nib.load('MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_t1.nii.gz').get_data(), axis=-1)
    # data = np.concatenate((data, scan), axis=-1)
    # scan = np.expand_dims(
    #     nib.load('MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_t2.nii.gz').get_data(), axis=-1)
    # data = np.concatenate((data, scan), axis=-1)
    # print(data.shape)
    # probs = model.predict(data)
    # print(probs.shape, 'ppppp')
    # result = []
    # for prob in probs:
    #     r = dense_crf(probs)
    #     print(r.shape, 'ccccc')
    #     result.append(np.squeeze(r))
    # print(len(result))
    # result = np.array(result)
    # print(result.shape)
    # results = np.argmax(result, axis=-1).astype(np.uint8)
    # print(results.shape, 'rrrrr')
    # seg = nib.load('MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_seg.nii.gz').get_data()
    # print(scan1.shape)
    scan1 = scan1.squeeze()
    scan1 = scan1.astype(np.float32)
    print(scan1[120][70])
    for s in range(0, len(scan1)):
        scaled = preprocessing.MinMaxScaler().fit_transform(scan1[s])
        scan1[s] = scaled
    print(scan1[120][70])
    # for i in range(0, len(results)):
    #     plt.imshow(scan1[i], cmap='gray')
    #     plt.imshow(results[i], cmap=pred_colormap(), alpha=0.4)
    #     plt.imshow(seg[i], cmap=colormap(), alpha=0.3)
    #     plt.show()
    # exit(0)

    # model_path = 'whole_tumor_label.h5'
    # model = load_model(model_path)
    # probs = model.predict_generator(generator=test_batch_generator(data_dict, args.batch_size), steps=1000)
    # print(probs.shape)
    exit(0)
    output_crf = tf.py_func(dense_crf, [probs, scan], tf.float32)
    result = np.argmax(np.squeeze(output_crf, axis=-1)).astype(np.uint8)
    print(result.shape)
    print(result[0])
    plot(scan, result)


if __name__ == '__main__':
    main()
