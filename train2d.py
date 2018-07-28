# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/16
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import argparse
import json
import tensorflow as tf
from ResNet2D import ResnetBuilder
from vgg16 import vgg_model
import vgg16
# import UNet
from load_data_2d import ScanReader
from keras.utils import to_categorical
import numpy as np
import nibabel as nib

n_classes = 4
BATCH_SIZE = 4
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Training'
INPUT_SIZE = '240, 155'
LEARNING_RATE = 1e-4
NUM_STEPS = 20000
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots/'
LABEL_CLASS = 'whole_tumor_label'


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIRECTORY, help='Path to BraTS2018 training set.')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help='Comma-separated string with height and width of images')
    parser.add_argument("--is-training", action='store_true',
                        help='Wether to update the running means and variances during training')
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help='Learning rate for training')
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help='Number of training steps')
    parser.add_argument("--random-scale", action='store_true',
                        help='Whether to randomly scale the inputs during training')
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES, help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--label-class", type=str, default=LABEL_CLASS,
                        help="Which kind of classification. Whole tumor, tumor core or cystic")

    return parser.parse_args()


def load_data(path_list):
    scans = []
    for path in path_list:
        scan_data = nib.load(path).get_data()
        for data in scan_data:
            scans.append(data)
    scans = np.expand_dims(scans, -1)
    return scans


def train_batch_generator(train_dict, label_class):
    for key in train_dict:
        data_path_list = []
        label_list = []
        for value in train_dict[key]:
            if 'nii.gz' in value:
                data_path_list.append(value)
            elif label_class in value:
                with open(label_class, 'r') as f:
                    label_dict = json.load(f)
                    label_list = list(label_dict.values())
        x = load_data(data_path_list)
        label = [l for (_, l) in label_list]
        y = label + label
        y = y + label
        y = y + label
        y = np.reshape(np.array(y), newshape=(960, ))
        for i in range(0, len(x), 10):
            yield (x[i:i+10], y[i:i+10])


def main():
    # Build ResNet-101 model
    print("Building Neural Net...")
    model = ResnetBuilder.build_resnet_101((240, 155, 1), 1)
    # unet = UNet.UNet3D((240, 240, 155, 1))
    # vgg_model = vgg16.vgg_model()
    print("Input shape", model.input_shape)
    print("Output shape", model.output_shape)
    # Compiling
    print("Compiling...")
    # model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    # unet.compile(loss="categorical_crossentropy", optimizer="sgd")
    # vgg_model.compile(loss="categorical_crossentropy", optimizer="sgd")

    # Get input and output
    print("Reading data...")
    args = get_arguments()
    scan_reader = ScanReader(args.data_dir)
    train_dict, validation_dict = scan_reader.train_dic, scan_reader.validation_dic

    # Train the model
    print("Training...")
    model.fit_generator(generator=train_batch_generator(train_dict, args.label_class), epochs=10, steps_per_epoch=19200)
    print("Done...")
    # prediction = model.predict(validation_scans)
    # print(prediction.shape())
    # print(len(prediction))


if __name__ == '__main__':
    main()
