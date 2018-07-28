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
from load_data_2d import ScanReader
import numpy as np
import nibabel as nib

n_classes = 2
BATCH_SIZE = 10
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Training'
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH = 19200
LABEL_CLASS = 'whole_tumor_label'


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIRECTORY, help='Path to BraTS2018 training set.')
    parser.add_argument("--steps-per-epoch", type=str, default=STEPS_PER_EPOCH, help='Steps per epoch.')
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


def train_batch_generator(train_dict, label_class, batch_size):
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
        y = np.reshape(np.array(y), newshape=(960,))
        for i in range(0, len(x), batch_size):
            yield (x[i:i + batch_size], y[i:i + batch_size])


def main():
    # Build ResNet-101 model
    print("Building Neural Net...")
    model = ResnetBuilder.build_resnet_101((240, 155, 1), 1)
    # vgg_model = vgg16.vgg_model()
    print("Input shape", model.input_shape)
    print("Output shape", model.output_shape)
    # Compiling
    print("Compiling...")
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
    model.fit_generator(generator=train_batch_generator(train_dict, args.label_class, args.batch_size), epochs=10,
                        steps_per_epoch=args.steps_per_epoch)
    print("Done...")
    # prediction = model.predict(validation_scans)
    # print(prediction.shape())
    # print(len(prediction))


if __name__ == '__main__':
    main()
