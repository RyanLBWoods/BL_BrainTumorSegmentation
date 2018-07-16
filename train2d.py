# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/16
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import argparse
import tensorflow as tf
from ResNet2D import ResnetBuilder
from load_data_2d import ScanReader

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

    return parser.parse_args()


def load_data():
    args = get_arguments()
    scan_reader = ScanReader(args.data_dir)
    scans, labels = scan_reader.scan_img, scan_reader.label_img
    print(scans)
    print(labels)


def main():
    args = get_arguments()
    scan_reader = ScanReader(args.data_dir)
    # Get input and output
    # scans = [samples, channels, depth, height, width]
    # labels = [samples, depth, height, width]
    scans, labels = scan_reader.scan_img, scan_reader.label_img
    validation_scans, validation_labels = scan_reader.validation_scans, scan_reader.validation_labels
    # Reshape input to fit Tensorflow backend
    # [samples, depth, height, width, channel]
    scans = tf.expand_dims(scans, -1)
    validation_scans = tf.expand_dims(validation_scans, -1)
    print(scans)
    print(labels)
    print(validation_scans)
    print(validation_labels)
    # exit(0)

    # Build ResNet-101 model
    model = ResnetBuilder.build_resnet_101((240, 155, 1), 1)

    # Train
    model.compile(loss="categorical_crossentropy", optimizer="sdg")
    model.fit(scans, labels)


if __name__ == '__main__':
    main()
