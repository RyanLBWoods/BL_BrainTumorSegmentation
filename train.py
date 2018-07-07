# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/6
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from ResNet import ResNet
from data_reader import ScanReader

BATCH_SIZE = 4
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Training'
INPUT_SIZE = '240, 155'
LEARNING_RATE = 1e-4
NUM_STEPS = 20000



def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIRECTORY, help='Path to BraTS2018 training set.')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE, help='Comma-separated string with height and width of images')
    parser.add_argument("--is-training", action='store_true', help='Wether to update the running means and variances during training')
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help='Learning rate for training')
    parser.add_argument("--random-scale", action='store_true', help='Whether to randomly scale the inputs during training')

    return parser.parse_args()


def main():
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    coord = tf.train.Coordinator()

    # Load data reader
    with tf.name_scope('create_inputs'):
        scan_reader = ScanReader(args.data_dir, input_size, args.random_scale, coord)
        scan_batch, label_batch = scan_reader.dequeue(args.batch_size)

    # Build neural net
    net = ResNet({'data': scan_batch}, is_training=args.is_training)