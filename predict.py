# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/31
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

from keras.layers import Input, Conv2DTranspose, UpSampling2D, Dense
from keras.models import load_model, Model

import load_data_2d
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
from utils import *
from load_data_2d import ScanReader

label_colors = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (176, 1226, 255), (0, 255, 0)]


n_classes = 5
BATCH_SIZE = 10
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Validation'
STEPS_PER_EPOCH = 192000 / BATCH_SIZE
VALIDATION_STEPS = 81600 / BATCH_SIZE
STEPS = 63360 / BATCH_SIZE
LABEL_CLASS = 'whole_tumor_label'
LEARNING_RATE = 0.001
NUM_EPOCH = 10


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIRECTORY, help='Path to BraTS2018 training set.')
    parser.add_argument("--steps", type=str, default=STEPS, help='Number of batches of samples to yield from generator.')
    parser.add_argument("--label-class", type=str, default=LABEL_CLASS,
                        help="Which kind of classification. Whole tumor, tumor core or cystic")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for training.")
    parser.add_argument("--nb-epoch", type=int, default=NUM_EPOCH, help="Number of epochs")

    return parser.parse_args()


def decode_prediction(prediction, num_slices=240, num_classes=5):
    n, h, w, d = prediction.shape
    outputs = np.zeros((num_slices, h, w, 1), dtype=np.uint8)
    for i in range(num_slices):
        img = Image.new('L', (h, w))
        pixel = img.load()
        for j_, j in enumerate(prediction[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixel[k_, j_] = label_colors[k]
        outputs[i] = np.array(img)
    return outputs


def main():
    args = get_arguments()
    # Get input and output
    print("Reading data...")
    scan_reader = ScanReader(args.data_dir)
    data_dict = scan_reader.mri_dic
    model_path = 'whole_tumor_label_adam_seg.h5'
    model = load_model(model_path)
    probs = model.predict_generator(generator=batch_generator(data_dict, args.batch_size, n_classes), steps=args.steps)
    print(probs.shape)
    exit(0)

    data_path = 'MICCAI_BraTS_2018_Data_Training'
    dict = load_data_2d.load_scans_dic(data_path)
    scan = []
    for key in dict:
        for value in dict[key]:
            print(value)
            data = nib.load(value).get_data()
            scan.extend(data)
            break
        break

    scan = np.expand_dims(np.array(scan), -1)
    print(scan.shape)

    # model_path = 'whole_tumor_label.h5'
    # model = load_model(model_path)
    # probs = model.predict_generator(generator=batch_generator(), steps=)
    # print(probs.shape)
    # exit(0)
    output_crf = tf.py_func(dense_crf, [probs, scan], tf.float32)
    result = np.argmax(np.squeeze(output_crf, axis=-1)).astype(np.uint8)
    print(result.shape)
    print(result[0])
    plot(scan, result)


if __name__ == '__main__':
    main()
