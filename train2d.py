# -*- coding: utf-8 -*-
#
# @Date    : 2018/7/16
# @Author  : Bin LIN
# Copyright (c) 2017-2018 University of St. Andrews, UK. All rights reserved
#

import argparse
import json
from ResNet2D import ResnetBuilder
from load_data_2d import ScanReader
import numpy as np
import nibabel as nib
from keras import optimizers
from keras.utils import to_categorical

n_classes = 2
BATCH_SIZE = 10
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Training'
STEPS_PER_EPOCH = 192000 / BATCH_SIZE
VALIDATION_STEPS = 81600 / BATCH_SIZE
LABEL_CLASS = 'whole_tumor_label'
LEARNING_RATE = 0.001
NUM_EPOCH = 10


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of scans sent to network in one step.')
    parser.add_argument("--data-dir", type=str, default=TRAINING_DATA_DIRECTORY, help='Path to BraTS2018 training set.')
    parser.add_argument("--steps-per-epoch", type=str, default=STEPS_PER_EPOCH, help='Steps per epoch.')
    parser.add_argument("--validation-steps", type=str, default=VALIDATION_STEPS, help='Validation steps.')
    parser.add_argument("--label-class", type=str, default=LABEL_CLASS,
                        help="Which kind of classification. Whole tumor, tumor core or cystic")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for training.")
    parser.add_argument("--nb-epoch", type=int, default=NUM_EPOCH, help="Number of epochs")

    return parser.parse_args()


def load_data(path_list):
    scans = []
    for path in path_list:
        scan_data = nib.load(path).get_data()
        scans.extend(scan_data)
    scans = np.expand_dims(scans, -1).astype(np.float32)
    return scans


def batch_generator(dict, label_class, batch_size):
    while True:
        for key in dict:
            data_path_list = []
            seg_data = nib.load(key).get_data()
            one_hot_seg = to_categorical(seg_data, 5)
            for value in dict[key]:
                if 'nii.gz' in value:
                    data_path_list.append(value)
            x = load_data(data_path_list)
            y = np.concatenate((one_hot_seg, one_hot_seg, one_hot_seg, one_hot_seg))
            for i in range(0, len(x), batch_size):
                yield (x[i:i + batch_size], y[i:i + batch_size])


def main():
    # Get arguments
    args = get_arguments()

    # Build ResNet-101 model
    print("Building Neural Net...")
    model = ResnetBuilder.build_resnet_101((240, 155, 1), 5)

    # Set learning rate
    # sgd = optimizers.SGD(lr=args.learning_rate, momentum=0.9, decay=0, nesterov=False)
    # adam = optimizers.Adam(lr=args.learning_rate)
    # Compiling
    print("Compiling...")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    with open('model_summary.txt', 'w') as ms:
        model.summary(print_fn=lambda x: ms.write(x + '\n'))
        # ms.write(str(model.summary()))

    # Get input and output
    print("Reading data...")
    scan_reader = ScanReader(args.data_dir)
    train_dict, validation_dict = scan_reader.train_dic, scan_reader.validation_dic

    # Train the model
    print("Training...")
    history = model.fit_generator(generator=batch_generator(train_dict, args.label_class, args.batch_size),
                                  epochs=args.nb_epoch, steps_per_epoch=args.steps_per_epoch)
    # history = model.fit_generator(generator=batch_generator(train_dict, args.label_class, args.batch_size),
    #                               epochs=args.nb_epoch, steps_per_epoch=args.steps_per_epoch,
    #                               validation_data=batch_generator(validation_dict, args.label_class, args.batch_size),
    #                               validation_steps=args.validation_steps)
    print("Done...")
    print("Saving Model...")
    model.save(args.label_class + "_adam_seg.h5")
    with open('log_adam_10_10_seg.txt', 'w') as adam_log:
        adam_log.write(str(history.history))
    #
    # print("Evaluating...")
    # error = model.evaluate_generator(generator=batch_generator(validation_dict, args.label_class, args.batch_size),
    #                                  steps=args.validation_steps)
    # with open('log_adam_val_seg.txt', 'w') as val_error:
    #     val_error.write(str(error))


if __name__ == '__main__':
    main()
