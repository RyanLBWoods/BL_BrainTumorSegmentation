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
from utils import *
from keras.callbacks import TensorBoard, ModelCheckpoint

n_classes = 2
BATCH_SIZE = 10
TRAINING_DATA_DIRECTORY = 'MICCAI_BraTS_2018_Data_Training'
STEPS_PER_EPOCH = 300000 / BATCH_SIZE
VALIDATION_STEPS = 127500 / BATCH_SIZE
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


def main():
    # Get arguments
    args = get_arguments()

    # Build ResNet-101 model
    print("Building Neural Net...")
    model = ResnetBuilder.build_resnet_101((240, 155, 4), 2)

    # Set learning rate
    # sgd = optimizers.SGD(lr=args.learning_rate, momentum=0.9, decay=0, nesterov=False)
    adam = optimizers.Adam(lr=args.learning_rate, decay=0.0001)
    # Compiling
    print("Compiling...")
    model.compile(loss=dice_coef_loss, optimizer='adam', metrics=[dice_coef])
    print(model.summary())
    with open('ms_img_norm.txt', 'w') as ms:
        model.summary(print_fn=lambda x: ms.write(x + '\n'))

    # Get input and output
    print("Reading data...")
    scan_reader = ScanReader(args.data_dir)
    train_dict, validation_dict = scan_reader.train_dic, scan_reader.validation_dic

    # Train the model
    print("Training...")
    tensorboard = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=False)
    checkpoint_path = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1)
    history = model.fit_generator(generator=batch_generator(train_dict, args.batch_size, n_classes, args.label_class),
                                  epochs=args.nb_epoch, steps_per_epoch=args.steps_per_epoch)
    # history = model.fit_generator(generator=batch_generator(train_dict, args.batch_size, n_classes, args.label_class),
    #                               epochs=args.nb_epoch, steps_per_epoch=args.steps_per_epoch,
    #                               validation_data=batch_generator(validation_dict, args.batch_size, n_classes, args.label_class),
    #                               validation_steps=args.validation_steps, callbacks=[tensorboard, checkpoint])
    print("Done...")
    print("Saving Model...")
    model.save(args.label_class + "only_adam.h5")
    with open('log_adam_wt_only.txt', 'w') as adam_log:
        adam_log.write(str(history.history))


if __name__ == '__main__':
    main()
