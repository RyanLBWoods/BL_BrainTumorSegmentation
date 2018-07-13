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

from ResNet101 import resnet101_model
from ResNet import ResNetModel
from data_reader import ScanReader
from utils import decode_labels, prepare_label, inverse_preprocess

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


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    coord = tf.train.Coordinator()

    # Load data reader
    with tf.name_scope('create_inputs'):
        scan_reader = ScanReader(args.data_dir, input_size, args.random_scale, coord)
        scans, labels = scan_reader.scan_img, scan_reader.label_img
        print(scans)
        print(labels)
        scans = tf.reshape(scans, [192000, 240, 155, 1])
        labels = tf.reshape(labels, [48000, 240, 155, 1])
        print(scans)
        print(labels)
    # exit(0)
    # Build neural net
    net = ResNetModel({'data': scans}, is_training=args.is_training)
    print(net)
    exit(0)
    # Predictions
    output = net.layers['fc_voc12']
    restore_var = tf.global_variables()
    trainable = tf.trainable_variables()

    # prediction = tf.reshape(output, [-1, n_classes])
    # label_proc = prepare_label(labels, tf.pack(output.get_shape()[1: 3, ]))
    # gt = tf.reshape(label_proc, [-1, n_classes])

    # Pixel-wise softmax loss
    loss = tf.nn.softmax_cross_entropy_with_logits(output, tf.shape(scans)[1:3, ])
    reduced_loss = tf.reduce_mean(loss)

    # Process predictions
    output_up = tf.image.resize_bilinear(output, tf.shape(scans)[1:3, ])
    output_up = tf.argmax(output_up, dimension=1)
    pred = tf.expand_dims(output_up, dim=1)

    # Image summary
    scans_summary = tf.py_func(inverse_preprocess, [scans, args.save_num_images], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [labels, args.save_num_images], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images], tf.uint8)

    total_summary = tf.summary.image('images', tf.concat(2, [scans_summary, labels_summary, preds_summary]),
                                     max_outputs=args.save_num_images)
    summary_writer = tf.summary.FileWriter(args.snapshot_dir)

    # Define loss and optimisation parameters
    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    optim = optimiser.minimize(reduced_loss, var_list=trainable)

    # Set up tf session and initialize variables
    config = tf.ConfigProto()
    config.gpu_option.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=40)

    # Load variables if the checkpoint is provided
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)

    # Start queue threads
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps
    for step in range(args.num_steps):
        start_time = time.time()

        if step % args.save_pred_every == 0:
            loss_value, scans, labels, preds, summary, _ = sess.run(
                [reduced_loss, scans, labels, pred, total_summary, optim])
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([reduced_loss, optim])
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
