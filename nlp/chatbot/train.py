# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import math
import sys
import os
import numpy as np
from nlp.chatbot.dataset import data_utils
from nlp.chatbot import model as s2s_model


def train(args):

    print('准备数据')
    bucket_dbs = data_utils.read_bucket_dbs(args.buckets_dir)
    bucket_sizes = []
    buckets = data_utils.buckets
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    with tf.Session() as sess:
        model = s2s_model.create_model(sess, False)
        sess.run(tf.initialize_all_variables())
        buckets_scale = [
            sum(bucket_sizes[:i + 1]) / total_size
            for i in range(len(bucket_sizes))
        ]
        metrics = '  '.join([
            '\r[{}]',
            '{:.1f}%',
            '{}/{}',
            'loss={:.3f}',
            '{}/{}'
        ])
        bars_max = 20
        for epoch_index in range(1, args.num_epoch + 1):
            print('Epoch {}:'.format(epoch_index))
            time_start = time.time()
            epoch_trained = 0
            batch_loss = []
            while True:
                random_number = np.random.random_sample()
                bucket_id = min([
                    i for i in range(len(buckets_scale))
                    if buckets_scale[i] > random_number
                ])
                data, data_in = model.get_batch_data(
                    bucket_dbs,
                    bucket_id
                )
                encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                    bucket_dbs,
                    bucket_id,
                    data
                )
                _, step_loss, output = model.step(
                    sess,
                    encoder_inputs,
                    decoder_inputs,
                    decoder_weights,
                    bucket_id,
                    False
                )
                epoch_trained += args.batch_size
                batch_loss.append(step_loss)
                time_now = time.time()
                time_spend = time_now - time_start
                time_estimate = time_spend / (epoch_trained / args.num_per_epoch)
                percent = min(100, epoch_trained / args.num_per_epoch) * 100
                bars = math.floor(percent / 100 * bars_max)
                sys.stdout.write(metrics.format(
                    '=' * bars + '-' * (bars_max - bars),
                    percent,
                    epoch_trained, args.num_per_epoch,
                    np.mean(batch_loss),
                    data_utils.time(time_spend), data_utils.time(time_estimate)
                ))
                sys.stdout.flush()
                if epoch_trained >= args.num_per_epoch:
                    break
            print('\n')

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        model.saver.save(sess, os.path.join(args.model_dir, args.model_name))
