# -*- coding:utf-8 -*-

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from nlp.textsum import model as seq2seq_model
from nlp.textsum.dataset import data_utils, dataset


def read_data(buckets, source_path, target_path):

    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 10000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def train(args, steps_per_checkpoint=10):
    # Prepare Headline data.
    print("Preparing Headline data in %s" % args.data_dir)
    src_train, dest_train, src_dev, dest_dev, vocab_size =\
        data_utils.prepare_data(args.data_dir, args.utils_dir)

    args.vocab_size = vocab_size

    # device config for CPU usage
    config = tf.ConfigProto(device_count={"CPU": 4},  # limit to 4 CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=2)  # n threads parallel for ops

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (args.num_layers, args.hidden_size))
        model = seq2seq_model.create_model(sess, args.train_dir, args, False)

        # Read data into buckets and compute their sizes.
        dev_set = read_data(args.buckets, src_dev, dest_dev)
        train_set = read_data(args.buckets, src_train, dest_train)


        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(args.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        trainbuckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                              for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step<args.num_steps:

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(trainbuckets_scale))
                             if trainbuckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights =\
                dataset.DataSet(args, train_set).next_batch(bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(args.train_dir, "headline_large.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(args.buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights =\
                        dataset.DataSet(args, dev_set).next_batch(bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()