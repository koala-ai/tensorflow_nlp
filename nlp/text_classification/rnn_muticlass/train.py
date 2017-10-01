# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os

from nlp.text_classification.rnn_muticlass.dataset.dataset import Dataset
from nlp.text_classification.rnn_muticlass.model import Model


def train_rnn_classfier(args):
    if not os.path.isdir(args.utils_dir):
        os.mkdir(args.utils_dir)

    if not os.path.isdir(args.train_dir):
        os.mkdir(args.train_dir)

    data_loader = \
        Dataset(True, args.utils_dir, args.data_dir,
                args.batch_size, args.seq_length, None, None)
    args.vocab_size = data_loader.vocab_size
    args.label_size = data_loader.label_size

    model = Model(args)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                train_loss, state, _, accuracy = sess.run(
                    [model.cost, model.final_state, model.optimizer, model.accuracy], feed_dict=feed)
                end = time.time()
                print('{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}' \
                      .format(e * data_loader.num_batches + b + 1,
                              args.num_epochs * data_loader.num_batches,
                              e + 1,
                              train_loss,
                              accuracy,
                              end - start))
                if (e * data_loader.num_batches + b + 1) % 500 == 0:
                    checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b + 1)
                    print('model saved to {}'.format(checkpoint_path))
