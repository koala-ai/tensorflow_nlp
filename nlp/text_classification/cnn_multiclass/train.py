# -*- coding: utf-8 -*-

import tensorflow as tf
import time, os
from nlp.text_classification.cnn_multiclass import model as cnn_model
from nlp.text_classification.cnn_multiclass.dataset.dataset import Dataset


def train_cnn_classfier(args):
    if not os.path.isdir(args.utils_dir):
        os.mkdir(args.utils_dir)

    if not os.path.isdir(args.train_dir):
        os.mkdir(args.train_dir)

    data_loader =\
        Dataset(True, args.utils_dir, args.data_dir, args.batch_size, args.sentence_length, None, None)

    args.vocab_size = data_loader.vocab_size
    args.label_size = data_loader.label_size
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        model = cnn_model.CNNModel(config=args)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        for e in range(args.num_epochs):
            print("Start epoch:" + str(e))
            data_loader.reset_batch_pointer()

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.x_in: x, model.y_in: y, model.keep_prob: 0.5}
                _, global_step, accuracy, loss = sess.run(
                    [model.optimizer, model.global_step, model.accuracy, model.loss], feed_dict=feed)
                end = time.time()
                print('{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}' \
                    .format(e * data_loader.num_batches + b + 1,
                            args.num_epochs * data_loader.num_batches,
                            e + 1,
                            loss,
                            accuracy,
                            end - start))
                if global_step % 500 == 0:
                    checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print('model saved to {}'.format(checkpoint_path))
