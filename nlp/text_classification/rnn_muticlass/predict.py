# -*- coding: utf-8 -*-

import os
import time
import csv
import _pickle as pickle
import tensorflow as tf

from nlp.text_classification.rnn_muticlass.model import Model


def transform(text, seq_length, vocab):
    x = list(map(vocab.get, text))
    x = list(map(lambda i: i if i else 0, x))
    if len(x) >= seq_length:
        x = x[:seq_length]
    else:
        x = x + [0] * (seq_length - len(x))
    return x


def predict(args):

    with open(os.path.join(args.utils_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    vocab = dict(zip(vocab, range(1, len(vocab) + 1)))

    args.vocab_size = len(vocab) + 1
    args.label_size = len(labels)
    model = Model(args)

    with open(args.predict_file, 'r') as f:
        reader = csv.reader(f)
        texts = list(reader)

    texts = list(map(lambda i: i[0], texts))
    x = list(map(lambda i: transform(i.strip(), args.seq_length, vocab), texts))

    with tf.Session() as sess:
        saver =tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = time.time()
        results = model.predict_label(sess, labels, x)
        end = time.time()
        print('prediction costs time: ', end - start)

    with open(args.result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(texts, results))
