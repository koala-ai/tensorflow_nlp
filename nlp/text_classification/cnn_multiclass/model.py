# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class CNNModel(object):
    def __init__(self, config):
        self.config = config
        self.x_in = tf.placeholder(tf.int64, shape=[None, config.sentence_length])
        self.y_in = tf.placeholder(tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.embeddings = tf.Variable(
            tf.random_uniform([config.vocab_size, config.vector_size], -1.0, 1.0))

        self.loss, self.accuracy, self.scores = self.build_model()

        self.global_step = tf.Variable(0)
        self.learning_rate = \
            tf.train.exponential_decay(1e-2, self.global_step, config.num_epochs, 0.99, staircase=True)  # 学习率递减
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)\
            .minimize(self.loss, global_step=self.global_step)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool(self, x, filter_h):
        return tf.nn.max_pool(x, ksize=[1, self.config.img_h - filter_h + 1, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    def build_model(self):
        config = self.config
        # Embedding layer===============================
        x_image_tmp = tf.nn.embedding_lookup(self.embeddings, self.x_in)
        x_image = tf.expand_dims(x_image_tmp, -1)  # 单通道

        with tf.variable_scope('cnn_conv', reuse=None):
            h_conv = []
            for filter_h in config.filter_hs:
                filter_shape = [filter_h, config.vector_size, 1, config.num_filters]
                W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
                h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
                h_conv.append(h_conv1)

        h_pool_output = []
        for h_conv1, filter_h in zip(h_conv, config.filter_hs):
            h_pool1 = self.max_pool(h_conv1, filter_h)  # 输出szie:1
            h_pool_output.append(h_pool1)

        l2_reg_lambda = 0.001
        num_filters_total = config.num_filters * len(config.filter_hs)
        h_pool = tf.concat(h_pool_output, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)
        with tf.variable_scope('cnn_score', reuse=None):
            W = tf.Variable(tf.truncated_normal([num_filters_total, config.label_size], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[config.label_size]), name="b")
            l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self.y_in)
        loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        prediction = tf.argmax(scores, 1)
        correct_prediction = tf.equal(prediction, self.y_in)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return loss, accuracy, scores

    def predict_label(self, sess, labels, text):
        x = np.array(text)
        feed = {self.x_in: x, self.keep_prob: 1.0}
        probs = sess.run([self.scores], feed_dict=feed)

        results = np.argmax(probs[0], 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels
