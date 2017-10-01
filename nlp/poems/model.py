# -*- coding: utf-8 -*-

import tensorflow as tf


class Model(object):
    def __init__(self, method,
                 input_data,
                 output_data,
                 vocab_size,
                 rnn_size,
                 num_layers,
                 batch_size,
                 learning_rate):

        self.method = method
        self.input_data = input_data
        self.output_data = output_data
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.end_points = {}

        self._build_graph()

    def _build_graph(self):

        cell = None
        if self.method == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        elif self.method == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        elif self.method == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)

        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        if self.output_data is not None:
            initial_state = cell.zero_state(self.batch_size, tf.float32)
        else:
            initial_state = cell.zero_state(1, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
                [self.vocab_size + 1, self.rnn_size], -1.0, 1.0))
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, self.rnn_size])

        weights = tf.Variable(tf.truncated_normal([self.rnn_size, self.vocab_size + 1]))
        bias = tf.Variable(tf.zeros(shape=[self.vocab_size + 1]))
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

        if self.output_data is not None:
            # output_data must be one-hot encode
            labels = tf.one_hot(tf.reshape(self.output_data, [-1]), depth=self.vocab_size + 1)
            # should be [?, vocab_size+1]

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # loss shape should be [?, vocab_size+1]
            total_loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

            self.end_points['initial_state'] = initial_state
            self.end_points['output'] = output
            self.end_points['train_op'] = train_op
            self.end_points['total_loss'] = total_loss
            self.end_points['loss'] = loss
            self.end_points['last_state'] = last_state
        else:
            prediction = tf.nn.softmax(logits)

            self.end_points['initial_state'] = initial_state
            self.end_points['last_state'] = last_state
            self.end_points['prediction'] = prediction