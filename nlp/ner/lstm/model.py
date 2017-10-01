#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import tensorflow as tf
import os


class NERTagger(object):
    """The NER Tagger Model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.seq_length = seq_length = config.seq_length
        size = config.hidden_size
        vocab_size = config.vocab_size
        tag_size = config.tag_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self._targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        # Check if Model is Training
        self.is_training = is_training

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("ner_lstm"):
            for time_step in range(seq_length):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, tag_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [tag_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(self._targets, [-1])],
            weights=[tf.ones([batch_size * seq_length], dtype=tf.float32)])

        # Fetch Reults in session.run()
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._logits = logits

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def logits(self):
        return self._logits

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    def predict_tag(self, sess, tags, text):
        x = np.array(text)
        feed = {self._input_data: x}
        logits = sess.run([self._logits], feed_dict=feed)

        results = np.argmax(logits, 1)
        id2labels = dict(zip(tags.values(), tags.keys()))
        labels = map(id2labels.get, results)
        return labels


def run(session, model, dataset, eval_op, ner_train_dir, epoch):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    step = 0
    while dataset.hasNext():
        step = step + 1
        (x, y) = dataset.nextBatch(model.batch_size)
        fetches = [model.cost, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        cost, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.seq_length

    # Save Model to CheckPoint when is_training is True
    if model.is_training:
        checkpoint_path = os.path.join(ner_train_dir, "lstm/model.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))

    return np.exp(costs / iters)