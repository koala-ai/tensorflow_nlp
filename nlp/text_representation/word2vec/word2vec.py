# -*- coding: utf-8 -*-

import sys
import argparse
from glob import glob
from datetime import datetime
import time
import threading

import numpy as np
import tensorflow as tf

UNK_ID = 1

PRELOAD_LINES = 500
BATCH_LINES = 50
MIN_AFTER_DEQUEUE = 200


def read_word_freq(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    lines = tf.string_split([value], "\n")

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run([lines])
        lines_eval = lines.eval()
        result = []
        for line in lines_eval.values:
            s = line.split()
            result.append((s[0], int(s[1])))
        coord.request_stop()
        coord.join(threads)
    return result


class Word2Vec(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session

        opts = options

        word_freq = read_word_freq(opts.train_dir + "/word_freq")

        self._corpus_size = sum([c for (_, c) in word_freq])

        word_freq = [(w, c) for w, c in word_freq if c >= opts.min_count]
        id2word = ['_PAD', '_UNK'] + [word for word, _ in word_freq]
        word2id = {w: i for i, w in enumerate(id2word)}
        self._id2word = id2word
        self._word2id = word2id

        unk_count = self._corpus_size - sum([c for _, c in word_freq])
        self._word_freq = [1, unk_count] + [count for _, count in word_freq]
        self._vocab_size = len(self._id2word)

        p_sample = []
        subsample = opts.subsample
        for i in range(self._vocab_size):
            p = subsample * self._corpus_size / self._word_freq[i]
            p_sample.append(np.sqrt(p) + p)

        self._p_sample = p_sample

        print('corpus size: %s, freq vocab size %s' %
              (self._corpus_size, self._vocab_size))

    def init(self):
        self.build_graph()
        self._valid_words = self.pick_valid_words()

    def load_tfrecord(self):
        opts = self._options
        file_names = glob(opts.train_dir + '/output.tfrecord')
        file_queue = tf.train.string_input_producer(file_names,
                                                    num_epochs=opts.epochs_to_train)
        reader = tf.TFRecordReader()
        _, record_string = reader.read(file_queue)
        features = {'sentence': tf.VarLenFeature(tf.int64)}
        one_line_example = tf.parse_single_example(record_string, features=features)
        capacity = PRELOAD_LINES
        batch_lines = tf.train.batch(one_line_example,
                                     batch_size=BATCH_LINES,
                                     capacity=capacity,
                                     num_threads=opts.io_threads)
        corpus_slice = batch_lines['sentence'].values
        return corpus_slice

    def skip_gram(self, corpus_slice):
        opts = self._options
        window_size = opts.window_size
        span = 2 * window_size + 1
        _size = tf.shape(corpus_slice)[0]
        corpus = tf.cond(_size >= span,
                         lambda : tf.identity(corpus_slice),
                         lambda : tf.pad(corpus_slice, [[0, span - _size]]))
        size = tf.shape(corpus)[0]

        index = tf.fill([size - span, span], 1)
        index = tf.concat([[tf.range(0, span)], index], axis=0)
        index = tf.scan(lambda a, b: a + b + size, index)
        index = tf.reshape(index, [-1])

        M = tf.tile(corpus, [size - span + 1])
        L = tf.cast(size * (size - span + 1), tf.int64)
        sparse_index = tf.reshape(tf.range(0, L), [-1, 1])
        M = tf.SparseTensor(sparse_index, M, [L])

        X = tf.gather(M.values, index)
        X = tf.reshape(X, [size - span + 1, span])

        examples = tf.slice(X, [0, window_size], [size - span + 1, 1])
        examples = tf.reshape(examples, [-1])
        keep_pro = tf.cast(tf.gather(self._p_sample, examples), tf.float32)
        sub_sample_mask = tf.random_uniform([size - span + 1]) <= keep_pro
        examples = tf.boolean_mask(examples, sub_sample_mask)
        examples = tf.tile(tf.reshape(examples, [-1, 1]), [1, window_size * 2])
        examples = tf.reshape(examples, [-1])

        X_left = tf.slice(X, [0, 0], [size - span + 1, window_size])
        X_left = tf.boolean_mask(X_left, sub_sample_mask)
        X_right = tf.slice(X, [0, window_size + 1], [size - span + 1, window_size])
        X_right = tf.boolean_mask(X_right, sub_sample_mask)
        labels = tf.reshape(tf.stack([X_left, X_right], axis=1), [-1])

        return examples, labels

    def gen_batch(self):
        opts = self._options
        min_after_dequeue = MIN_AFTER_DEQUEUE
        capacity = min_after_dequeue + (opts.io_threads + 1) * opts.batch_size
        corpus_slice = self.load_tfrecord()
        examples, labels = self.skip_gram(corpus_slice)
        x_batch, y_batch = tf.train.shuffle_batch([examples, labels],
                                                  batch_size=opts.batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue,
                                                  num_threads=opts.io_threads,
                                                  enqueue_many=True)
        return x_batch, y_batch

    def forward(self, examples, labels):
        opts = self._options
        vocab_size = self._vocab_size

        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        # Softmax weight and bias
        sm_w_t = tf.Variable(tf.zeros([vocab_size, opts.emb_dim]), name="sm_w_t")
        sm_b = tf.Variable(tf.zeros([vocab_size]), name="sm_b")

        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels, dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.neg_sample_num,
            unique=True,
            range_max=vocab_size,
            distortion=0.75,
            unigrams=self._word_freq[:]))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise lables for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.neg_sample_num])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def optimize(self, loss):
        opts = self._options
        self._lr = tf.train.exponential_decay(opts.learning_rate,
                                              self.global_step,
                                              opts.decay_steps,
                                              opts.decay_rate,
                                              staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train_op = optimizer.minimize(loss,
                                      global_step=self.global_step,
                                      gate_gradients=optimizer.GATE_NONE)


        nemb = tf.nn.l2_normalize(self._emb, 1)
        self._nemb = nemb

        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, self._vocab_size))
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        return train_op

    def build_graph(self):
        examples, labels = self.gen_batch()
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        self._loss = loss
        train_op = self.optimize(loss)
        self._train = train_op

    def _train_thread(self, coord):
        try:
            while not coord.should_stop():
                self._session.run([self._train])
        except tf.errors.OutOfRangeError:
            coord.request_stop()

    def train(self):
        opts = self._options
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._session.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._session, coord=coord)

        train_workers = []
        for _ in range(opts.parallel):
            worker = threading.Thread(target=self._train_thread, args=(coord,))
            worker.start()
            train_workers.append(worker)

        last_time, last_step = time.time(), 0
        last_eval_time = last_save_time = last_time

        try:
            while not coord.should_stop():
                time.sleep(opts.stat_time_interval)
                (step, loss, lr) = self._session.run(
                    [self.global_step, self._loss, self._lr])
                t_now = time.time()
                rate = (step - last_step) * opts.batch_size / (t_now - last_time)
                last_step = step
                last_time = t_now
                stamp = datetime.now().strftime('%F %T')
                print("%s Step %8d: lr = %5.4f loss = %6.3f samples/sec = %8.0f\r" %
                      (stamp, step, lr, loss, rate))
                sys.stdout.flush()

                if t_now - last_eval_time > opts.eval_time_interval:
                    self.nearby(self._valid_words, num=8)
                    last_eval_time = t_now
                if t_now - last_save_time > opts.save_time_interval:
                    self.save_model(stamp)
                    last_save_time = t_now

        except tf.errors.OutOfRangeError:
            print(last_step)
            print('Training Done')
        finally:
            coord.request_stop()
        coord.join(threads)

        for worker in train_workers:
            worker.join()

    def pick_valid_words(self, valid_words_count=8):
        opts = self._options
        if not opts.valid_words:
            valid_words = set()
            while len(valid_words) < valid_words_count:
                w_id, = np.random.choice(range(100, 200), 1)
                word = self._id2word[w_id]
                if word in valid_words or len(word) < 2:
                    continue
                valid_words.add(word)
            valid_words = list(valid_words)
        else:
            valid_words = opts.valid_words.strip().split(',')

        return list(valid_words)

    def nearby(self, words, num=20):
        ids = np.array([self._word2id.get(x, UNK_ID) for x in words])
        vals, idx = self._session.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in range(len(words)):
            line = '%s: ' % words[i]
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                line += '  (%s, %.3f)' % (self._id2word[neighbor], distance)
            print(line)
        print('----------')

    def save_model(self, index):
        opts = self._options
        nemb, = self._session.run([self._nemb])
        with open(opts.save_path + '_' + index, 'w') as fout:
            for i, vec in enumerate(nemb):
                word = self._id2word[i]
                fout.write(word + " ")
                fout.write(','.join(str(value) for value in vec))
                fout.write("\n")
        print('model saved in %s' % opts.save_path)


def main(args):
    with tf.Graph().as_default(), tf.Session() as session:
        model = Word2Vec(options=args, session=session)
        model.init()
        model.train()
        model.save_model("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str)
    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--valid_words', type=str)

    parser.add_argument('--epochs_to_train', type=int, default=30)
    parser.add_argument('--io_threads', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.98)
    parser.add_argument('--decay_steps', type=int, default=10000)

    parser.add_argument('--stat_time_interval', type=int, default=5)
    parser.add_argument('--eval_time_interval', type=int, default=20)
    parser.add_argument('--save_time_interval', type=int, default=150)

    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--neg_sample_num', type=int, default=32)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=1e-3)
    args = parser.parse_args()

    main(args)