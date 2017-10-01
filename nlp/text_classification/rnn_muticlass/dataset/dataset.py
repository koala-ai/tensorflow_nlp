# -*- coding: utf-8 -*-

import os
import collections
import _pickle as pickle

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, is_training, utils_dir, data_path, batch_size,
                 seq_length, vocab, labels):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.utils_dir = utils_dir
        label_file = os.path.join(utils_dir, 'labels.pkl')
        vocab_file = os.path.join(utils_dir, 'vocab.pkl')

        if is_training:
            corpus_file = os.path.join(data_path, 'data.csv')
            train_file = os.path.join(data_path, 'train.csv')

            if not os.path.exists(vocab_file):
                print('reading corpus and processing data')
                self.preprocess(vocab_file, corpus_file, train_file, label_file)
            else:
                print('loading vocab and processing data')
                self.load_preprocessed(vocab_file, train_file, label_file)

        elif vocab is not None and labels is not None:
            self.vocab = vocab
            self.vocab_size = len(vocab) + 1
            self.labels = labels
            self.label_size = len(self.labels)

            train_file = os.path.join(data_path, 'train.csv')
            self.load_preprocessed(None, train_file, label_file)

        self.reset_batch_pointer()

    def transform(self, d):
        new_d = list(map(self.vocab.get, d[:self.seq_length]))
        new_d = list(map(lambda i: i if i else 0, new_d))

        if len(list(new_d)) >= self.seq_length:
            new_d = new_d[:self.seq_length]
        else:
            new_d = new_d + [0] * (self.seq_length - len(list(new_d)))
        return new_d

    def preprocess(self, vocab_file, corpus_file, data_path, label_file):

        corpus = pd.read_csv(corpus_file, encoding='utf8')
        labels = corpus['label'].drop_duplicates().values
        corpus = corpus['text']
        corpus = ''.join(map(lambda i: i.strip(), corpus))

        self.labels = dict(zip(labels, range(len(labels))))
        self.label_size = len(labels)

        with open(label_file, 'wb') as f:
            pickle.dump(self.labels, f)

        counter = collections.Counter(corpus)
        count_pairs = sorted(counter.items(), key=lambda i: -i[1])
        self.chars, _ = zip(*count_pairs)
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        self.vocab_size = len(self.chars) + 1
        self.vocab = dict(zip(self.chars, range(1, len(self.chars) + 1)))

        data = pd.read_csv(data_path, encoding='utf8')
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)

    def load_preprocessed(self, vocab_file, data_path, label_file):
        if vocab_file is not None:
            with open(vocab_file, 'rb') as f:
                self.chars = pickle.load(f)
            self.vocab_size = len(self.chars) + 1
            self.vocab = dict(zip(self.chars, range(1, len(self.chars) + 1)))

        if label_file is not None:
            with open(label_file, 'rb') as f:
                self.labels = pickle.load(f)
            self.label_size = len(self.labels)

        data = pd.read_csv(data_path, encoding='utf8')
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'

        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)

    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0