# -*- coding:utf-8 -*-

import pprint
import tensorflow as tf
import numpy as np
import os
import argparse

from nlp.sentiment.reader import read_data
from nlp.sentiment.model import MemN2N


pp = pprint.PrettyPrinter()


def init_word_embeddings(args, word2idx, data_path):
  wt = np.random.normal(0, args.init_std, [len(word2idx), args.edim])

  pretrain_file = os.path.join(data_path, "glove.6B.300d.txt")
  with open(pretrain_file, 'r') as f:
    for line in f:
      content = line.strip().split()
      if content[0] in word2idx:
        wt[word2idx[content[0]]] = np.array(map(float, content[1:]))
  return wt


def train_sentiment(args):
    source_count, target_count = [], []
    source_word2idx, target_word2idx = {}, {}

    train_file = os.path.join(args.data_path, "Laptop_Train_v2.xml")
    test_file = os.path.join(args.data_path, "Laptops_Test_Gold.xml")

    train_data = read_data(train_file, source_count, source_word2idx, target_count, target_word2idx)
    test_data = read_data(test_file, source_count, source_word2idx, target_count, target_word2idx)

    args.pad_idx = source_word2idx['<pad>']
    args.nwords = len(source_word2idx)
    args.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]

    print('loading pre-trained word vectors...')
    args.pre_trained_context_wt = init_word_embeddings(args, source_word2idx, args.data_path)
    args.pre_trained_target_wt = init_word_embeddings(args, target_word2idx, args.data_path)

    with tf.Session() as sess:
        model = MemN2N(args, sess)
        model.build_model()
        model.run(train_data, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, default='data/textsum2/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/textsum2/data')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/textsum2/utils')

    parser.add_argument('--max_grad_norm', type=int, default=50)

    parser.add_argument('--edim', type=int, default=300)
    parser.add_argument('--lindim', type=int, default=75)
    parser.add_argument('--nhop', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--nepoch', type=int, default=2)

    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--init_hid', type=float, default=0.1)
    parser.add_argument('--init_std', type=float, default=0.05)
    parser.add_argument('--buckets', type=list, default=[(120, 30)])

    args = parser.parse_args()
    train_sentiment(args)