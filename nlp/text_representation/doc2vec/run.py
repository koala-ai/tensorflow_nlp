# -*- coding:utf8 -*-

import argparse
from nlp.text_representation.doc2vec.train import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/doc2vec/ckpt')

    parser.add_argument('--data_file', type=str, default='data/doc2vec/train.txt')
    parser.add_argument('--loss_type', type=str, default='sampled_softmax_loss')
    parser.add_argument('--optimize', type=str, default='Adagrad')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=8)

    parser.add_argument('--concat', type=bool, default=True)
    parser.add_argument('--embedding_size_w', type=int, default=128)
    parser.add_argument('--embedding_size_d', type=int, default=128)
    parser.add_argument('--vocabulary_size', type=int, default=5000)
    parser.add_argument('--n_neg_samples', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--n_steps', type=int, default=10000)

    args = parser.parse_args()

    train(args)