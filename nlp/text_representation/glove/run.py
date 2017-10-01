# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from nlp.text_representation.glove.train import train


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        train(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str, default='data/glove')
    parser.add_argument('--log_dir', required=True, type=str, default='data/glove/log')

    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--max_vocab_size', type=int, default=100000)
    parser.add_argument('--min_occurrences', type=int, default=1)

    parser.add_argument('--scaling_factor', type=float, default=0.75)
    parser.add_argument('--cooccurrence_cap', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.05)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--tsne_epoch_interval', type=int, default=50)

    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='train')

    args = parser.parse_args()

    main(args)
