# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse

from nlp.text_classification.rnn_muticlass import train
from nlp.text_classification.rnn_muticlass import predict


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        train.train_rnn_classfier(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        predict.predict(args)
    else:
        raise Exception("cannot support this process:" + args.process)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/rnn-multiclass-classify/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/rnn-multiclass-classify/data/')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/rnn-multiclass-classify/utils')

    parser.add_argument('--model', type=str, default='lstm')

    parser.add_argument('--rnn_size', type=int, default=128)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay_rate', type=float, default=0.9)

    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='infer')

    args = parser.parse_args()

    main(args)

