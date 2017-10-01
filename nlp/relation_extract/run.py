# -*- coding:utf-8 -*-

import argparse
import tensorflow as tf
from nlp.relation_extract.train import train
from nlp.relation_extract.test import test

def main(args):

    if args.process == tf.estimator.ModeKeys.TRAIN:
        train(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        test(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/relation-extract/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/relation-extract/data')
    parser.add_argument('--summary_dir', required=True, type=str, default='data/relation-extract/summary')

    parser.add_argument('--vocab_size', type=int, default=16691)
    parser.add_argument('--num_steps', type=int, default=70)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--gru_size', type=int, default=230)

    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--pos_size', type=int, default=5)
    parser.add_argument('--pos_num', type=int, default=123)
    parser.add_argument('--big_num', type=int, default=50)

    parser.add_argument('--process', type=str, default='infer')

    args = parser.parse_args()

    main(args)
