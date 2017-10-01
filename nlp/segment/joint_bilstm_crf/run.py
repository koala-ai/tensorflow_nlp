# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from nlp.segment.joint_bilstm_crf import train
from nlp.segment.joint_bilstm_crf import test


def main(args):

    if args.process == tf.estimator.ModeKeys.TRAIN:
        train.train(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        test.joint_predict(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/bi_lstm_crf_segment/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/bi_lstm_crf_segment/data')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/bi_lstm_crf_segment/utils')
    parser.add_argument('--vocab_dir', required=True, type=str, default='data/bi_lstm_crf_segment/vocab')

    parser.add_argument('--optimizer', type=str, default='adagrad')
    parser.add_argument('--tag_scheme', type=str, default='BIES')

    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--word_vector', type=bool, default=True)
    parser.add_argument('--pre_embeddings', type=bool, default=True)
    parser.add_argument('--gru', type=bool, default=True)
    parser.add_argument('--clipping', type=bool, default=True)
    parser.add_argument('--ensemble', type=bool, default=False)
    parser.add_argument('--tag_large', type=bool, default=False)

    parser.add_argument('--crf', type=int, default=1)
    parser.add_argument('--cell_dimension', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--embeddings_dimension', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--large_size', type=int, default=200000)
    parser.add_argument('--bucket_size', type=int, default=10)

    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.05)


    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='train')

    args = parser.parse_args()

    main(args)
