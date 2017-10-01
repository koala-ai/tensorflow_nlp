# -*- coding:utf-8 -*-

import tensorflow as tf
from nlp.text_classification.cnn_multiclass import train
from nlp.text_classification.cnn_multiclass import predict
import argparse


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        train.train_cnn_classfier(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        predict.predict(args)
    else:
        raise Exception("cannot support this process:" + args.process)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/cnn_multiclass_classify/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/cnn_multiclass_classify/data/train.csv')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/cnn_multiclass_classify/utils')

    parser.add_argument('--vector_size', type=int, default=128)

    parser.add_argument('--hidden_layer_input_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sentence_length', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--filter_hs', type=list, default=[3, 4, 5])
    parser.add_argument('--num_filters', type=float, default=128)
    parser.add_argument('--img_h', type=int, default=20)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--filter_w', type=int, default=100)

    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='infer')

    args = parser.parse_args()

    main(args)

