# -*- coding:utf-8 -*-

import tensorflow as tf
import argparse
from nlp.pos.lstm.train import train_pos
from nlp.pos.lstm.predict import predict_pos


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        if args.model == "lstm":
            train_pos.train_lstm(args)
        else:
            train_pos.train_bilstm(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        if args.model == "lstm":
            predict_pos.predict(args)
        else:
            predict_pos.predict(args)
    else:
        raise Exception("cannot support this process:" + args.process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/lstm_pos/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/lstm_pos/data')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/lstm_pos/utils')

    parser.add_argument('--model', required=True, type=str, default='lstm')

    parser.add_argument('--max_grad_norm', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--init_scale', type=float, default=0.04)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.9)

    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='train')

    args = parser.parse_args()

    main(args)
