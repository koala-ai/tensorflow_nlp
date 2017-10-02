# -*- coding:utf-8 -*-


import tensorflow as tf
import argparse
from nlp.textsum import train
from nlp.textsum import predict


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        train.train(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        predict.generate_summary(args)
    else:
        raise Exception("cannot support this process:" + args.process)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/textsum2/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/textsum2/data')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/textsum2/utils')

    parser.add_argument('--num_samples', type=int, default=2)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_steps', type=int, default=100)

    parser.add_argument('--max_gradient_norm', type=float, default=5.0)

    parser.add_argument('--init_scale', type=float, default=0.04)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--buckets', type=list, default=[(120, 30)])

    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='train')

    args = parser.parse_args()

    main(args)