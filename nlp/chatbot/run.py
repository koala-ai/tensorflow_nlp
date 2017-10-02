# -*- coding: utf-8 -*-

import argparse

from nlp.chatbot import train
from nlp.chatbot import test


def main(args):
    if args.bleu > -1:
        test.test_bleu(args.bleu, args)
    elif args.test:
        test.test(args)
    else:
        train.train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str, default='/ckpt')
    parser.add_argument('--buckets_dir', required=True, type=str, default='./bucket_dbs')
    parser.add_argument('--use_fp16', type=bool, default=False)

    parser.add_argument('--model_name', type=str, default='model')

    parser.add_argument('--dropout', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--max_gradient_norm', type=float, default=5.0)

    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_per_epoch', type=int, default=100000)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--bleu', type=int, default=-1)

    parser.add_argument('--test', type=bool, default=False)

    args = parser.parse_args()