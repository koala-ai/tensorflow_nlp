# -*- coding:utf-8 -*-

import argparse
from nlp.poems.train import train
from nlp.poems.generate import gen_poem


def main(args):
    if args.process == 'train':
        train(args)
    elif args.process == 'generate':
        # test_seg_tagger.joint_predict(args)
        gen_poem(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Intelligence Poem Writer.')

    parser.add_argument('--train_dir', type=str, default="train_dir", help="model dir")
    parser.add_argument('--data_dir', type=str, default="data_dir", help="data dir")

    parser.add_argument('--method', type=str, default="lstm", help="rnn type, [rnn,gru,lstm]")
    parser.add_argument('--epochs', type=int, default=10, help="epochs")
    parser.add_argument('--rnn_size', type=int, default=128, help="rnn size")
    parser.add_argument('--num_layers', type=int, default=2, help="num layers")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="learning_rate")

    parser.add_argument('--process', type=str, default="generate", help="Whether train model")
    parser.add_argument('--begin_word', type=str, default="慧", help="begin word to generate")

    args = parser.parse_args()

    main(args)