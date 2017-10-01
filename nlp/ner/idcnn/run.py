# -*- coding:utf-8 -*-

import argparse

from nlp.ner.idcnn.train import train
from nlp.ner.idcnn.predict import predict


def main(args):
    if args.train:
        train(args)
    else:
        predict(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True, help="Whether train the model")
    parser.add_argument('--clean', type=bool, default=True, help="Whether clean the model")

    parser.add_argument('--ckpt_path', type=str, default="ckpt", help="Path to save model")
    parser.add_argument('--log_path', type=str, default="train.log", help="File for log")
    parser.add_argument('--vocab_path', type=str, default="vocab.json", help="Path to vocab file")
    parser.add_argument('--config_path', type=str, default="config_file", help="File for config")
    parser.add_argument('--script', type=str, default="conlleval", help="evaluation script")
    parser.add_argument('--result_path', type=str, default="result", help="Path to result")
    parser.add_argument('--emb_file', type=str, default="vec.txt", help="Path for pre_trained embedding")
    parser.add_argument('--train_file', type=str, default="train.txt", help="Path for train data")
    parser.add_argument('--dev_file', type=str, default="dev.txt", help="Path for dev data")
    parser.add_argument('--test_file', type=str, default="test.txt", help="Path for test data")
    parser.add_argument('--raw_file', type=str, default="example.raw", help="Path for predict data")

    parser.add_argument('--model_type', type=str, default="bilstm", help="Model type, can be idcnn or bilstm")

    parser.add_argument('--seg_dim', type=int, default=50, help="Embedding size for segmentation, 0 if not used")
    parser.add_argument('--char_dim', type=int, default=100, help="Embedding size for characters")
    parser.add_argument('--lstm_dim', type=int, default=100, help="Num of hidden units in LSTM, or num of filters in IDCNN")
    parser.add_argument('--tag_schema', type=str, default="iobes", help="tagging schema iobes or iob")

    parser.add_argument('--clip', type=int, default=5, help="Gradient clip")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")

    parser.add_argument('--optimizer',  type=str, default='adam')
    parser.add_argument('--pre_emb', type=bool, default=True, help="Whether use pre-trained embedding")
    parser.add_argument('--zeros', type=bool, default=False, help="Whether replace digits with zero")
    parser.add_argument('--lower', type=bool, default=True, help="Whether lower case")

    parser.add_argument('--max_epoch', type=int, default=4, help="maximum training epochs")
    parser.add_argument('--steps_check', type=int, default=100, help="steps per checkpoint")

    args = parser.parse_args()
    main(args)

