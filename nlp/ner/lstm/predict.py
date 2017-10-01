# -*- coding:utf-8 -*-

import codecs
import os
import _pickle as pickle
import tensorflow as tf
import glob
import numpy as np
from nlp.ner.lstm import model as lstm_model , bilstm_model
from nlp.ner.lstm.dataset import rawdata


class ModelLoader(object):
    def __init__(self, args, ckpt_path):
        self.config = args
        self.method = args.model
        self.data_dir = args.data_dir
        self.utils_dir = args.utils_dir
        self.ckpt_path = ckpt_path
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing ner_tagger model...")
        self.model = self.init_ner_model(self.session, self.ckpt_path)

    def init_ner_model(self, session, ckpt_path):

        with tf.variable_scope("ner_var_scope"):
            if self.method == "lstm":
                model = lstm_model.NERTagger(is_training=False, config=self.config)
            else:
                model = bilstm_model.NERTagger(is_training=False, config=self.config)

        if len(glob.glob(ckpt_path + '.data*')) > 0:
            print("Loading model parameters from %s" % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("ner_var_scope")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())

        return model

    def predict(self, words):
        word_data = rawdata.sentence_to_word_ids(self.utils_dir, words)
        word_arr = np.zeros((1, self.model.seq_length), np.int32)

        for i, word in enumerate(word_data):
            word_arr[0][i] = word

        fetches = [self.model.logits]
        feed_dict = {self.model.input_data: word_arr}

        logits = self.session.run(fetches, feed_dict)
        predict_id = np.argmax(logits[0], 1)[0:len(words)]

        # print(predict_id)

        predict_tag = rawdata.tag_ids_to_tags(self.utils_dir, predict_id)
        return zip(words, predict_tag)


def load_model(args):

    with open(os.path.join(args.utils_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'tags.pkl'), 'rb') as f:
        tags = pickle.load(f)

    args.vocab_size = len(vocab)
    args.tag_size = len(tags)
    args.batch_size = 1

    if args.model == "lstm":
        ckpt_path = os.path.join(args.train_dir, "lstm/model.ckpt")  # POS model checkpoint path
    else:
        ckpt_path = os.path.join(args.train_dir, "bilstm/model.ckpt")  # POS model checkpoint path
    return ModelLoader(args, ckpt_path)


def predict(args):
    tagger = load_model(args)
    with codecs.open(args.predict_file, 'r', 'utf-8') as predict,\
            codecs.open(args.result_file, 'w', 'utf-8') as output:
        lines = predict.readlines()
        for line in lines:
            words = line.split()
            tagging = tagger.predict(words)
            for (w, t) in tagging:
                str = w + "/" + t
                output.write(str + ' ')
            output.write('\n')

