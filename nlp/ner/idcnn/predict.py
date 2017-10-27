# -*- coding:utf-8 -*-

import os
import pickle
import tensorflow as tf
from nlp.ner.idcnn.model import Model
from nlp.ner.idcnn.dataset.data_utils import load_word2vec, input_from_line

from nlp.ner.idcnn.utils import get_logger, create_model, load_config


def predict(args):
    config = load_config(os.path.join(args.config_path, "config.json"))
    logger = get_logger(os.path.join(args.log_path, "predict.log"))
    map_file = os.path.join(args.vocab_path, "map.pkl")
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args.ckpt_path, load_word2vec, config, id_to_char, logger)
        with open(args.raw_file, "rb") as f:
            for line in f.readlines():
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)