# -*- coding:utf-8 -*-

import itertools
import pickle
from collections import OrderedDict

from nlp.ner.idcnn.model import Model
from nlp.ner.idcnn.loader import *
from nlp.ner.idcnn.dataset.batch_manager import BatchManager
from nlp.ner.idcnn.dataset.data_utils import *
from nlp.ner.idcnn.utils import *


def evaluate(sess, model, name, data, id_to_tag, logger, result_path):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1

# config for the model
def config_model(args, char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = args.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = args.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = args.seg_dim
    config["lstm_dim"] = args.lstm_dim
    config["batch_size"] = args.batch_size

    config["emb_file"] = args.emb_file
    config["clip"] = args.clip
    config["dropout_keep"] = 1.0 - args.dropout
    config["optimizer"] = args.optimizer
    config["lr"] = args.lr
    config["tag_schema"] = args.tag_schema
    config["pre_emb"] = args.pre_emb
    config["zeros"] = args.zeros
    config["lower"] = args.lower
    return config

def train(args):
    if args.clean:
        clean_and_make_path(args)

    # load data sets
    train_sentences = load_sentences(args.train_file, args.zeros)
    dev_sentences = load_sentences(args.dev_file, args.zeros)
    test_sentences = load_sentences(args.test_file, args.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)

    # create maps if not exist
    map_file = os.path.join(args.vocab_path, "map.pkl")
    if not os.path.isfile(map_file):
        # create dictionary for word
        if args.pre_emb:
            dico_chars_train = char_mapping(train_sentences, args.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                args.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, args.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, args.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, args.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, args.lower)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, args.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    config = config_model(args, char_to_id, tag_to_id)
    save_config(config, os.path.join(args.config_path, "config.json"))

    log_path = os.path.join(args.log_path, "train.log")
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(args.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % args.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger, args.result_path)
            if best:
                save_model(sess, model, args.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger, args.result_path)