# -*- coding: utf-8 -*-

import os, time
from nlp.segment.joint_bilstm_crf.dataset import data_utils
from nlp.segment.joint_bilstm_crf.model import Model
import tensorflow as tf


def train(args):
    if not os.path.isdir(args.vocab_dir):
        os.makedirs(args.vocab_dir)

    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    train_file = os.path.join(args.data_dir, 'train1.txt')
    dev_file = os.path.join(args.data_dir, 'dev1.txt')
    glove_file = os.path.join(args.utils_dir, 'glove.txt')

    if args.ngram > 1 and not os.path.isfile(args.vocab_dir + '/' + str(args.ngram) + 'gram.txt') \
        or (not os.path.isfile(args.vocab_dir + '/chars.txt')):
        data_utils.get_vocab_tag(args.vocab_dir, [train_file, dev_file], ngram=args.ngram)

    chars, tags, ngram = data_utils.read_vocab_tag(args.vocab_dir, args.ngram)

    emb = None
    emb_dim = args.embeddings_dimension
    if args.pre_embeddings:
        short_emb = "glove.txt"
        if not os.path.isfile(args.vocab_dir + '/' + short_emb + '_sub.txt'):
            data_utils.get_sample_embedding(glove_file, args.vocab_dir, short_emb, chars)
            emb_dim, emb = data_utils.read_sample_embedding(args.vocab_dir, short_emb)
            assert args.embeddings_dimension == emb_dim
    else:
        print('Using random embeddings...')

    char2idx, idx2char, tag2idx, idx2tag = data_utils.get_dic(chars, tags)

    train_x, train_y, train_max_slen_c, train_max_slen_w, train_max_wlen = \
        data_utils.get_input_vec(train_file, char2idx, tag2idx, args.tag_scheme)

    dev_x, dev_y, dev_max_slen_c, dev_max_slen_w, dev_max_wlen = \
        data_utils.get_input_vec(dev_file, char2idx, tag2idx, args.tag_scheme)

    nums_grams = []
    ng_embs = None
    if args.ngram > 1:
        gram2idx = data_utils.get_ngram_dic(ngram)
        train_gram = data_utils.get_gram_vec(train_file, gram2idx)
        dev_gram = data_utils.get_gram_vec(dev_file, gram2idx)
        train_x += train_gram
        dev_x += dev_gram

        for dic in gram2idx:
            nums_grams.append(len(dic.keys()))

    max_step_c = max(train_max_slen_c, dev_max_slen_c)
    max_step_w = max(train_max_slen_w, dev_max_slen_w)
    max_w_len = max(train_max_wlen, dev_max_wlen)
    print('Longest sentence by character is %d. ' % max_step_c)
    print('Longest sentence by word is %d. ' % max_step_w)
    print('Longest word is %d. ' % max_w_len)

    b_train_x, b_train_y = data_utils.buckets(train_x, train_y, size=args.bucket_size)
    b_dev_x, b_dev_y = data_utils.buckets(dev_x, dev_y, size=args.bucket_size)

    b_train_x, b_train_y, b_buckets, b_counts = data_utils.pad_bucket(b_train_x, b_train_y)
    b_dev_x, b_dev_y, b_buckets, _ = data_utils.pad_bucket(b_dev_x, b_dev_y, bucket_len_c=b_buckets)

    print('Training set: %d instances; Dev set: %d instances.' % (len(train_x[0]), len(dev_x[0])))

    nums_tags = data_utils.get_nums_tags(tag2idx, args.tag_scheme)

    configProto = tf.ConfigProto(allow_soft_placement=True)
    print('Initialization....')

    initializer = tf.contrib.layers.xavier_initializer()
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger", reuse=None, initializer=initializer) as scope:
            model = Model(nums_chars=len(chars) + 2, nums_tags=nums_tags, buckets_char=b_buckets,
                          counts=b_counts, tag_scheme=args.tag_scheme, word_vec=args.word_vector,
                          crf=args.crf, ngram=nums_grams, batch_size=args.batch_size)

            model.model_graph(trained_model=args.train_dir + '/trained_model', scope=scope, emb_dim=emb_dim,
                              gru=args.gru, rnn_dim=args.cell_dimension, rnn_num=args.num_layers,
                              emb=emb, ng_embs=ng_embs, drop_out=args.dropout_rate)
            t = time.time()
        model.config(optimizer=args.optimizer, decay=args.decay_rate, lr_v=args.learning_rate,
                         momentum=None, clipping=args.clipping)
        init = tf.global_variables_initializer()

    main_graph.finalize()

    main_sess = tf.Session(config=configProto, graph=main_graph)

    if args.crf:
        decode_graph = tf.Graph()
        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()
        decode_sess = tf.Session(config=configProto, graph=decode_graph)
        sess = [main_sess, decode_sess]
    else:
        sess = [main_sess]

    with tf.device("/cpu:0"):
        main_sess.run(init)
        print('Done. Time consumed: %d seconds' % int(time.time() - t))
        t = time.time()
        model.train(t_x=b_train_x, t_y=b_train_y, v_x=b_dev_x, v_y=b_dev_y, idx2tag=idx2tag, idx2char=idx2char,
                    sess=sess, epochs=args.num_epochs, trained_model=args.train_dir + '/trained_model_weights',
                    lr=args.learning_rate, decay=args.decay_rate)
        print('Done. Time consumed: %d seconds' % int(time.time() - t))
