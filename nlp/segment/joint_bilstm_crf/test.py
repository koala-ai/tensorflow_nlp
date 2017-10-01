# -*- coding: utf-8 -*-

import os, time
import codecs
import pickle
from nlp.segment.joint_bilstm_crf.dataset import data_utils
from nlp.segment.joint_bilstm_crf.model import Model
import tensorflow as tf


def joint_predict(args):
    emb_path = None
    ng_emb_path = None

    model_path = args.train_dir + '/trained_model'
    weight_path = args.train_dir + '/trained_model_weights'

    if not os.path.isfile(model_path):
        raise Exception('No model file %s or weights file %s' % (model_path, weight_path))

    fin = open(model_path, 'rb')
    param_dic = pickle.load(fin)
    fin.close()

    nums_chars = param_dic['nums_chars']
    nums_tags = param_dic['nums_tags']
    tag_scheme = param_dic['tag_scheme']
    word_vector = param_dic['word_vec']
    crf = param_dic['crf']
    emb_dim = param_dic['emb_dim']
    gru = param_dic['gru']
    rnn_dim = param_dic['rnn_dim']
    rnn_num = param_dic['rnn_num']
    drop_out = param_dic['drop_out']
    num_ngram = param_dic['ngram']

    ngram = 1

    if num_ngram is not None:
        ngram = len(num_ngram) + 1

    chars, tags, grams = data_utils.read_vocab_tag(args.train_dir, ngram)
    char2idx, idx2char, tag2idx, idx2tag = data_utils.get_dic(chars, tags)
    new_chars, new_grams, new_gram_emb, gram2idx = None, None, None, None
    raw_x = None

    new_chars = data_utils.get_new_chars(args.predict_file, char2idx, type='raw')
    char2idx, idx2char = data_utils.update_char_dict(char2idx, new_chars)

    if not args.tag_large:
        raw_x, raw_len = data_utils.get_input_vec_raw(args.predict_file, char2idx)
        print('Numbers of sentences: %d.' % len(raw_x[0]))
        max_step = raw_len
    else:
        max_step = data_utils.get_maxstep(args.predict_file)

    print('Longest sentence is %d. ' % max_step)
    if ngram > 1:
        gram2idx = data_utils.get_ngram_dic(grams)
        if not args.tag_large:
            raw_gram = data_utils.get_gram_vec(args.predict_file, gram2idx, is_raw=True)
            raw_x += raw_gram
    if not args.tag_large:
        for k in range(len(raw_x)):
            raw_x[k] = data_utils.pad_zeros(raw_x[k], max_step)

    configProto = tf.ConfigProto(allow_soft_placement=True)
    print('Initialization....')
    t = time.time()
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200],
                          tag_scheme=tag_scheme, word_vec=word_vector,
                          crf=crf, ngram=num_ngram, batch_size=args.batch_size)

            model.model_graph(trained_model=None, scope=scope, emb_dim=emb_dim, gru=gru, rnn_dim=rnn_dim,
                              rnn_num=rnn_num, drop_out=drop_out)

            model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx, new_grams=new_grams,
                                 ng_emb_path=ng_emb_path, gram2idx=gram2idx)

            init = tf.global_variables_initializer()

            print('Done. Time consumed: %d seconds' % int(time.time() - t))

        main_graph.finalize()
        main_sess = tf.Session(config=configProto, graph=main_graph)

        if crf:
            decode_graph = tf.Graph()
            with decode_graph.as_default():
                model.decode_graph()
            decode_graph.finalize()
            decode_sess = tf.Session(config=configProto, graph=decode_graph)
            sess = [main_sess, decode_sess]
        else:
            sess = [main_sess]

        with tf.device("/cpu:0"):
            print('Loading weights....')
            main_sess.run(init)
            model.run_updates(main_sess, weight_path)
            l_writer = codecs.open(args.result_file, 'w', encoding='utf-8')

            with codecs.open(args.predict_file, 'r', encoding='utf-8') as l_file:
                lines = []
                for line in l_file:
                    lines.append(line.strip())
                    if len(lines) >= args.large_size:
                        raw_x, _ = data_utils.get_input_vec_line(lines, char2idx)

                        if ngram > 1:
                            raw_gram = data_utils.get_gram_vec_raw(lines, gram2idx)
                            raw_x += raw_gram

                        for k in range(len(raw_x)):
                            raw_x[k] = data_utils.pad_zeros(raw_x[k], max_step)

                        out = model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                        outpath=args.output_file, ensemble=args.ensemble,
                                        batch_size=args.tag_batch, large_file=True)

                        for l_out in out:
                            l_writer.write(l_out + '\n')
                        lines = []

                if len(lines) > 0:
                    raw_x, _ = data_utils.get_input_vec_line(lines, char2idx)

                    if ngram > 1:
                        raw_gram = data_utils.get_gram_vec_raw(lines, gram2idx)
                        raw_x += raw_gram

                    for k in range(len(raw_x)):
                        raw_x[k] = data_utils.pad_zeros(raw_x[k], max_step)

                    model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                    outpath=args.result_file, ensemble=args.ensemble, batch_size=args.batch_size,
                                    large_file=False)
            l_writer.close()