# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from nlp.poems.model import Model
from nlp.poems.dataset.poems import *

start_token = 'G'
end_token = 'E'


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(args):
    batch_size = 1
    print('[INFO] loading corpus from %s' % os.path.join(args.data_dir, "poems.txt"))
    poems_vector, word_int_map, vocabularies = process_poems(os.path.join(args.data_dir, "poems.txt"))

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    model = Model(args.method, input_data, None, len(
        vocabularies), args.rnn_size, args.num_layers, args.batch_size, args.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(args.train_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([model.end_points['prediction'], model.end_points['last_state']],
                                         feed_dict={input_data: x})
        if args.begin_word:
            word = args.begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([model.end_points['prediction'], model.end_points['last_state']],
                                             feed_dict={input_data: x, model.end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        pretty_print_poem(poem)


def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')