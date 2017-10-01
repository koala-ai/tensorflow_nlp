# -*- coding:utf-8 -*-

import os
import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
from nlp.relation_extract.model import GRU


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def test(args):
    model_path = os.path.join(args.train_dir, "ATT_GRU_model-100")

    wordembedding = np.load(os.path.join(args.data_dir, 'vec.npy'))

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            print("read model")
            with tf.variable_scope("model"):
                mtest = GRU(is_training=False, word_embeddings=wordembedding, args=args)

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)

            #for model_iter in testlist:
            # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
            print("restore %s" % model_path)
            saver.restore(sess, model_path)

            time_str = datetime.datetime.now().isoformat()
            print(time_str)
            print('Evaluating all test data and save data for PR curve')

            test_y = np.load(os.path.join(args.data_dir, 'testall_y.npy'))
            test_word = np.load(os.path.join(args.data_dir, 'testall_word.npy'))
            test_pos1 = np.load(os.path.join(args.data_dir, 'testall_pos1.npy'))
            test_pos2 = np.load(os.path.join(args.data_dir, 'testall_pos2.npy'))
            allprob = []
            acc = []
            for i in range(int(len(test_word) / float(args.big_num))):
                prob, accuracy = test_step(test_word[i * args.big_num:(i + 1) * args.big_num],
                                               test_pos1[i * args.big_num:(i + 1) * args.big_num],
                                               test_pos2[i * args.big_num:(i + 1) * args.big_num],
                                               test_y[i * args.big_num:(i + 1) * args.big_num])
                acc.append(np.mean(np.reshape(np.array(accuracy), (args.big_num))))
                prob = np.reshape(np.array(prob), (args.big_num, args.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])
            allprob = np.reshape(np.array(allprob), (-1))

            print('saving all test result...')

            np.save(os.path.join(args.data_dir, 'allprob_iter.npy'), allprob)
            allans = np.load(os.path.join(args.data_dir, 'allans.npy'))

            # caculate the pr curve area
            average_precision = average_precision_score(allans, allprob)
            print('PR curve area:' + str(average_precision))
