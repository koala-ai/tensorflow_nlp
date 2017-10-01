# -*- coding:utf8 -*-

import os
import json
import tensorflow as tf
from scipy import spatial
from nlp.text_representation.doc2vec.model import Doc2Vec
from nlp.text_representation.doc2vec.dataset.data_utils import *


def train(args):
    docs = read_doc(os.path.join(args.data_file, "train.txt"))
    doc_ids, word_ids, count, dictionary, reverse_dictionary = \
        build_doc_dataset(docs, args.vocabulary_size)

    args.document_size = len(docs)
    model = Doc2Vec(args)

    configProto = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=configProto, graph=model.graph).as_default() as sess:
        sess.run(model.init_op)
        average_loss = 0
        print("Initialized")
        for step in range(args.n_steps):
            batch_data, batch_labels = generate_batch_pvdm(doc_ids, word_ids,
                                                               args.batch_size, args.window_size)
            feed_dict = {model.train_dataset: batch_data, model.train_labels: batch_labels}
            op, l = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

        # bind embedding matrices to self
        word_embeddings = sess.run(model.normalized_word_embeddings)
        doc_embeddings = sess.run(model.normalized_doc_embeddings)

        save(sess, args.train_dir, model.saver,
             docs, dictionary, reverse_dictionary,
             args.vocabulary_size,
             word_embeddings, doc_embeddings)


def save(sess, path, saver, docs, dictionary,
         reverse_dictionary, vocabulary_size,
         word_embeddings, doc_embeddings):

    with open(os.path.join(path, "wemb.txt"), 'w') as fout:
        for i, word in enumerate(dictionary):
            if i < vocabulary_size:
                fout.write(word + ' ')
                fout.write(','.join(str(value) for value in word_embeddings[i]))
                fout.write("\n")
    print('embedding saved in %s' % os.path.join(path, "wemb.txt"))

    with open(os.path.join(path, "demb.txt"), 'w') as fout:
        doclen = len(docs)
        for i in range(doclen):
            fout.write(','.join(docs[i]) + '\n')
            fout.write(','.join(str(value) for value in doc_embeddings[i]))
            fout.write("\n")
    distance = spatial.distance.cosine(doc_embeddings[doclen-1], doc_embeddings[doclen-2])
    print(distance)
    print('embedding saved in %s' % os.path.join(path, "demb.txt"))

    save_path = saver.save(sess, os.path.join(path, 'model.ckpt'))
    # save dictionary, reverse_dictionary
    json.dump(dictionary,
                  open(os.path.join(path, 'model_dict.json'), 'w'))
    json.dump(reverse_dictionary,
                  open(os.path.join(path, 'model_rdict.json'), 'w'))

    print("Model saved in file: %s" % save_path)