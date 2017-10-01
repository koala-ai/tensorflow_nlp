# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from nlp.poems.model import Model
from nlp.poems.dataset.poems import *


def train(args):

    poems_vector, word_to_int, vocabularies = process_poems(os.path.join(args.data_dir, "poems.txt"))
    batches_inputs, batches_outputs = generate_batch(args.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [args.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [args.batch_size, None])

    model = Model(args.method,
                  input_data,
                  output_targets,
                  len(vocabularies),
                  args.rnn_size,
                  args.num_layers,
                  args.batch_size,
                  learning_rate=args.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(args.train_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, args.epochs):
                n = 0
                n_chunk = len(poems_vector) // args.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        model.end_points['total_loss'],
                        model.end_points['last_state'],
                        model.end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                if epoch % 2 == 0:
                    saver.save(sess, os.path.join(args.train_dir, "poem"), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(args.train_dir, "poem"), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))
