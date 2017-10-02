# -*- coding: utf-8 -*-

import tensorflow as tf
from nlp.chatbot.dataset import data_utils
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import os,sys
import numpy as np
from nlp.chatbot import model as s2s_model


def test_bleu(count, args):
    print('准备数据')
    bucket_dbs = data_utils.read_bucket_dbs(args.buckets_dir)
    buckets = data_utils.buckets

    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    if count <= 0:
        count = total_size
    buckets_scale = [
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    with tf.Session() as sess:
        model = s2s_model.create_model(sess, True)
        model.batch_size = 1
        sess.run(tf.initialize_all_variables())
        model.saver.restore(sess, os.path.join(args.model_dir, args.model_name))

        total_score = 0.0
        for i in tqdm(range(count)):
            random_number = np.random.random_sample()
            bucket_id = min([
                i for i in range(len(buckets_scale))
                if buckets_scale[i] > random_number
            ])
            data, _ = model.get_batch_data(
                bucket_dbs,
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                bucket_dbs,
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ask, _ = data[0]
            all_answers = bucket_dbs[bucket_id].all_answers(ask)
            ret = data_utils.indice_sentence(outputs)
            if not ret:
                continue
            references = [list(x) for x in all_answers]
            score = sentence_bleu(
                references,
                list(ret),
                weights=(1.0,)
            )
            total_score += score
        print('BLUE: {:.2f} in {} samples'.format(total_score / count * 10, count))


def test(args):
    class TestBucket(object):
        def __init__(self, sentence):
            self.sentence = sentence
        def random(self):
            return sentence, ''
    buckets = data_utils.buckets

    with tf.Session() as sess:
        model = s2s_model.create_model(sess, True)
        model.batch_size = 1
        sess.run(tf.initialize_all_variables())
        model.saver.restore(sess, os.path.join(args.model_dir, args.model_name))
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            bucket_id = min([
                b for b in range(len(buckets))
                if buckets[b][0] > len(sentence)
            ])
            data, _ = model.get_batch_data(
                {bucket_id: TestBucket(sentence)},
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                {bucket_id: TestBucket(sentence)},
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ret = data_utils.indice_sentence(outputs)
            print(ret)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()