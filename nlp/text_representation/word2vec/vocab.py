#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import os
import tensorflow as tf
import argparse
import sys

UNK_ID = 1


def _int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_read_data(filename):
    data = []
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    lines = tf.string_split([value], "\n")

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run([lines])
            lines_eval = lines.eval()
            for line in lines_eval.values:
                words = line.strip().split()
                result = []
                for word in words:
                    if word == "":
                        continue
                    else:
                        result.append(word.split('/')[0])
                data.append(result)
            coord.request_stop()
            coord.join(threads)
    return data


class Vocab(object):

    def __init__(self, train_dir, data_file, min_count):
        self._train_dir = train_dir
        self._data_file = data_file
        self._min_count = min_count
        self._word2id = None
        self._data = None
        self._word_freq = Counter()

    def build_vocab(self):
        self._data = tf_read_data(self._data_file)
        for line in self._data:
            self._word_freq.update(line)

        self._word_freq = self._word_freq.most_common()
        word_freq1 = [(w, c) for w, c in self._word_freq if c >= self._min_count]
        id2word = ['_PAD', '_UNK'] + [word for word, _ in word_freq1]
        self._word2id = {w: i for i, w in enumerate(id2word)}

    def dump_data(self):
        self.dump_tfrecord()
        self.dump_word_freq()

    def dump_tfrecord(self):
        out_file = os.path.join(self._train_dir, 'output.tfrecord')
        writer = tf.python_io.TFRecordWriter(out_file)
        for line in self._data:
            if len(line):
                sent = []
                for w in line:
                    id = self._word2id.get(w, UNK_ID)
                    sent.append(id)
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'sentence': _int_feature(sent)
                    })
                )
                writer.write(example.SerializeToString())
            else:
                continue
        writer.close()

    def dump_word_freq(self):
        filename = os.path.join(self._train_dir, "word_freq")
        with open(filename, 'w') as file:
            for (word, id) in self._word_freq:
                file.write(word)
                file.write(' ')
                file.write(str(id))
                file.write('\n')

FLAGS = None


def main(_):
    vocob = Vocab(FLAGS.train_dir, FLAGS.data_file, FLAGS.min_count)
    vocob.build_vocab()
    vocob.dump_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="train",
        help="The local path or hdfs path of result files"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data.txt",
        help="The data file that used to handle"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="The min word count"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)