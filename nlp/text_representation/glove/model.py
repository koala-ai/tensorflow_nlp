# -*- coding: utf-8 -*-

import tensorflow as tf


class GloVeModel(object):
    def __init__(self, args):
        self.embedding_size = args.embedding_size
        self.min_occurrences = args.min_occurrences
        self.scaling_factor = args.scaling_factor
        self.cooccurrence_cap = args.cooccurrence_cap
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.vocab_size = args.max_vocab_size
        self.build_graph()

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), self.graph.device(device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.total_loss)
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)
            self.summary = tf.summary.merge_all()

            self.combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")


def device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"