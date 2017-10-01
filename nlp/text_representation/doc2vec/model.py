# -*- coding:utf8 -*-

import math
import random
import numpy as np

import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from nlp.text_representation.doc2vec.dataset.data_utils import *

SEED = 2016
random.seed(SEED)
np.random.seed(SEED)


class Doc2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        # bind params to class
        self.batch_size = args.batch_size
        self.window_size = args.window_size
        self.concat = args.concat
        self.embedding_size_w = args.embedding_size_w
        self.embedding_size_d = args.embedding_size_d
        self.vocabulary_size = args.vocabulary_size
        self.document_size = args.document_size
        self.loss_type = args.loss_type
        self.n_neg_samples = args.n_neg_samples
        self.optimize = args.optimize
        self.learning_rate = args.learning_rate
        self.n_steps = args.n_steps
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.float32, shape=[self.batch_size, None])
            # Variables.
            # embeddings for words, W in paper
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for documents (can be sentences or paragraph), D in paper
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat:  # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
            else:  # concatenating the average of word vectors and the doc vector
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size], dtype=tf.float32))

            # Model.
            # Look up embeddings for inputs.
            # shape: (batch_size, embeddings_size)
            embed = []  # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_d)
            # concat word and doc vectors
            self.embed = tf.concat(embed, 1)

            # Compute the loss, using a sample of the negative labels each time.
            loss = None
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels, self.embed,
                                                  self.n_neg_samples, self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.train_labels, self.embed,
                                      self.n_neg_samples, self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / norm_d

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
