# -*- coding: utf-8 -*-

import os
from random import shuffle
import tensorflow as tf
from nlp.text_representation.glove.dataset.dataset import *
from nlp.text_representation.glove.model import GloVeModel


def train(args):
    corpus = build_corpus(os.path.join(args.data_dir, "train_little.txt"))
    if isinstance(args.context_size, tuple):
        left_context, right_context = args.context_size
    elif isinstance(args.context_size, int):
        left_context = right_context = args.context_size
    else:
        raise ValueError("`context_size` should be an int or a tuple of two ints")

    args.left_size = left_context
    args.right_size = right_context

    words, _, batches = Database(corpus, args).prepare_batches()

    model = GloVeModel(args)
    with tf.Session(graph=model.graph) as session:
        tf.global_variables_initializer().run()
        for epoch in range(args.num_epochs):
            print("epoch: %d" % epoch)
            shuffle(batches)
            for batch_index, batch in enumerate(batches):
                i_s, j_s, counts = batch
                if len(counts) != args.batch_size:
                    continue
                feed_dict = {model.focal_input: i_s, model.context_input: j_s, model.cooccurrence_count: counts}
                session.run([model.optimizer], feed_dict=feed_dict)

            if (epoch + 1) % args.tsne_epoch_interval == 0:
                current_embeddings = model.combined_embeddings.eval()
                output_path = os.path.join(args.log_dir, "epoch{:03d}.png".format(epoch + 1))
                generate_tsne(path=output_path, words=words, embeddings=current_embeddings)

        embeddings = model.combined_embeddings.eval()
        with open(args.result_file, 'w') as fout:
            for i, word in enumerate(words):
                fout.write(word + " ")
                fout.write(','.join(str(value) for value in embeddings[i]))
                fout.write("\n")
        print('model saved in %s' % args.result_file)


def generate_tsne(path=None, words = None, size=(100, 100), word_count=1000, embeddings=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
    labels = words[:word_count]
    return plot_with_labels(low_dim_embs, labels, path, size)


def plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                             va='bottom')
    if path is not None:
        figure.savefig(path)

    plt.close(figure)