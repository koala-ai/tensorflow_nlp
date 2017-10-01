# -*- coding:utf8 -*-

import collections
import numpy as np
from itertools import compress


def build_dataset(words, vocabulary_size=50000):
    '''
    Build the dictionary and replace rare words with UNK token.

    Parameters
    ----------
    words: list of tokens
    vocabulary_size: maximum number of top occurring tokens to produce, 
        rare tokens will be replaced by 'UNK'
    '''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()  # {word: index}
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list()  # collect index
        unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count  # list of tuples (word, count)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def build_doc_dataset(docs, vocabulary_size=50000):
    '''
    Build the dictionary and replace rare words with UNK token.

    Parameters
    ----------
    docs: list of token lists, each token list represent a sentence/document
    vocabulary_size: maximum number of top occurring tokens to produce, 
        rare tokens will be replaced by 'UNK'
    '''
    count = [['UNK', -1]]
    # words = reduce(lambda x,y: x+y, docs)
    words = []
    doc_ids = []  # collect document(sentence) indices
    for i, doc in enumerate(docs):
        doc_ids.extend([i] * len(doc))
        words.extend(doc)

    word_ids, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=vocabulary_size)

    return doc_ids, word_ids, count, dictionary, reverse_dictionary


def read_doc(data_file):
    lines = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            words = []
            word_tags = line.split()
            for word_tag in word_tags:
                words.append(word_tag.split("/")[0])
            lines.append(words)
    return lines


def generate_batch_pvdm(doc_ids, word_ids, batch_size, window_size):
    '''
    Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
    batch should be a shape of (batch_size, window_size+1)
    Parameters
    ----------
    doc_ids: list of document indices 
    word_ids: list of word indices
    batch_size: number of words in each mini-batch
    window_size: number of leading words before the target word 
    '''
    data_index = 0
    batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = window_size + 1
    buffer = collections.deque(maxlen=span)  # used for collecting word_ids[data_index] in the sliding window
    buffer_doc = collections.deque(maxlen=span)  # collecting id of documents in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    mask = [1] * span
    mask[-1] = 0
    i = 0
    while i < batch_size:
        if len(set(buffer_doc)) == 1:
            doc_id = buffer_doc[-1]
            # all leading words and the doc_id
            batch[i, :] = list(compress(buffer, mask)) + [doc_id]
            labels[i, 0] = buffer[-1]  # the last word at end of the sliding window
            i += 1
        # move the sliding window
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    return batch, labels


