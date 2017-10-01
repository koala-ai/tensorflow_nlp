# -*- coding:utf-8 -*-

import os
import codecs
import re
import collections
import numpy as np
import _pickle as pickle

UNKNOWN = "*"
DELIMITER = "\s+" # line delimiter


def _read_file(filename):
    sentences = []  # list(list(str))
    words = []
    file = codecs.open(filename, encoding='utf-8')
    for line in file:
        wordsplit = re.split(DELIMITER, line.replace("\n", ""))
        sentences.append(wordsplit)  # list(list(str))
        words.extend(wordsplit)  # list(str)
    return words, sentences


# input format word2/tag2 word2/tag2
def _split_word_tag(data):
    word = []
    tag = []
    for word_tag_pair in data:
        pairs = word_tag_pair.split("/")
        if len(pairs)==2:
            # word or tag not equal to ""
            if len(pairs[0].strip())!=0 and len(pairs[1].strip())!=0:
                word.append(pairs[0])
                tag.append(pairs[1])
    return word, tag


def _build_vocab(filename, utils_dir):

    vocab_file = os.path.join(utils_dir, "vocab.pkl")
    tag_file = os.path.join(utils_dir, "tags.pkl")

    if not os.path.exists(vocab_file) or not os.path.exists(tag_file):
        words, sentences = _read_file(filename)
        word, tag = _split_word_tag(words)

        word.append(UNKNOWN)
        counter_word = collections.Counter(word)
        count_pairs_word = sorted(counter_word.items(), key=lambda x: (-x[1], x[0]))

        wordlist, _ = list(zip(*count_pairs_word))
        word_to_id = dict(zip(wordlist, range(len(wordlist))))

        # tag dictionary
        tag.append(UNKNOWN)
        counter_tag = collections.Counter(tag)
        count_pairs_tag = sorted(counter_tag.items(), key=lambda x: (-x[1], x[0]))

        taglist, _ = list(zip(*count_pairs_tag))
        tag_to_id = dict(zip(taglist, range(len(taglist))))

        with open(vocab_file, 'wb') as f:
            pickle.dump(word_to_id, f)

        with open(tag_file, 'wb') as f:
            pickle.dump(tag_to_id, f)
    else:
        with open(vocab_file, 'rb') as f:
            word_to_id = pickle.load(f)

        with open(tag_file, 'rb') as f:
            tag_to_id = pickle.load(f)

    return word_to_id, tag_to_id


def sentence_to_word_ids(data_path, words):
  with open(os.path.join(data_path, "vocab.pkl"), 'rb') as f:
    word_to_id = pickle.load(f)
  wordArray = [word_to_id[w] if w in word_to_id else word_to_id[UNKNOWN] for w in words]
  return wordArray


def tag_ids_to_tags(data_path, ids):
  with open(os.path.join(data_path, "tags.pkl"), 'rb') as f:
      tag_to_id = pickle.load(f)
  id_to_tag = {id:tag for tag, id in tag_to_id.items()}
  tagArray = [id_to_tag[i] if i in id_to_tag else id_to_tag[0] for i in ids]
  return tagArray


def _file_to_word_ids(filename, word_to_id, tag_to_id, seq_length):
  _, sentences = _read_file(filename)
  word_array = np.zeros((len(sentences), seq_length), np.int32)
  tag_array = np.zeros((len(sentences), seq_length), np.int32)

  for index, sentence in enumerate(sentences):
      words, tags = _split_word_tag(sentence)
      word_ids = [word_to_id.get(w, word_to_id[UNKNOWN]) for w in words]
      tag_ids = [tag_to_id.get(w, tag_to_id[UNKNOWN]) for w in tags]

      if len(words) >= seq_length:
          word_ids = word_ids[:seq_length]
          tag_ids = tag_ids[:seq_length]
      else:
          rest_len = seq_length - len(words)
          word_ids.extend([word_to_id[UNKNOWN]] * rest_len)
          tag_ids.extend([tag_to_id[UNKNOWN]] * rest_len)
      word_array[index] = word_ids
      tag_array[index] = tag_ids

  return word_array, tag_array


def load_data(data_path, utils_dir, seq_length):

    data_file = os.path.join(data_path, "data.txt")
    train_path = os.path.join(data_path, "train.txt")
    dev_path = os.path.join(data_path, "dev.txt")

    # build vocab
    word_to_id, tag_to_id = _build_vocab(data_file, utils_dir)

    print("word dictionary size " + str(len(word_to_id)))
    print("tag dictionary size " + str(len(tag_to_id)))

    train_word, train_tag = _file_to_word_ids(train_path, word_to_id, tag_to_id, seq_length)
    print("train dataset: " + str(len(train_word)) + " " + str(len(train_tag)))
    dev_word, dev_tag = _file_to_word_ids(dev_path, word_to_id, tag_to_id, seq_length)
    print("dev dataset: " + str(len(dev_word)) + " " + str(len(dev_tag)))
    vocab_size = len(word_to_id)
    tag_size = len(tag_to_id)

    return train_word, train_tag, dev_word, dev_tag, vocab_size, tag_size
