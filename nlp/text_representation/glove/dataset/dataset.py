# -*- coding: utf-8 -*-

from collections import Counter, defaultdict


class Database(object):
    def __init__(self, corpus, args):
        self.corpus = corpus
        self.batch_size = args.batch_size
        self.left_size = args.left_size
        self.right_size = args.right_size
        self.vocab_size = args.max_vocab_size
        self.min_occurrences = args.min_occurrences

    def window(self, region, start_index, end_index):
        """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from region. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in region, this function will pad
        its return value with `NULL_WORD`.
        """
        last_index = len(region) + 1
        selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
        return selected_tokens

    def context_windows(self, region):
        for i, word in enumerate(region):
            start_index = i - self.left_size
            end_index = i + self.right_size
            left_context = self.window(region, start_index, i - 1)
            right_context = self.window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    def fit_to_corpus(self):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in self.corpus:
            word_counts.update(region)
            for l_context, word, r_context in self.context_windows(region):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        words = [word for word, count in word_counts.most_common(self.vocab_size)
                        if count >= self.min_occurrences]
        word_to_id = {word: i for i, word in enumerate(words)}
        cooccurrence_matrix = {
            (word_to_id[words[0]], word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in word_to_id and words[1] in word_to_id}

        return words, word_to_id, cooccurrence_matrix

    def prepare_batches(self):
        words, word_to_id, cooccurrence_matrix = self.fit_to_corpus()
        cooccurrences = [(word_ids[0], word_ids[1], count)
                             for word_ids, count in cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return words, word_to_id, list(batchify(self.batch_size, i_indices, j_indices, counts))


def batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i + batch_size] for sequence in sequences)


def build_corpus(path):
    print("build corpus")
    corpus = []
    with open(path) as file_:
        for line in file_:
            words = []
            word_tags = line.split()
            for word_tag in word_tags:
                words.append(word_tag.split("/")[0])
            corpus.append(words)
    return corpus


