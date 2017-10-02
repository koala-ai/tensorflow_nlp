# -*- coding:utf-8 -*-

import os
import codecs

import numpy as np
from six.moves import xrange
import tensorflow as tf

from nlp.textsum import model as seq2seq_model
from nlp.textsum.dataset import data_utils, dataset


def generate_summary(args):

    sentences = []
    summaries = []

    input_file = codecs.open(args.predict_file, encoding='utf-8')
    for line in input_file:
        sentences.append(line.replace("\n", "").encode('utf-8'))  # list of 'str', convert 'unicode' to 'str'

    # Load vocabularies.
    vocab_path = os.path.join(args.utils_dir, "vocab")
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    args.vocab_size = len(vocab)

    with tf.Session() as sess:
        # Create model and load parameters.
        model = seq2seq_model.create_model(sess, args.train_dir, args, True)
        model.batch_size = 1
        args.batch_size = 1

        for i in range(len(sentences)):
            sentence = sentences[i]
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
            bucket_id = min([b for b in xrange(len(args.buckets))
                             if args.buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = \
                dataset.DataSet(args, {bucket_id: [(token_ids, [])]}).next_batch(bucket_id)

            # Get output logits for the sentence.
            _, _, output_logits_batch = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                   True)
            output_logits = []
            for item in output_logits_batch:
                output_logits.append(item[0])

            outputs = [int(np.argmax(logit)) for logit in output_logits]
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]  # list of IDs

            summary = [tf.compat.as_str(rev_vocab[output]) for output in outputs]
            print(summary)
            summaries.append(summary)
            print(" ".join(summary))

    # Write Output to summary_dir
    summary_file = codecs.open(args.result_file, 'w', encoding='utf-8')
    for summary in summaries:
        line = " ".join(summary) + "\n"  # 'str' in 'utf-8' coding
        summary_file.write(line)  # write unicode to file