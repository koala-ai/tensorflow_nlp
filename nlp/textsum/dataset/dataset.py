# -*- coding:utf-8 -*-

import numpy as np
import random
from nlp.textsum.dataset import data_utils


class DataSet(object):
    def __init__(self, args, data):
        self._config = args
        self._start = 0
        self._index_in_epoch = 0
        self._data = data

    def next_batch(self, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
                To feed data in step(..) it must be a list of batch-major vectors, while
                data here contains single length-major cases. So the main logic of this
                function is to re-index data cases to be in the proper format for feeding.
                Args:
                  data: a tuple of size len(self.buckets) in which each element contains
                    lists of pairs of input and output data that we use to create a batch.
                  bucket_id: integer, which bucket to get the batch for.
                Returns:
                  The triple (encoder_inputs, decoder_inputs, target_weights) for
                  the constructed batch that has the proper format to call step(...) later.
                """
        encoder_size, decoder_size = self._config.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self._config.batch_size):
            encoder_input, decoder_input = random.choice(self._data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self._config.batch_size)], np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self._config.batch_size)], np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self._config.batch_size, np.float32)
            for batch_idx in xrange(self._config.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights