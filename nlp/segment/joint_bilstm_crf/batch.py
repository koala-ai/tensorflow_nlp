# -*- coding: utf-8 -*-

import random
import numpy as np

from nlp.segment.joint_bilstm_crf.dataset import data_utils


def train(sess, model, batch_size, config, lr, lrv,
          data, dr=None, drv=None, verbose=False):
    assert len(data) == len(model)
    num_items = len(data)
    samples = list(zip(*data))
    random.shuffle(samples)
    start_idx = 0
    n_samples = len(samples)
    model.append(lr)
    if dr is not None:
        model.append(dr)

    while start_idx < len(samples):
        if verbose:
            print('%d' % (start_idx * 100 / n_samples) + '%')
        next_batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch_size = len(next_batch_samples)
        if real_batch_size < batch_size:
            next_batch_samples.extend(samples[:batch_size - real_batch_size])
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_samples])
        holders.append(lrv)
        if dr is not None:
            holders.append(drv)

        sess.run(config, feed_dict={m: h for m, h in zip(model, holders)})
        start_idx += batch_size


def predict(sess, model, data, dr=None, transitions=None, crf=True, decode_sess=None,
            scores=None, decode_holders=None, argmax=True, batch_size=100,
            ensemble=False, verbose=False):
    en_num = None
    if ensemble:
        en_num = len(sess)
    num_items = len(data)
    input_v = model[:num_items]
    if dr is not None:
        input_v.append(dr)

    predictions = model[num_items:]
    output = [[] for _ in range(len(predictions))]
    samples = list(zip(*data))
    start_idx = 0
    n_samples = len(samples)
    if crf:
        trans = []
        for i in range(len(predictions)):
            if ensemble:
                en_trans = 0
                for en_sess in sess:
                    en_trans += en_sess.run(transitions[i])
                trans.append(en_trans/en_num)
            else:
                trans.append(sess.run(transitions[i]))
    while start_idx < n_samples:
        if verbose:
            print('%d' % (start_idx*100/n_samples) + '%')
        next_batch_input = samples[start_idx:start_idx + batch_size]
        batch_size = len(next_batch_input)
        holders= []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_input])
        if dr is not None:
            holders.append(0.0)

        length = np.sum(np.sign(holders[0]), axis=1)
        if crf:
            assert transitions is not None and len(transitions) == len(predictions) and len(scores) == len(decode_holders)
            for i in range(len(predictions)):
                if ensemble:
                    en_obs = 0
                    for en_sess in sess:
                        en_obs += en_sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                    ob = en_obs/en_num
                else:
                    ob = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre_values = [ob, trans[i], length, batch_size]
                assert len(pre_values) == len(decode_holders[i])
                max_scores, max_scores_pre = decode_sess.run(scores[i], feed_dict={i: h for i, h in zip(decode_holders[i], pre_values)})
                output[i].extend(data_utils.viterbi(max_scores, max_scores_pre, length, batch_size))
        elif argmax:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = np.argmax(pre, axis=2)
                pre = pre.tolist()
                pre = data_utils.trim_output(pre, length)
                output[i].extend(pre)
        else:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = pre.tolist()
                pre = data_utils.trim_output(pre, length)
                output[i].extend(pre)
        start_idx += batch_size
    return output