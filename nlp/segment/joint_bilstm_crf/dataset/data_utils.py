# -*- coding: utf-8 -*-

import numpy as np
import math
import os
import re
import tensorflow as tf
from nlp.segment.joint_bilstm_crf.evaluation import score


def get_ngrams(raw, gram):
    gram_set = set()
    li = gram // 2
    ri = gram - li - 1
    p = '<PAD>'
    for line in raw:
        for i in range(len(line)):
            if i - li < 0:
                lp = p * (li - i) + line[:i]
            else:
                lp = line[i - li:i]
            if i + ri + 1 > len(line):
                rp = line[i:] + p*(i + ri + 1 - len(line))
            else:
                rp = line[i:i+ri+1]
            ch = lp + rp
            gram_set.add(ch)
    return gram_set


'''
处理训练集和验证集获得下面的文件：
1 字符集
2 标签集，格式（标签 标签对应的最大词长）
3 二元词集
4 三元词集
'''
def get_vocab_tag(train_dir, fileList, ngram=1):
    out_char = open(train_dir + '/chars.txt', 'w', encoding='utf-8')
    out_tag = open(train_dir + '/tags.txt', 'w', encoding='utf-8')
    char_set = set()
    tag_set = {}
    raw = []
    for file_name in fileList:
        for line in open(file_name, 'r'):
            line = line.strip()
            raw_l = ''
            sets = line.split(' ')
            if len(sets) > 0:
                for seg in sets:
                    spos = seg.split('_')
                    if len(spos) == 2:
                        for ch in spos[0]:
                            char_set.add(ch)
                            raw_l += ch
                        if spos[1] in tag_set:
                            if tag_set[spos[1]] < len(spos[0]):
                                tag_set[spos[1]] = len(spos[0])
                        else:
                            tag_set[spos[1]] = len(spos[0])
                raw.append(raw_l)
            elif len(line) == 0:
                continue
            else:
                print(line)
                raise Exception('Check your text file.')

    char_set = list(char_set)
    if ngram > 1:
        for i in range(2, ngram + 1):
            out_gram = open(train_dir + '/' + str(i) + 'gram.txt', 'w', encoding='utf-8')
            grams = get_ngrams(raw, i)
            for g in grams:
                out_gram.write(g + '\n')
            out_gram.close()
    for item in char_set:
        out_char.write(item + '\n')
    out_char.close()
    for k, v in tag_set.items():
        out_tag.write(k + ' ' + str(v) + '\n')
    out_tag.close()

'''
读取上面方法获得的数据集
'''
def read_vocab_tag(path, ngrams=1):
    char_set = set()
    tag_set = {}
    ngram_set = None
    for line in open(path + '/chars.txt', 'r'):
        char_set.add(line.strip())
    for line in open(path + '/tags.txt', 'r'):
        line = line.strip()
        sp = line.split(' ')
        tag_set[sp[0]] = int(sp[1])
    char_set = list(char_set)
    if ngrams > 1:
        ngram_set = []
        for i in range(2, ngrams + 1):
            ng_set = set()
            for line in open(path + '/' + str(i) + 'gram.txt', 'r'):
                line = line.strip()
                ng_set.add(line)
            ngram_set.append(ng_set)
    return char_set, tag_set, ngram_set

'''
初始化<P>，<UNK>，<NUM>，<FW>的词向量，并获得字集合对应的词向量集合
'''
def get_sample_embedding(glove_file, path, short_emb, chars, default='unk'):
    # short_emb = emb[emb.index('/') + 1: emb.index('.')]
    emb_dic = {}
    for line in open(glove_file, 'r'):
        line = line.strip()
        sets = line.split(' ')
        emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    v = list(emb_dic.values())
    emb_dim = len(v[0])
    fout = open(path + '/' + short_emb + '_sub.txt', 'w', encoding='utf-8')
    p_line = '<P>'
    if '<P>' in emb_dic:
        for emb in emb_dic['<P>']:
            p_line += ' ' + str(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3)/emb_dim), math.sqrt(float(3)/ emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + str(emb)
    fout.write(p_line + '\n')
    p_line = '<UNK>'
    if '<UNK>' in emb_dic:
        for emb in emb_dic['<UNK>']:
            p_line += ' ' + str(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        emb_dic['<UNK>'] = rand_emb
        for emb in rand_emb:
            p_line += ' ' + str(emb)
    fout.write(p_line + '\n')

    p_line = '<NUM>'
    if '<NUM>' in emb_dic:
        for emb in emb_dic['<NUM>']:
            p_line += ' ' + str(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + str(emb)
    fout.write(p_line + '\n')

    p_line = '<FW>'
    if '<FW>' in emb_dic:
        for emb in emb_dic['<FW>']:
            p_line += ' ' + str(emb)
    else:
        rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        for emb in rand_emb:
            p_line += ' ' + str(emb)
    fout.write(p_line + '\n')

    for ch in chars:
        p_line = ch
        if ch in emb_dic:
            for emb in emb_dic[ch]:
                p_line += ' ' + str(emb)
        else:
            if default == 'unk':
                for emb in emb_dic['<UNK>']:
                    p_line += ' ' + str(emb)
            else:
                rand_emb = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
                for emb in rand_emb:
                    p_line += ' ' + str(emb)
        fout.write(p_line + '\n')
    fout.close()

def read_sample_embedding(path, short_emb):
    emb_values = []
    for line in open(path + '/' + short_emb + '_sub.txt', 'r'):
        line = line.strip()
        sets = line.split(' ')
        emb_values.append(np.asarray(sets[1:], dtype='float32'))
    emb_dim = len(emb_values[0])
    return emb_dim, emb_values


def get_comb_tags(tags, tag_type):
    tag2index = {}
    tag2index['<P>'] = 0
    idx = 1
    for k, v in tags.items():
        real_tag_type = tag_type
        if v == 1:
            if tag_type == 'BIES':
                real_tag_type = tag_type[-1:]
            else:
                real_tag_type = tag_type[0]
        elif v == 2:
            if tag_type == 'BIES' or tag_type == 'BIE':
                real_tag_type = tag_type[: 1] + tag_type[-2:]
        for t_type in real_tag_type:
            tag2index[str(t_type + '-' + k)] = idx
            idx += 1
    return tag2index

'''
获取标签及相应的index
'''
def get_dic(chars, tags):
    char2index = {}
    char2index['<P>'] = 0
    char2index['<UNK>'] = 1
    char2index['<NUM>'] = 2
    char2index['<FW>'] = 3
    idx = 4
    for ch in chars:
        char2index[ch] = idx
        idx += 1
    index2char = {v: k for k, v in char2index.items()}

    #0.seg BIES  1. BI; 2. BIE; 3. BIES
    seg_tags2index = {'<P>':0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    tag2index = {'seg': seg_tags2index,
                 'BIES': get_comb_tags(tags, 'BIES')}
    index2tag = {}
    for dic_keys in tag2index:
        index2tag[dic_keys] = {v: k for k, v in tag2index[dic_keys].items()}
    return char2index, index2char, tag2index, index2tag

'''
获取每个字符的id以及它对应的tag id
'''
def get_input_vec(fname, char2index, tag2index, tag_scheme='BIES'):
    max_sent_len_c = 0 # 句子的最大字数
    max_sent_len_w = 0 # 句子的最大词数
    max_word_len = 0 # 句子的最大词长
    t_len = 0
    x_m = [[]]
    y_m = [[]]
    for line in open(fname, 'r', encoding='utf-8'):
        charIndices = []
        raw_l = ''
        tagIndices = {}

        for k in tag2index.keys():
            tagIndices[k] = []
        line = line.strip()
        segs = line.split(' ')

        if len(segs) > max_sent_len_w:
            max_sent_len_w = len(segs)
        if len(segs) > 0 and len(line) > 0:
            for seg in segs:
                splits = seg.split('_')
                assert len(splits) == 2

                w_len = len(splits[0])
                raw_l += splits[0]
                if w_len > max_word_len:
                    max_word_len = w_len

                t_len += w_len

                if w_len == 1:
                    charIndices.append(char2index[splits[0]])
                    tagIndices['seg'].append(tag2index['seg']['S'])
                    tagIndices['BIES'].append(tag2index['BIES']['S-' + splits[1]])
                else:
                    for x in range(w_len):
                        c_ch = splits[0][x]
                        charIndices.append(char2index[c_ch])
                        if x == 0:
                            tagIndices['seg'].append(tag2index['seg']['B'])
                            tagIndices['BIES'].append(tag2index['BIES']['B-' + splits[1]])
                        elif x == len(splits[0]) - 1:
                            tagIndices['seg'].append(tag2index['seg']['E'])
                            tagIndices['BIES'].append(tag2index['BIES']['E-' + splits[1]])
                        else:
                            tagIndices['seg'].append(tag2index['seg']['I'])
                            tagIndices['BIES'].append(tag2index['BIES']['I-' + splits[1]])

            if t_len > max_sent_len_c:
                max_sent_len_c = t_len
            t_len = 0
            x_m[0].append(charIndices)
            y_m[0].append(tagIndices[tag_scheme])
    return x_m, y_m, max_sent_len_c, max_sent_len_w, max_word_len


def get_ngram_dic(ngrams):
    gram_dics = []
    for i, gram in enumerate(ngrams):
        g_dic = {}
        g_dic['<P>'] = 0
        g_dic['<UNK>'] = 1
        idx = 2
        for g in gram:
            g_dic[g] = idx
            idx += 1
        gram_dics.append(g_dic)
    return gram_dics

def gram_vec(raw, dic):
    out = []
    keys = list(dic.keys())
    ngram = len(keys[0])
    li = ngram // 2
    ri = ngram - li - 1
    p = '<PAD>'
    for line in raw:
        indices = []
        for i in range(len(line)):
            if i - li < 0:
                lp = p * (li - i) + line[:i]
            else:
                lp = line[i - li:i]
            if i + ri + 1 > len(line):
                rp = line[i:] + p*(i + ri + 1 - len(line))
            else:
                rp = line[i:i+ri+1]
            ch = lp + rp
            if ch in dic:
                indices.append(dic[ch])
            else:
                indices.append(dic['<UNK>'])
        out.append(indices)
    return out

# 获取多元词的向量
def get_gram_vec(fname, gram2index, is_raw=False):
    raw = []
    if is_raw:
        for line in open(fname, 'r', encoding='utf-8'):
            line = line.strip()
            raw.append(line)
    else:
        for line in open(fname, 'r', encoding='utf-8'):
            line = line.strip()
            segs = line.split(' ')
            if len(segs) > 0 and len(line) > 0:
                raw_l = ''
                for seg in segs:
                    sp = seg.split('_')
                    if len(sp) == 2:
                        raw_l += sp[0]
                raw.append(raw_l)
    out = []
    for g_dic in gram2index:
        out.append(gram_vec(raw, g_dic))
    return out


# 将不同的句子分桶，桶为[0-10],[11-20],[21,30] ...
def buckets(x, y, size=10):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    samples = x + y
    num_items = len(samples)
    xy = zip(*samples)
    xy = sorted(xy, key=lambda i: len(i[0]))
    # xy.sort(key=lambda i: len(i[0]))
    t_len = size
    idx = 0
    bucks = [[[]] for _ in range(num_items)]
    for item in xy:
        if len(item[0]) > t_len:
            if len(bucks[0][idx]) > 0:
                for buck in bucks:
                    buck.append([])
                idx += 1
            while len(item[0]) > t_len:
                t_len += size
        for i in range(num_items):
            bucks[i][idx].append(item[i])

    return bucks[:num_inputs], bucks[num_inputs:]

def pad_zeros(l, max_len):
    if type(l) is list:
        return [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in l]
    elif type(l) is dict:
        padded = {}
        for k, v in l.iteritems():
            padded[k] = [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in v]
        return padded

def unpad_zeros(l):
    out = []
    for tags in l:
        out.append([np.trim_zeros(line) for line in tags])
    return out

# 将不满足长度的句子填充0
def pad_bucket(x, y, bucket_len_c=None):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    num_tags = len(y)
    padded = [[] for _ in range(num_tags + num_inputs)]
    bucket_counts = []
    samples = x + y
    xy = zip(*samples)
    if bucket_len_c is None:
        bucket_len_c = []
        for item in xy:
            max_len = len(item[0][-1])
            bucket_len_c.append(max_len)
            bucket_counts.append(len(item[0]))
            for idx in range(num_tags + num_inputs):
                padded[idx].append(pad_zeros(item[idx], max_len))
        print('Number of buckets: ', len(bucket_len_c))
    else:
        idy = 0
        for item in xy:
            max_len = len(item[0][-1])
            while idy < len(bucket_len_c) and max_len > bucket_len_c[idy]:
                idy += 1
            bucket_counts.append(len(item[0]))
            if idy >= len(bucket_len_c):
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], max_len))
                bucket_len_c.append(max_len)
            else:
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], bucket_len_c[idy]))

    return padded[:num_inputs], padded[num_inputs:], bucket_len_c, bucket_counts

def merge_bucket(x):
    out = []
    for item in x:
        m = []
        for i in item:
            m += i
        out.append(m)
    return out

# 获取标签个数
def get_nums_tags(tag2idx, tag_scheme):
    nums_tags = [len(tag2idx[tag_scheme])]
    return nums_tags
# 获取batch
def get_real_batch(counts, b_size):
    real_batch_sizes = []
    for c in counts:
        if c < b_size:
            real_batch_sizes.append(c)
        else:
            real_batch_sizes.append(b_size)
    return real_batch_sizes

def decode_tags(idx, index2tags, tag_scheme):
    out = []
    dic = index2tags[tag_scheme]
    for id in idx:
        sents = []
        for line in id:
            sent = []
            for item in line:
                tag = dic[item]
                if '-' in tag:
                    tag = tag.replace('E-', 'I-')
                    tag = tag.replace('S-', 'B-')
                else:
                    tag = tag.replace('E', 'I')
                    tag = tag.replace('S', 'B')
                sent.append(tag)
            sents.append(sent)
        out.append(sents)
    return out


def decode_chars(idx, idx2chars):
    out = []
    for line in idx:
        line = np.trim_zeros(line)
        out.append([idx2chars[item] for item in line])
    return out

def generate_output(chars, tags, tag_scheme):
    out = []
    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        for chs, tgs in zip(chars, tag):
            #print len(chs), len(tgs)
            assert len(chs) == len(tgs)
            c_word = ''
            c_tag = ''
            p_line = ''
            for ch, tg in zip(chs, tgs):
                if tag_scheme == 'seg':
                    if tg == 'I':
                        c_word += ch
                    else:
                        p_line += ' ' + c_word + '_' + '<UNK>'
                        c_word = ch
                else:
                    tg_sets = tg.split('-')
                    if tg_sets[0] == 'I' and tg_sets[1] == c_tag:
                        c_word += ch
                    else:
                        p_line += ' ' + c_word + '_' + c_tag
                        c_word = ch
                        if len(tg_sets) < 2:
                            c_tag = '<UNK>'
                        else:
                            c_tag = tg_sets[1]
            if len(c_word) > 0:
                if tag_scheme == 'seg':
                    p_line += ' ' + c_word + '_' + '<UNK>'
                elif len(c_tag) > 0:
                    p_line += ' ' + c_word + '_' + c_tag
            if tag_scheme == 'seg':
                sub_out.append(p_line[8:])
            else:
                sub_out.append(p_line[3:])
        out.append(sub_out)
    return out

def evaluator(prediction, gold, tag_scheme='BIES', verbose=False):
    assert len(prediction) == len(gold)
    scores = score(gold[0], prediction[0],  verbose)
    print('Segmentation F-score: %f' % scores[0])
    if tag_scheme != 'seg':
        print('Tagging F-score: %f' % scores[1])
    scores = [scores]
    return scores

def viterbi(max_scores, max_scores_pre, length, batch_size):
    best_paths = []
    for m in range(batch_size):
        path = []
        last_max_node = np.argmax(max_scores[m][length[m] - 1])
        path.append(last_max_node)
        for t in range(1, length[m])[::-1]:
            last_max_node = max_scores_pre[m][t][last_max_node]
            path.append(last_max_node)
        path = path[::-1]
        best_paths.append(path)
    return best_paths

def trim_output(out, length):
    assert len(out) == len(length)
    trimmed_out = []
    for item, l in zip(out, length):
        trimmed_out.append(item[:l])
    return trimmed_out

def get_new_chars(path, char2idx, type='ctb'):
    new_chars = set()
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if type == 'ctb':
            segs = line.split(' ')
            for seg in segs:
                items = seg.split('_')
                assert len(items) == 2
                for ch in items[0]:
                    if ch not in char2idx:
                        new_chars.add(ch)
        else:
            line = re.sub('[\s+]', '', line)
            for ch in line:
                if ch not in char2idx:
                    new_chars.add(ch)
    return new_chars

def get_new_grams(path, gram2idx, type='ctb'):
    raw = []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        raw_l = ''
        if type == 'ctb':
            segs = line.split(' ')
            for seg in segs:
                items = seg.split('_')
                assert len(items) == 2
                raw_l += items[0]
        else:
            line = re.sub('[\s+]', '', line)
            raw_l = line
        raw.append(raw_l)
    new_grams = []
    for g_dic in gram2idx:
        new_g = []
        n = len(g_dic.keys()[0])
        grams = get_ngrams(raw, n)
        for g in grams:
            if g not in g_dic:
                new_g.append(g)
        new_grams.append(new_g)
    return new_grams

def update_char_dict(char2idx, new_chars):
    dim = len(char2idx)
    for char in new_chars:
        if char not in char2idx:
            char2idx[char] = dim
            dim += 1
    idx2char = {k: v for v, k in char2idx.items()}
    return char2idx, idx2char

def update_gram_dicts(gram2idx, new_grams):
    assert len(gram2idx) == len(new_grams)
    new_gram2idx = []
    for dic, n_gram in zip(gram2idx, new_grams):
        assert len(dic.keys()[0]) == len(n_gram[0])
        new_dic, _ = update_char_dict(dic, n_gram)
        new_gram2idx.append(new_dic)
    return new_gram2idx

def get_input_vec_raw(fname, char2index):
    max_len = 0
    x_m = [[]]

    for line in open(fname, 'r', encoding='utf-8'):
        charIndices = []
        line = re.sub('[\s+]', '', line)
        if len(line) > max_len:
            max_len = len(line)
        for ch in line:
            charIndices.append(char2index[ch])
        x_m[0].append(charIndices)
    return x_m, max_len

def get_maxstep(raw_file):
    maxstep = 0
    for line in open(raw_file, 'r', encoding='utf-8'):
        line = line.strip()
        if len(line) > maxstep:
            maxstep = len(line)
    return maxstep

def get_new_embeddings(new_chars, emb_dim, emb_path=None):
    if emb_path is None:
        return tf.random_uniform([len(new_chars), emb_dim], -math.sqrt(3 / emb_dim), math.sqrt(3 / emb_dim))
    else:
        assert os.path.isfile(emb_path)
        emb = {}
        new_emb = []
        for line in open(emb_path, 'r'):
            line = line.strip()
            sets = line.split(' ')
            emb[sets[0]] = np.asarray(sets[1:], dtype='float32')
        if '<UNK>' not in emb:
            unk = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
            emb['<UNK>'] = np.asarray(unk, dtype='float32')
        for ch in new_chars:
            if ch in emb:
                new_emb.append(emb[ch])
            else:
                new_emb.append(emb['<UNK>'])
        return tf.stack(new_emb)


def get_new_ng_embeddings(new_grams, emb_dim, emb_path=None):
    new_embs = []
    for i in range(len(new_grams)):
        n = len(new_grams[i][0])
        if emb_path is not None:
            real_path = emb_path + '_' + str(n) + 'gram.txt'
        else:
            real_path = None
        n_emb = get_new_embeddings(new_grams[i], emb_dim, real_path)
        new_embs.append(n_emb)
    return new_embs

def get_gram_vec_raw(raw, gram2index):
    out = []
    for g_dic in gram2index:
        out.append(gram_vec(raw, g_dic))
    return out

def get_input_vec_line(lines, char2index):
    max_len = 0
    x_m = [[]]
    for line in lines:
        charIndices = []
        line = re.sub('[\s+]', '', line)
        if len(line) > max_len:
            max_len = len(line)
        for ch in line:
            charIndices.append(char2index[ch])
        x_m[0].append(charIndices)
    return x_m, max_len

def printer(predictions, out_path):
    fout = open(out_path, 'w', encoding='utf-8')
    for line in predictions:
        fout.write(line + '\n')
    fout.close()