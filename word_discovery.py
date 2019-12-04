# ! -*- coding: utf-8 -*-

import struct
import os
import math
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format=u'%(asctime)s - %(levelname)s - %(message)s')

from tqdm import tqdm


class KenlmNgrams:
    """加载Kenlm的ngram统计结果
    vocab_file: Kenlm统计出来的词(字)表；
    ngram_file: Kenlm统计出来的ngram表；
    order: 统计ngram时设置的n，必须跟ngram_file对应；
    min_count: 自行设置的截断频数。
    """

    def __init__(self, vocab_file, ngram_file, order, min_count):
        self.vocab_file = vocab_file
        self.ngram_file = ngram_file
        self.order = order
        self.min_count = min_count
        self.read_chars()
        self.read_ngrams()

    def read_chars(self):
        f = open(self.vocab_file, encoding='utf-8')
        chars = f.read()
        f.close()
        chars = chars.split('\x00')
        self.chars = chars

    def read_ngrams(self):
        """读取思路参考https://github.com/kpu/kenlm/issues/201
        """
        self.ngrams = [{} for _ in range(self.order)]
        self.total = 0
        size_per_item = self.order * 4 + 8
        f = open(self.ngram_file, 'rb')
        filedata = f.read()
        filesize = f.tell()
        f.close()
        for i in tqdm(range(0, filesize, size_per_item)):
            s = filedata[i: i + size_per_item]
            n = self.unpack('l', s[-8:])
            if n >= self.min_count:
                self.total += n
                c = [self.unpack('i', s[j * 4: (j + 1) * 4]) for j in range(self.order)]
                c = [self.chars[j] for j in c if j > 2]
                for j in range(len(c)):
                    self.ngrams[j][' '.join(c[:j + 1])] = self.ngrams[j].get(' '.join(c[:j + 1]), 0) + n

    def unpack(self, t, s):
        return struct.unpack(t, s)[0]


def write_corpus(texts, corpus):
    """将语料写到文件中，词与词(字与字)之间用空格隔开
    """
    print('exporting corpus...')
    with open(corpus, 'w', encoding='utf-8') as f:
        for s in texts:
            f.write(s)


def write_vocab(corpus, vocab):
    print('writing vocab...')

    tmp = open(corpus, encoding='utf-8').read().split()
    words = []
    for w in tqdm(tmp):
        w = w.strip('.')
        w = w.strip(',')
        w = w.strip(')')
        w = w.strip('(')
        w = w.strip('\'')
        w = w.strip('?')
        w = w.strip('!')

        w = w.strip('.')
        w = w.strip(',')
        w = w.strip(')')
        w = w.strip('(')
        w = w.strip('\'')
        w = w.strip('?')
        w = w.strip('!')
        words.append(w)
    words = list(Counter(words).items())
    words.sort(key=lambda k: k[1], reverse=True)
    with open(vocab, 'w', encoding='utf-8') as f:
        for w in words:
            # f.write(w[0] + ' ' + str(w[1])+'\n')
            f.write(w[0] + '\n')


def count_ngrams(corpus_file, order, vocab_file, ngram_file):
    """通过os.system调用Kenlm的count_ngrams来统计频数
    """
    return os.system(
        '/home/yhj/paper/ijcai-2020/kenlm/build/bin/count_ngrams -o %s --write_vocab_list %s <%s >%s'
        % (order, vocab_file, corpus_file, ngram_file)
    )


def filter_ngrams(ngrams, total, min_pmi=1):
    """通过互信息过滤ngrams，只保留“结实”的ngram。
    """
    order = len(ngrams)
    if hasattr(min_pmi, '__iter__'):
        min_pmi = list(min_pmi)
    else:
        min_pmi = [min_pmi] * order
    output_ngrams = set()
    total = float(total)
    for i in range(order - 1, 0, -1):
        print('order: ', i)
        for w, v in tqdm(ngrams[i].items()):
            w = w.split(' ')
            pmi = min([
                total * v / (ngrams[j].get(' '.join(w[:j + 1]), total) * ngrams[i - j - 1].get(' '.join(w[j + 1:]),
                                                                                               total))
                for j in range(i)
            ])
            if math.log(pmi) >= min_pmi[i]:
                output_ngrams.add(' '.join(w))
    return output_ngrams


class SimpleTrie:
    """通过Trie树结构，来搜索ngrams组成的连续片段
    """

    def __init__(self):
        self.dic = {}
        self.end = True

    def add_word(self, word):
        _ = self.dic
        for c in word:
            if c not in _:
                _[c] = {}
            _ = _[c]
        _[self.end] = word

    def tokenize(self, sent):  # 通过最长联接的方式来对句子进行分词
        result = []
        start, end = 0, 1
        for i, c1 in tqdm(enumerate(sent), total=len(sent)):
            _ = self.dic
            if i == end:
                result.append(sent[start: end])
                start, end = i, i + 1
            for j, c2 in enumerate(sent[i:]):
                if c2 in _:
                    _ = _[c2]
                    if self.end in _:
                        if i + j + 1 > end:
                            end = i + j + 1
                else:
                    break
        result.append(sent[start: end])
        return result


def filter_vocab(candidates, ngrams, order):
    """通过与ngrams对比，排除可能出来的不牢固的词汇(回溯)
    """
    result = {}
    for i, j in candidates.items():
        if len(i) < 3:
            result[i] = j
        elif len(i) <= order and i in ngrams:
            result[i] = j
        elif len(i) > order:
            flag = True
            for k in range(len(i) + 1 - order):
                if i[k: k + order] not in ngrams:
                    flag = False
            if flag:
                result[i] = j
    return result


# ======= 算法构建完毕，下面开始执行完整的构建词库流程 =======

import re
import glob
import json


# 语料生成器，并且初步预处理语料
# 这个生成器例子的具体含义不重要，只需要知道它就是逐句地把文本yield出来就行了
def text_generator():
    txts = [os.path.join('./data', each) for each in os.listdir('./data')]
    for txt in txts:
        d = open(txt, encoding='utf-8').read()
        d = d.split('\n')
        res = ''
        for line in d:
            if '\t' in line:
                line = line.split('\t')[1]
            line.rstrip('.')
            line.strip()
            res += line + ' '

        yield res


min_count = 4
order = 6
corpus_file = 'scierc.corpus'  # 语料保存的文件名
vocab_file = 'scierc.vocab'  # 字符集保存的文件名
ngram_file = 'scierc.ngrams'  # ngram集保存的文件名
output_file = 'scierc.phrase'  # 最后导出的词表文件名
chars_file = 'scierc.chars'
ngrams_json = 'scierc.ngrams.json'

# write_corpus(text_generator(), corpus_file)  # 将语料转存为文本
# write_vocab(corpus_file, vocab_file)

# count_ngrams(corpus_file, order, chars_file, ngram_file)  # 用Kenlm统计ngram

# ngrams = KenlmNgrams(chars_file, ngram_file, order, min_count)

# json.dump({'ngrams': ngrams.ngrams, 'total': ngrams.total}, open(ngrams_json, 'w'), indent=4)

ngrams = json.load(open(ngrams_json))
ngrams = filter_ngrams(ngrams['ngrams'], ngrams['total'], [0, 2, 4, 6, 8, 10])  # 过滤ngram
ngrams = list(ngrams)
ngrams.sort(key=lambda k: (k, len(k)))
with open(output_file, 'w') as f:
    f.write('\n'.join(ngrams))

# ngtrie = SimpleTrie()  # 构建ngram的Trie树
# print('build ngram trie...')
# for w in tqdm(ngrams):
#     _ = ngtrie.add_word(w)
#
# candidates = {}  # 得到候选词
# print('discovering words...')
# txts = [os.path.join('./data', each) for each in os.listdir('./data')]
# for txt in txts:
#     d = open(txt, encoding='utf-8').read()
#     d = d.replace(u'\u3000', ' ').strip()
#     d = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', d)
#     print(txt, 'tokenizing...')
#     tokens = ngtrie.tokenize(d)
#     print(txt, 'gen candidates...')
#     for w in tqdm(tokens):  # 预分词
#         candidates[w] = candidates.get(w, 0) + 1
#
# # 频数过滤
# candidates = {i: j for i, j in candidates.items() if j >= min_count}
# # 互信息过滤(回溯)
# candidates = filter_vocab(candidates, ngrams, order)
#
# # 输出结果文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     for i, j in sorted(candidates.items(), key=lambda s: -s[1]):
#         s = '%s %s\n' % (i, j)
#         f.write(s)
