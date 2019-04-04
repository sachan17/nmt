from __future__ import unicode_literals, print_function, division
from io import open
import random
import unicodedata
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import string

###### Globals #######
SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 10
######################
# re.sub(r'[.!?,“”""'']', r'', s)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def add_tokenized_sentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def reduce(self):
        temp_wc = {}
        temp_wi = {}
        temp_iw = {0: "SOS", 1: "EOS", 2:"UNK"}
        temp_ncount = 3
        for word in self.word2count.keys():
            if self.word2count[word] > 9:
                temp_wc[word] = self.word2count[word]
                temp_wi[word] = temp_ncount
                temp_iw[temp_ncount] = word
                temp_ncount += 1

        self.word2count = temp_wc
        self.word2index = temp_wi
        self.index2word = temp_iw
        self.n_words = temp_ncount

    def embeddings(self, fname):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
        embd = torch.zeros([self.n_words, 300])
        for i in range(self.n_words):
            if self.index2word[i] in data.keys():
                embd[i] = torch.FloatTensor(data[self.index2word[i]])
            else:
                embd[i] = torch.randn(300)
        return embd


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# def check(str):



def read_langs(datafile, lang1, lang2):
    data_file = open(datafile, 'r', encoding='utf-8')
    lines = data_file.readlines()

    garbage = string.punctuation+"0123456789“”। "
    standalone_garbage = list(set(string.ascii_letters).difference({'I', 'i', 'a', 'A'}).union({' ', ''}))

    table = str.maketrans(garbage, " "*len(garbage))
    pairs = [[[w.strip().lower() for w in pair.translate(table).split(' ') \
            if w.strip().lower().translate(table) not in standalone_garbage] \
            for pair in line.split('\t')] for line in lines]

    # pairs = []
    # for line in lines:
        # line = line.strip().split('\t')

        # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        # pairs.append([line[0], line[1]])

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepare_data(datafile, lang1, lang2):
    input_lang, output_lang, pairs = read_langs(datafile, lang1, lang2)
    # pairs = [pair for pair in pairs if len(pair[0]) < MAX_LENGTH]

    print("Training: %s sentence pairs" % len(pairs))

    for pair in pairs:
        input_lang.add_tokenized_sentence(pair[0])
        output_lang.add_tokenized_sentence(pair[1])
        # input_lang.addSentence(pair[0])
        # output_lang.addSentence(pair[1])
    input_lang.reduce()
    output_lang.reduce()
    print("Counted words:")

    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def read_test_file(datafile):
    _1, _2, pairs = read_langs(datafile, 'a', 'b')
    del(_1)
    del(_2)
    return pairs

if __name__ == '__main__':
    import pickle
    f = open('preprocess_en_hi.pickle', 'w')
    datafile = 'data/traindata.enhi.txt'
    # input_lang, output_lang, pairs = prepare_data(datafile, 'eng', 'hindi')
    # datafile = 'data/eng_ger_train.txt'
    print('Data File:', datafile)
    input_lang, output_lang, pairs = prepare_data(datafile, 'eng', 'ger')
    input_embd = input_lang.embeddings('data/eng.vec')
    output_embd = output_lang.embeddings('data/german.vec')
    training_pairs = [tensorsFromPair(pairs[i]) for i in range(len(pairs))]
    data = {"in":input_lang, "out": output_lang, "in_e":input_embd, "out_e": output_embd, 'p': pairs, 'tp': training_pairs}
    pickle.dump(data, f)
    f.close()
