# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import codecs
import pickle

class Vocab():
    def __init__(self, fname, voc={}, ivoc={}, mode=0):
        self.vocab = voc
        self.ivocab= ivoc
        self.dataset = []
        self.vocab["<eos>"] = 0
        self.ivocab[self.vocab["<eos>"]] = "<eos>"
        unko = codecs.open("./data/%s" % (fname),"r","utf_8_sig")
        self.maxlen = 0
        for line in unko:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if mode == 1: # word mode
                line = line.split()
            data = [] #np.empty((len(line) + 1,), dtype=np.int32)
            for i in xrange(len(line)):
                word = line[i]
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    self.ivocab[self.vocab[word]] = word
                data.append(self.vocab[word])
            data.append(0) # last
            self.dataset.append(data)
            self.maxlen = max(self.maxlen, len(line)+1)
        n_vocab = len(self.vocab)
        n_dataset = len(self.dataset)
        print "file=\"{}\"  mode={} Nchars={} Ndata={} Maxlen={}".format(fname, mode, n_vocab, n_dataset, self.maxlen)


    def id2word(self, id):
        if id < len(self.ivocab):
            return self.ivocab[id]
        else:
            return "???"

    def word2id(self, word):
        return self.vocab[word]

    def listid(self):
        for i in xrange(len(self.ivocab)):
            print i, self.id2word(i)


enf = ['data.txt', 'suusiki.txt', 'test_data.txt', 'test_suusiki.txt']
jaf = ['data.txt', 'suusiki.txt', 'test_data.txt', 'test_suusiki.txt']

def readv(dfs, mode=0):
    vocab = {}
    ivocab = {}
    vocab["<eos>"] = 0
    ivocab[vocab["<eos>"]] = "<eos>"
    total = 0
    for f in dfs:
        unko = codecs.open("./data/%s" % (f),"r","utf_8_sig")
        for line in unko:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if mode == 1: # word mode
                line = line.split()
            total += len(line)
            for i in xrange(len(line)):
                word = line[i]
                if word not in vocab:
                    vocab[word] = len(vocab)
                    ivocab[vocab[word]] = word
    lv = len(vocab)
    liv = len(ivocab)
    print "Nvocab={0} Nivocab={1}".format(lv, liv),
    print "total counts:", total
    return vocab, ivocab

if __name__ == "__main__":
    if True: # char dictionary
        worch = 0
        print 'Ja Chars:',
        jav, jaiv = readv(jaf,worch)
        print 'En Chars:',
        env, eniv = readv(enf,worch)
        mydict = {'jav':jav, 'jaiv':jaiv, 'env':env, 'eniv':eniv}
        with open("./mychar", 'w') as f:
            pickle.dump(mydict, f)
            # Nvocab=1948 Nivocab=1948  Ja
            # Nvocab=48 Nivocab=48      En

    if True: # word dictionary
        worch = 1
        print 'Ja Words:',
        jav, jaiv = readv(jaf,worch)
        print 'En Words:',
        env, eniv = readv(enf,worch)
        mydict = {'jav':jav, 'jaiv':jaiv, 'env':env, 'eniv':eniv}
        with open("./mydict", 'w') as f:
            pickle.dump(mydict, f)
