# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import codecs
import pickle

class Vocab():
    def __init__(self, fname, voc, mode=0):
        dataset = []
        unko = codecs.open("./data/%s" % (fname),"r","utf_8_sig")
        for line in unko:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if mode == 1: # word mode
                line = line.split()
            data = [] #np.empty((len(line) + 1,), dtype=np.int32)
            for i in xrange(len(line)):
                word = line[i]
                data.append(voc[word])
            data.append(0) # last
            dataset.append(data)
        dataset = np.asarray(dataset)
        data_len = [len(x) for x in dataset ]
        self.dataset = dataset #[np.argsort(data_len)]
        self.maxlen = max(data_len)
        n_vocab = len(voc)
        n_dataset = len(self.dataset)
        #print "file=\"{0}\"  mode={1} Nchars={2} Ndata={3} Maxlen={4}".format(fname, mode, n_vocab, n_dataset, self.maxlen),
        #print "MeanLen=", np.mean(data_len)


class Dict():
    def __init__(self, ty):
        with open('./{0}'.format(ty), 'r') as f:
            mydict = pickle.load(f)
        self.jav = mydict['jav']
        self.jaiv = mydict['jaiv']
        self.env = mydict['env']
        self.eniv = mydict['eniv']
    def id2j(self, id):
        if id < len(self.jaiv):
            return self.jaiv[id]
        else:
            return "???"

    def j2id(self, word):
        return self.jav[word]

    def id2e(self, id):
        if id < len(self.eniv):
            return self.eniv[id]
        else:
            return "???"

    def e2id(self, word):
        return self.env[word]


if __name__ == "__main__":
    dic = Dict()
    if len(sys.argv) > 1 :
        fn = sys.argv[1]
    else:
        fn = "train50000.ja"
    worch = 1
    ev = Vocab(fn,dic.jav, worch)
    #ev = Vocab("train1000.en",1)
    for i in xrange(len(ev.dataset)):
        for j in xrange(len(ev.dataset[i])):
            w = dic.id2j(ev.dataset[i][j])
            #if w =='<eos>' :
                #print len(ev.dataset[i]),""
            #else :
                #print w,
