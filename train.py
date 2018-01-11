# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import training, datasets, iterators # 1.11
from chainer.training import extensions # 1.11
import chainer.functions as F
import chainer.links as L
from vocab import Vocab, Dict


class EncDec(Chain):
    def __init__(self, nv, nh):
        self.nv = nv
        self.nh = nh
        super(EncDec, self).__init__(
            embed=L.EmbedID(nv, nh, ignore_label=-1),  # word embedding
            enc=L.LSTM(nh, nh),  
            mid=L.LSTM(nh, nh), 
            dec=L.LSTM(nh, nh), 
            out=L.Linear(nh, nv),  # the feed-forward output layer
        )
        

    def trans(self, fromd, tod, train=True):
        self.bs = fromd.shape[1]
        self.enc.reset_state()
        self.mid.reset_state()
        self.dec.reset_state()
        self.zd = Variable(cuda.zeros((self.bs,self.nh), dtype=np.float32))
        loss = 0
        out = []
        for wd in reversed(fromd):
            #print wd
            x = Variable(cuda.to_gpu(np.reshape(wd,(self.bs,1)))) 
            #print x.data
            y = self.enc(self.embed(x))
        self.mid.set_state(self.enc.c, self.enc.h)
        y = self.mid(self.zd)
        self.dec.set_state(self.mid.c, self.mid.h)
        if train :
            for wd in tod:
                t = Variable(cuda.to_gpu(np.reshape(wd,(self.bs)))) # t.data)
                x  = self.out(self.dec(self.zd))
                loss += F.softmax_cross_entropy(x, t)
        else :
            for i in xrange(100):
                x  = self.out(self.dec(self.zd))
                ch = np.argmax(x.data[0]) # bs should be 1
                out.append(ch)
                if ch == 0 :
                    break

        return loss, out

def makebatch(d, pos, batchsize):
    rest = len(d) - pos

    if rest < batchsize:
        batchsize = rest
    maxlen = 0
    for i in xrange(pos,pos+batchsize):
        maxlen = max(maxlen,len(d[i]))
    #print "rest ", rest, "lend", len(d), "bs=", batchsize, "maxlen=", maxlen
    jd = np.full((maxlen, batchsize), -1, dtype=np.int32)
    #jd = np.full((batchsize, maxlen), -1, dtype=np.int32)
    #print "jd shape", jd.shape
    for i in range(batchsize):
        #print "pos+i", pos+i, len(d[pos+i])
        for j in xrange(len(d[pos+i])):
            jd[j][i] = d[pos+i][j]
            #jd[i][j] = d[pos+i][j]
    #print jd.shape
    #print jd
    return jd
# jd shape (15, 11)  batch=11 maxlen=15
# [[   8   18    8  442   18    8 1589   24    8  119  440]
#  [   9   62  116    9    9    9  320    9    9    9   25]
#  [1080    9    9 2917  308  722    2  218   18  120   63]
#  [  82  954 1868    2   28   28   67  908   25  121   25]
#  [ 243    4  477 3412   20  512   11   51  496    6  143]
#  [  49  955    4   41   25   52   12   35  201  122   28]
#  [  15   14   99   30    2  726   13  415 1380    2  128]
#  [   0   60   25   35  284   49   14    2 1481  123   41]
#  [  -1   35  101   42  291   15   15   46   28   41   49]
#  [  -1   42   28   15   35    0    0   35  656   49   25]
#  [  -1   15  102    0   15   -1   -1   15   14   15    7]
#  [  -1    0   41   -1    0   -1   -1    0   21    0   15]
#  [  -1   -1   49   -1   -1   -1   -1   -1   15   -1    0]
#  [  -1   -1   15   -1   -1   -1   -1   -1    0   -1   -1]
#  [  -1   -1    0   -1   -1   -1   -1   -1   -1   -1   -1]]

def savestate(parm):
    rnnje.to_cpu()
    rnnej.to_cpu()
    serializers.save_npz(outd+'/rnnej'+str(parm['worch']), rnnej);
    serializers.save_npz(outd+'/rnnje'+str(parm['worch']), rnnje);
    np.savez(outd+'/log', train_lossej=train_lossej,train_lossje=train_lossje, test_lossej=test_lossej,test_lossje=test_lossje)
    rnnje.to_gpu()
    rnnej.to_gpu()
    with open(outd+'/parm', 'w') as pf:
        json.dump(parm, pf)
    

if __name__ == "__main__":
    if len(sys.argv) != 8:
        outd = "result_train"
        jatrain = "train10000.ja"
        entrain = "train10000.en"
        n_epoch = 5000  # 20000は長すぎ1日で終わらない
        worch = 1 # 0:char mode, 1: word mode
        nv =5000
        nh = 400
    else :
        outd = sys.argv[1]
        jatrain = sys.argv[2]
        entrain = sys.argv[3]
        n_epoch = int(sys.argv[4])
        worch = int(sys.argv[5])
        nv = int(sys.argv[6])
        nh = int(sys.argv[7])
    #print outd, jatrain, entrain, len(sys.argv)

    try:
        os.mkdir(outd)
    except OSError:
        print outd, 'already exists'
    if worch == 1:
        dicttype = 'mydict'
    else:
        dicttype = 'mychar'
    dic = Dict(dicttype)
    trainj = Vocab(jatrain, dic.jav, worch) #"train10000.ja", 1)
    traine = Vocab(entrain, dic.env, worch) #"train10000.en", 1)
    testj = Vocab("test1000.ja", dic.jav, worch)
    teste = Vocab("test1000.en", dic.env, worch)

    rnnej = EncDec(nv, nh) #RNN() vocab and hidden
    rnnje = EncDec(nv, nh) #RNN() vocab and hidden
    #model = L.Classifier(rnn)
    optej = optimizers.Adam()
    optje = optimizers.Adam()
    optej.setup(rnnej)
    optje.setup(rnnje)
    rnnej.to_gpu(0)
    rnnje.to_gpu(0)

    batchsize = 100
    # N = len(trainj.dataset) # # of case
    N = len(traine.dataset) # same length
    N_test = len(teste.dataset)
    train_lossej = []
    train_lossje = []
    test_lossej = []
    test_lossje = []

    parm = { 'outd':outd, 'nv': nv, 'nh':nh, 'jatrain':jatrain, 'entrain':entrain,
             'worch':worch, 'dicttype':dicttype,
             'batch':batchsize, 'nepoch':n_epoch,
             'modelver':u'20161008'
        }
    print parm
# Learning loop
    for epoch in xrange(1, n_epoch+1):
        stt = time.clock()
        print 'epoch', epoch
        perm = np.random.permutation(N)
        ptrainj = np.array(trainj.dataset)[perm]
        ptraine = np.array(traine.dataset)[perm]

        sum_lossej = 0
        sum_lossje = 0
        for i in xrange(0, N, batchsize):
            jd = np.asarray(makebatch(ptrainj, i, batchsize))
            ed = np.asarray(makebatch(ptraine, i, batchsize))
            #print "jd", jd.shape, jd
            #print "ed", ed.shape, ed
            # jd (41, batchsize) ed (53, batchsize) 
            #t = Variable(ed.T)

            #  J to E
            lossje,x = rnnje.trans(jd, ed)
            rnnje.zerograds()
            lossje.backward()
            optje.update()

            #  E to J
            lossej,x = rnnej.trans(ed, jd)
            rnnej.zerograds()
            lossej.backward()
            optej.update()

            train_lossej.append(lossej.data)
            train_lossje.append(lossje.data)
            sum_lossej += lossej.data * batchsize
            sum_lossje += lossje.data * batchsize

        #  validation

        sumtest_lossej = 0
        sumtest_lossje = 0
        for i in xrange(0, N_test, batchsize): 
            jd = np.asarray(makebatch(testj.dataset, i, batchsize))
            ed = np.asarray(makebatch(teste.dataset, i, batchsize))
            #t = Variable(ed.T)
            #print "b=", batchsize,jd
            #  J to E
            lossje,x = rnnje.trans(jd, ed)
            #  E to J
            lossej,x = rnnej.trans(ed, jd)

            test_lossej.append(lossej.data)
            test_lossje.append(lossje.data)
            sumtest_lossej += lossej.data * batchsize
            sumtest_lossje += lossje.data * batchsize


        ett = time.clock();

        print 'train lossje={},lossej={}, test lossje={}, lossej={}, time={}'.format(sum_lossje /N, sum_lossej/N,sumtest_lossje /N_test, sumtest_lossej/N_test, ett-stt)
        if (epoch % 100) == 0:
            parm['epoch'] = epoch
            savestate(parm)

    savestate(parm)



