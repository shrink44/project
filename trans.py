# -*- coding: utf-8 -*-
import os, sys, time, json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
#from chainer import training, datasets, iterators # 1.11
#from chainer.training import extensions # 1.11
import chainer.functions as F
import chainer.links as L
from vocab import Vocab, Dict
from train import EncDec, makebatch

if __name__ == "__main__":
    if len(sys.argv) > 1:
        outd = sys.argv[1]
    else :
        outd='result_trainw4'
    print "use \"{0}\"".format(outd)
    nv = 10000
    nh = 400
    try:
        pf = open(outd+"/parm", 'r')
        parm = json.load(pf)
        worch = parm['worch']
        nv = parm['nv']
        nh = parm['nh']
        dicttype = parm['dicttype']
    except:
	print "no parm file"
        worch = 1 # char mode 1:word mode
        if worch == 1:
            dicttype = 'mydict'
        else:
            dicttype = 'mychar'
    print parm
    dic = Dict(dicttype)
    dictj = Vocab("train1000.ja", dic.jav, worch)
    dicte = Vocab("train1000.en", dic.env, worch)
    #dictj = Vocab("test100.ja", dic.jav, worch)
    #dicte = Vocab("test100.en", dic.env, worch)
    #dictj = Vocab("test10.ja", dic.jav, worch)
    #dicte = Vocab("test10.en", dic.env, worch)

    rnnej = EncDec(nv,nh)#100) #RNN()
    rnnje = EncDec(nv,nh)#100) #RNN()

    serializers.load_npz(outd+'/rnnej'+str(worch), rnnej);
    serializers.load_npz(outd+'/rnnje'+str(worch), rnnje);

    rnnej.to_gpu(0)
    rnnje.to_gpu(0)
    N = len(dictj.dataset) # # of case

    # Learning loop
    batchsize = 1
    stt = time.clock()
    for i in xrange(10):#N):
        jd = np.asarray(makebatch(dictj.dataset, i, batchsize))
        ed = np.asarray(makebatch(dicte.dataset, i, batchsize))
        #  J to E
        lossje,jeout = rnnje.trans(jd, ed, train=False)
        #  E to J
        lossej,ejout = rnnej.trans(ed, jd, train=False)

        if False:
            print jd.shape, ed.shape
            print jd
            print jeout
            print ed
            print ejout
        for j in jd: # jap correct
            print dic.id2j(int(j)),
        print "\n-> ",
        for j in jeout: # japa translated
            print dic.id2e(int(j)),
        print ""

        for j in ed: # english correst
            print dic.id2e(int(j)),
        print "\n-> ",
        for j in ejout: # jap -> eng translated
            print dic.id2j(int(j)),
        print "\n"


    



