
# coding: utf-8

# # 算数の文章を解く（LSTM翻訳の応用）

# ### import 

# In[1]:



import os, sys, time
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import json
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import training, datasets, iterators # 1.11
from chainer.training import extensions # 1.11
import chainer.functions as F
import chainer.links as L
from vocab import Vocab, Dict

try:
    cuda.check_cuda_available()
    import cupy
    xp = cupy
except:
    xp = np


# ### 文をバッチ化した２次元配列にする関数の定義

# In[2]:



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



# ### ネットワークを保存する関数の定義

# In[3]:


def savestate(rnn, parm):
    rnn.to_cpu()
    serializers.save_npz(outd+'/'+str(parm['name'])+str(parm['epoch']), rnn);
    #np.savez(outd+'/log', train_lossej=train_lossej,train_lossje=train_lossje, test_lossej=test_lossej,test_lossje=test_lossje)
    rnn.to_gpu()
    with open(outd+'/parm_of_'+str(parm['name']), 'w') as pf:
        json.dump(parm, pf)
    


# ### ネットワークの定義

# 
# [重要な変更点]
# * dec に１つ前の単語を入力するように変更しました。
# 
# [その他: 精度を上げるための変更]
# * LSTMを2層にしました。
# * 出力層の直前に、linear+RelU層を追加しました。
# * mid のLSTM を消しました。
# * embed の次元を200に減らしました。
# * dropout を embed, linear層にかけました。 
# * LSTM の forget_bias の初期値を 1 にしました。
#  + 長距離依存を捉えやすくするため。
# 
# 
# [その他: 速度を上げるための改良]
# * 出力層のノード数 = 単語の総数 にしました。
#  + 最後の softmax の計算コストの比重がかなり大きいらしいため、ジャストサイズに。
# * 学習しないときは Variable の引数で volatile = True にしました。
# * translate の際にbatch で入れるようにしました。
# 
# [その他]
# * 「次の単語」の予測がピッタリ正解した個数を返すようにしました。
#  + 学習の指標とするための「正解率」を計算するため。

# In[4]:


class EncDec(Chain):
    def __init__(self, nv_from, nv_to, nh, do_dropout=True):
        self.dim_embed = dim_embed = 200
        self.do_dropout = do_dropout
        super(EncDec, self).__init__(
            embed1=L.EmbedID(nv_from, dim_embed, ignore_label=-1),  # word embedding
            embed2=L.EmbedID(nv_to, dim_embed, ignore_label=-1),  # word embedding
            enc1=L.LSTM(dim_embed, nh, forget_bias_init=1), 
            enc2=L.LSTM(nh, nh, forget_bias_init=1), 
            dec1=L.LSTM(dim_embed, nh, forget_bias_init=1), 
            dec2=L.LSTM(nh, nh, forget_bias_init=1), 

            lin1= L.Linear(nh, nh*2),
            out=L.Linear(nh*2, nv_to),  # the feed-forward output layer
        )
        self.loss = None
        

    def trans(self, fromd, tod, translate=False, notLearn=False):
        if translate:
            notLearn=True
        self.bs = fromd.shape[1]
        self.enc1.reset_state()
        self.enc2.reset_state()
        self.dec1.reset_state()
        self.dec2.reset_state()
        self.zd = Variable(xp.zeros((self.bs, self.dim_embed), dtype=np.float32), volatile = notLearn)            
        self.loss = 0
        out = []
        
        for wd in reversed(fromd):
            x = Variable(xp.asarray(wd) , volatile = notLearn)
            x = self.embed1(x)
            if self.do_dropout : 
                x = F.dropout(x,train = not notLearn)
            x = self.enc1(x)
            x = self.enc2(x)
            
            
        self.dec1.set_state(self.enc1.c, self.enc1.h)
        self.dec2.set_state(self.enc2.c, self.enc2.h)
        cr = 0.0
        
        
        if not translate :
            cur_in = self.zd
            for wd in tod:
                t = Variable(xp.asarray(wd), volatile = notLearn)
                x = self.dec1(cur_in)
                x = self.dec2(x)
                x = F.relu(self.lin1(x))
                x = F.dropout(x,train = not notLearn)
                x  = self.out(x)
                self.loss += F.softmax_cross_entropy(x, t)
                cur_in = self.embed2(t)
                if self.do_dropout : 
                    cur_in = F.dropout(cur_in,train = not notLearn)

                ch = np.argmax(x.data.get(),axis=1)
                cr += np.sum(wd == ch)
                
        if translate :
            cur_in = self.zd
            fin = np.zeros(self.bs,dtype=np.bool)
            for i in xrange(100):
                t = Variable(xp.asarray(wd), volatile = True)
                x = self.dec1(cur_in)
                x = self.dec2(x)
                x = F.relu(self.lin1(x))
                x = F.dropout(x,train = False)
                x  = self.out(x)
                self.loss += F.softmax_cross_entropy(x, t)
                cur_in = self.embed2(t)
                if self.do_dropout : 
                    cur_in = F.dropout(cur_in,train = False)
                #ch = np.argmax(x.data.get(), axis=1)
	        ch = np.argmax(x.data, axis=1)
                fin += (ch==0)
                ch[fin] = 0
                out.append(ch)
                if (fin==True).all():
                    break
                cur_in = self.embed2( Variable(xp.asarray(ch,dtype=np.int32), volatile = notLearn) )

        return self.loss, np.asarray(out), cr


# ### データの読み込み

# In[5]:
import codecs
import types
import sys

#sys.stdout = codecs.getwriter('utf_8')(sys.stdout) 
"""
string = raw_input('>> ')
#print(string)
#type(string)
#string = u'ああああ'
f = codecs.open('data/test_dataI.txt', 'w', 'utf_8_sig')
f.write(string.decode('utf-8'))
f.close()
"""


file = codecs.open('data/dataI.txt', 'r', 'utf_8_sig')
string = file.read()
lines = string.splitlines()
file.close()

bound = 4

for i in range(len(lines)):
    lines[i] = lines[i].replace(u'は', '')
    if lines[i].find(u'足す') != -1:
        nums = lines[i].split(u'足す')
        nums[0] = ' '.join(nums[0].zfill(bound))
        nums[1] = ' '.join(nums[1].zfill(bound))
        lines[i] = nums[0] + u' 足す ' + nums[1] + u' は\n'
    
    if lines[i].find(u'引く') != -1:
        nums = lines[i].split(u'引く')
        nums[0] = ' '.join(nums[0].zfill(bound))
        nums[1] = ' '.join(nums[1].zfill(bound))
        lines[i] = nums[0] + u' 引く ' + nums[1] + u' は\n'
    
file = codecs.open('data/data.txt', 'w', 'utf-8-sig')
file.writelines(lines)
file.close()

file = codecs.open('data/suusikiI.txt', 'r', 'utf-8-sig')
string = file.read()
lines = string.splitlines()
file.close()

for i in range(len(lines)):
    lines[i] = ' '.join(lines[i].zfill(bound + 1))
    lines[i] += '\n'
    
file = codecs.open('data/suusiki.txt', 'w', 'utf-8-sig')
file.writelines(lines)
file.close()

file = codecs.open('data/test_suusikiI.txt', 'r', 'utf-8-sig')
string = file.read()
lines = string.splitlines()
file.close()

for i in range(len(lines)):
    lines[i] = ' '.join(lines[i].zfill(bound + 1))
    lines[i] += '\n'
    
file = codecs.open('data/test_suusiki.txt', 'w', 'utf-8-sig')
file.writelines(lines)
file.close()

file = codecs.open('data/test_dataI.txt', 'r', 'utf-8-sig')
string = file.read()
lines = string.splitlines()
file.close()

for i in range(len(lines)):
    lines[i] = lines[i].replace(u'は', '')
    if lines[i].find(u'足す') != -1:
        nums = lines[i].split(u'足す')
        nums[0] = ' '.join(nums[0].zfill(bound))
        nums[1] = ' '.join(nums[1].zfill(bound))
        lines[i] = nums[0] + u' 足す ' + nums[1] + u' は\n'
    
    if lines[i].find(u'引く') != -1:
        nums = lines[i].split(u'引く')
        nums[0] = ' '.join(nums[0].zfill(bound))
        nums[1] = ' '.join(nums[1].zfill(bound))
        lines[i] = nums[0] + u' 引く ' + nums[1] + u' は\n'
    
file = codecs.open('data/test_data.txt', 'w', 'utf-8-sig')
file.writelines(lines)
file.close()


outd = "result"
jatrain = "suusiki.txt"
entrain = "data.txt"
worch = 1 # 0:char mode, 1: word mode
nv_en = 6682
nv_ja = 8844

nh = 800

dicttype = ['mychar', 'mydict'][worch]
dic = Dict(dicttype)

trainj = Vocab(jatrain, dic.jav, worch) #"train10000.ja", 1)
traine = Vocab(entrain, dic.env, worch) #"train10000.en", 1)
testj = Vocab("test_suusiki.txt", dic.jav, worch)
teste = Vocab("test_data.txt", dic.env, worch)


t = np.asarray( [x for s in traine.dataset for x in s] )
len({x for x in t})

# N = len(trainj.dataset) # # of case
N = len(traine.dataset) # same length
N_test = len(teste.dataset)
train_lossej = []
train_lossje = []
test_lossej = []
test_lossje = []

n_epoch = 50  # 20000は長すぎ1日で終わらない
batchsize = 100

parm = { 'outd':outd, 'nv_en': nv_en, 'nv_ja': nv_ja,'nh':nh, 'jatrain':jatrain, 'entrain':entrain,
         'worch':worch, 'dicttype':dicttype,
         'batch':batchsize, 'nepoch':n_epoch,
         'modelver':u'20161008'
    }
#print parm



# #### 単語の頻度が低いものはうまく学習できないおそれがあるので、訓練のサンプル数は多い方がいい → 50,000 のものを用いる

# In[6]:


t = np.asarray( [x for s in trainj.dataset for x in s] + [x for s in testj.dataset for x in s]  )
c = {x:0 for x in t}
#print "日本語の総単語数", len(c)
for x in t:
    c[x] = c[x]+1 
#print "日本語の総単語数(頻度が1)", np.sum(np.asarray(c.values())==1)
#for x in np.asarray(c.keys())[np.asarray(c.values())==1 ][:100]:
    #print dic.id2j(x),


# ### 学習を行う：
# 
# [重要な変更点] 
# * 50イテレーションを上限にしました。
#  + 過学習を起こすため。後述の結果を見てわかるように、テストデータにおいて、lossと正解率(「学習ループ」の出力結果), BLEU値(「BLEU値推移」の出力結果) など指標もからみて、約 20 epoc ぐらいでサチっているように見えます。

# #### 学習ループ

# * 出力結果の最後の２つの列は、それぞれ、訓練データの正解率と、テストデータの正解率です。
#  + 訓練データの正解率は90%を超えて行っていますが、テストデータの正解率は68%あたりでサチっています。

# In[ ]:


'''
nv_en = 6682
nv_ja = 8844
nh = 800

rnn = EncDec(nv_en, nv_ja, nh, do_dropout=True) #RNN() vocab and hidden
opt = optimizers.Adam()
opt.setup(rnn)
cuda.get_device(0).use()
rnn.to_gpu(0)

batchsize = 100
n_epoch = 1000

# Learning loop
for epoch in xrange(1, n_epoch+1):
    stt = time.clock()
    print 'epoch', epoch
    perm = np.random.permutation(N)
    data_from = np.asarray(traine.dataset)[perm]
    data_to   = np.asarray(trainj.dataset)[perm]
    test_from = np.asarray(teste.dataset)
    test_to   = np.asarray(testj.dataset)

    sum_loss = 0
    sum_cr = 0
    sum_all =0
    
    #  J to E
    for i in xrange(0, N, batchsize):
        _from = makebatch(data_from, i, batchsize)
        _to = makebatch(data_to, i, batchsize)
        loss,x,cr = rnn.trans(_from, _to)
        sum_cr += cr
        sum_all += np.sum(_to >= 0)
        rnn.zerograds()
        loss.backward()
        #opt.clip_grads(1.0)
        opt.update()
        sum_loss += loss.data * batchsize
    
    #  validation

    sumtest_loss = 0
    sumtest_cr = 0
    sumtest_all = 0


    for i in xrange(0, N_test, batchsize): 
        _from = makebatch(test_from, i, batchsize)
        _to = makebatch(test_to, i, batchsize)
        loss,x, cr = rnn.trans(_from, _to, notLearn=True)
        sumtest_all += np.sum(_to >= 0)
        sumtest_cr += cr 

        sumtest_loss += loss.data * batchsize


    ett = time.clock();

    print 'train loss={}, test loss={}, time={}'.format(
        sum_loss /N, sumtest_loss /N_test, ett-stt),
    print sum_cr/sum_all, sumtest_cr/sumtest_all
    if (epoch % 250) == 0:
        parm['epoch'] = epoch
        parm['name'] = 'rnnej_4_'
        savestate(rnn, parm)
'''

# import nltk.translate.bleu_score as bleu
# def get_total_bleu(rnn, data_from, data_to):
#     order = np.argsort([len(s) for s in data_from])
#     data_from = data_from[order]
#     data_to = data_to[order]
#     bs = 100
#     bl = []
#     for i in xrange(0,len(data_from),bs):
#         _from = np.asarray(makebatch(data_from, i, bs))
#         _to = np.asarray(makebatch(data_to, i, bs))
#         loss, out, cr = rnn.trans(_from, None, translate=True)
#         for t, o in zip(_to.T, out.T):
#             try:
#                 bl.append(bleu.sentence_bleu( [ t[t>0] ], o[o>0] ) ) 
#             except:
#                 bl.append(0.0)
#     return bl

# ### テストデータに対する計算(翻訳)結果を見てみる
# 

# #### 学習結果の読み込み

# In[8]:


rnn = EncDec(nv_en, nv_ja, nh) #RNN() vocab and hidden
#rnn.to_gpu(0)
serializers.load_npz(outd+'/rnnej_4_580', rnn);


# #### 表示

# In[10]:


bs=1
for i in xrange(0,10,2):
    _from = np.asarray(makebatch(teste.dataset, i, 'C'))
    _to = np.asarray(makebatch(testj.dataset, i, 'C'))
    #print jd.shape
    loss, out, cr = rnn.trans(_from, None, translate=True)
    for j in xrange(bs):
        f, t, o = _from[:,j], _to[:,j], out[:,j]
        f, t, o = f[f>0], t[t>0], o[o>0]
        #bl = bleu.sentence_bleu([ t ], o )
        print "------------"
        print u"問題: ",
	string = "".join([dic.id2e(w) for w in f])

	flag = 0
	for b in xrange(len(string)):
	    
	    #print string[b]

	    
	    if flag == 0 and string[b] != '0' and string[b] != '-':
	        flag = 1
	    if flag == 1 or string[b] == '-':
		sys.stdout.write(string[b])
	    if b == 10:
		flag = 0
	
	    #print flag,
	    
	print "\n"
        #print "".join([dic.id2e(w) for w in f]) 
        #result = "".join([dic.id2j(w) for w in t])
        #print "正解:",result
        string = "".join([dic.id2j(w) for w in o])
	print u"答え: ",

	flag = 0
	for b in xrange(len(string)):
	    
	    #print string[b]

	    
	    if flag == 0 and string[b] != '0' and string[b] != '-':
	        flag = 1
	    if flag == 1 or string[b] == '-':
		sys.stdout.write(string[b])
	    if b == 10:
		flag = 0
	#print u"\n正解:","".join([dic.id2j(w) for w in t])
        
	print "\n------------"
	break
        #if result == answer:
        #    print "●正解\n"
        #else:
        #    print "不正解\n"
        print "正解:","".join([dic.id2j(w) for w in t])
        #print "翻訳:","".join([dic.id2j(w) for w in o])
        #print " BLEU:%.2f"% bl
    break
        


# # GPU,CPUの使用状況

# In[ ]:


#get_ipython().system(u'nvidia-smi')

