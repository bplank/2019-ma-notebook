from nose.tools import eq_

from snlp import preproc

import nose

def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, counts_dv, counts_bl, x_dv_pruned, x_tr_pruned
    global vocab
    y_tr,x_tr = preproc.read_data('data/rock-lyrics-train.csv',preprocessor=preproc.bag_of_words)
    y_dv,x_dv = preproc.read_data('data/rock-lyrics-dev.csv',preprocessor=preproc.bag_of_words)


def test_d1_1_bow():
    global x_tr, y_tr
    eq_(len(x_tr), len(y_tr))
    eq_(x_tr[4]['all'],2)
    eq_(x_tr[48]['heart'],10)
    eq_(x_tr[410]['angels'],0)
    eq_(len(x_tr[1144]),79)

def test_d1_2_agg():
    global x_dv
    counts_tr = preproc.aggregate_counts(x_tr)
    counts_dv = preproc.aggregate_counts(x_dv)

    eq_(counts_dv['you'],3456)
    eq_(len(counts_dv),11547)
    eq_(counts_dv['money'],48)

def test_d1_3a_oov():
    counts_tr = preproc.aggregate_counts(x_tr)
    counts_dv = preproc.aggregate_counts(x_dv)

    eq_(len(preproc.compute_oov(counts_dv,counts_tr)),3145)
    eq_(len(preproc.compute_oov(counts_tr,counts_dv)),33146)


def test_d1_4_prune():
    global x_dv, counts_tr
    counts_tr = preproc.aggregate_counts(x_tr)
    counts_dv = preproc.aggregate_counts(x_dv)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr,x_tr,3)
    x_dv_pruned, vocab2 = preproc.prune_vocabulary(counts_tr,x_dv,3)

    eq_(len(vocab),len(vocab2))
    eq_(len(vocab),15105)

    eq_(len(x_dv[95].keys())-len(x_dv_pruned[95].keys()),14)
    

