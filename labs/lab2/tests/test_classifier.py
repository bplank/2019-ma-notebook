from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from snlp import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, logreg
import numpy as np

def setup_module():

    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned, x_bl_pruned
    global labels
    global vocab

    y_tr,x_tr = preproc.read_data('data/rock-lyrics-train.csv',preprocessor=preproc.bag_of_words)
    labels = set(y_tr)

    counts_tr = preproc.aggregate_counts(x_tr)

    y_dv,x_dv = preproc.read_data('data/rock-lyrics-dev.csv',preprocessor=preproc.bag_of_words)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_tr, 10)
    x_dv_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dv, 10)


def test_d2_1_featvec():
    label = '1980s'
    fv = clf_base.make_feature_vector({'test':1,'case':2},label)
    eq_(len(fv),3)
    eq_(fv[(label,'test')],1)
    eq_(fv[(label,'case')],2)
    eq_(fv[(label,constants.OFFSET)],1)


def test_d2_2_predict():
    global x_tr_pruned, x_dv_pruned, y_dv
    y_hat,scores = clf_base.predict(x_tr_pruned[1],hand_weights.theta_hand,labels)
    eq_(scores['pre-1980s'],0.0)
    eq_(scores['2000s'], 0.2)
    eq_(y_hat,'2000s')
    eq_(scores['1980s'],0.0)


def test_d3_1_corpus_counts():
    # public
    iama_counts = naive_bayes.get_corpus_counts(x_tr_pruned,y_tr,"1980s");
    eq_(iama_counts['today'],24)
    eq_(iama_counts['yesterday'],8)
    eq_(iama_counts['mez'],0)


def test_d3_2_pxy():
    global vocab, x_tr_pruned, y_tr
    
    # check that distribution normalizes to one
    log_pxy = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"1980s",0.1,vocab)
    assert_almost_equals(np.exp(list(log_pxy.values())).sum(),1)

    # check that values are correct
    assert_almost_equals(log_pxy['money'],-7.969973834156985,places=3)
    assert_almost_equals(log_pxy['fly'],-8.360722957632635,places=3)

    log_pxy_more_smooth = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"1980s",10,vocab)
    assert_almost_equals(log_pxy_more_smooth['money'],-8.173461476943206,places=3)
    assert_almost_equals(log_pxy_more_smooth['tonight'], -7.287410630258769,places=3)


def test_d3_3a_nb():
    global x_tr_pruned, y_tr

    theta_nb = naive_bayes.estimate_nb(x_tr_pruned,y_tr,0.1)

    y_hat,scores = clf_base.predict(x_tr_pruned[55],theta_nb,labels)
    eq_(y_hat,'pre-1980s')

    y_hat,scores = clf_base.predict(x_tr_pruned[155],theta_nb,labels)
    eq_(y_hat,'2000s')




