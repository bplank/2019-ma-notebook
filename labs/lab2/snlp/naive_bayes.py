from snlp.constants import OFFSET
from snlp import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter

# deliverable 3.1
def get_corpus_counts(x,y,label):
    # type: (object, object, object) -> object
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    ccounts = defaultdict(int)
    raise NotImplementedError

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    # computes P(x|y=label) for a specific label
    V = len(vocab)
    logprobs = defaultdict(float)
    
    raise NotImplementedError


# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    raise NotImplementedError




