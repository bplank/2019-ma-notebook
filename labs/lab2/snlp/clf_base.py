from snlp.constants import OFFSET
import numpy as np

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    feat_dict = {(label, OFFSET) : 1}
    for (x_i, count) in base_features.items():
        feat_dict[(label, x_i)] = count
    return feat_dict


def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''

    scores = {}
    base_features[OFFSET] = 1.0
    for y in labels:
        # get active features for a class
        weights_feats = [ weights[(label,f)] for (label, f) in weights.keys() if label == y and f in base_features]
        active_feats = [ base_features[feat] for feat in base_features if (y, feat) in weights ]
        if not active_feats:
            scores[y] = 0.0
        else:
            scores[y] = np.dot(active_feats, weights_feats)
    return argmax(scores),scores


def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat
