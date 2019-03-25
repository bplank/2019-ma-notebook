__author__ = "bplank"
"""
Exercise: sentiment classification with logistic regression in Sklearn

1) Examine the code/data.
   What is the distribution of labels in the data (how many positive/negative)?
   How is a text represented? (Check what the CountVectorizer does)
2) Add code to train and evaluate the classifier. What accuracy do you get? 
3) Implement a simple baseline to compare your system to. Add classification report output.
4) (optional): Add code to output and inspect the predicted instances. 
"""
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import random


def load_sentiment_sentences_and_labels():
    """
    loads the movie review data
    """
    positive_sentences = [l.strip() for l in open("data/rt-polaritydata/rt-polarity.pos").readlines()]
    negative_sentences = [l.strip() for l in open("data/rt-polaritydata/rt-polarity.neg").readlines()]

    positive_labels = [1 for sentence in positive_sentences]
    negative_labels = [0 for sentence in negative_sentences]

    sentences = np.concatenate([positive_sentences,negative_sentences], axis=0)
    labels = np.concatenate([positive_labels,negative_labels],axis=0)

    ## make sure we have a label for every data instance
    assert(len(sentences)==len(labels))
    data = list(zip(sentences,labels))

    # make sure we shuffle the data, for now seeded
    random.seed(113) #seed
    random.shuffle(data)

    return data


def split_and_vectorize(data):
    """
    split data into train (60%), dev (20%) and test (20%)
    and transform it into vector (featurize/vectorize)
    """
    # split data
    train_end = int(0.6 * len(data))
    dev_end = int(0.8 * len(data))

    sentences = [sentence for sentence, label in data]
    labels = [label for sentence, label in data]
    X_train_raw, X_dev_raw, X_test_raw = sentences[:train_end], sentences[train_end:dev_end], sentences[dev_end:]
    y_train, y_dev, y_test = labels[:train_end], labels[train_end:dev_end], labels[dev_end:]

    print("vectorize data..")
    vectorizer = CountVectorizer() # Q1: what does the CountVectorizer do?

    X_train = vectorizer.fit_transform(X_train_raw) # make sure you understand the difference between fit_transform and transform
    X_dev = vectorizer.transform(X_dev_raw)
    X_test = vectorizer.transform(X_test_raw)

    assert (X_train.shape[0] == len(y_train))
    assert (X_dev.shape[0] == len(y_dev))
    assert (X_test.shape[0] == len(y_test))

    print("#train instances: {} #dev: {} #test: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

    return X_train, y_train, X_dev, y_dev, X_test, y_test


## read input data
print("load data..")
data = load_sentiment_sentences_and_labels()
X_train, y_train, X_dev, y_dev, X_test, y_test = split_and_vectorize(data)


### Q2: Train and evaluate the classifier on the dev set -- your code here
clf = None # init the classifier

print("train model..")
print(clf) # shows which model and its parameters


y_predicted_dev = None


exit() # remove this line
## end your code here for Q2

## Q3: Add a simple baseline -- your code here

baseline_dev = None

## end your code here for Q3



### evaluate
accuracy_dev = accuracy_score(y_dev, y_predicted_dev)

print("===== dev set ====")
print("Baseline:   {0:.2f}".format(accuracy_score(y_dev, baseline_dev)*100))
print("Classifier: {0:.2f}".format(accuracy_dev*100))
