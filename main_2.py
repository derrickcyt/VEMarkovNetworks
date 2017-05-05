# -*- coding: utf-8 -*-
import numpy as np
from feature import *
from scipy import misc, optimize
from mn_2 import VEMarkovNetworks
import nltk


def is_utf8(text):
    try:
        text.decode("utf-8")
        return True
    except:
        return False

from matplotlib import pylab
def plot_pr(auc_score, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.5, 1.0])
    pylab.xlim([0.05, 0.45])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.show()


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import codecs
    # read text
    print "read corpus"
    corpus = []
    y = []
    count = 0
    limit = 10000
    sent_len_threshold = 0

    with codecs.open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/mouse_1_rating_title_verbexpression.txt","r","utf-8") as f:
        for line in f:
            if len(nltk.tokenize.word_tokenize(line.strip())) >= sent_len_threshold and count < limit:
                corpus.append(line.strip())
                y.append(1)
                count += 1

    count = 0
    with codecs.open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/mouse_5_rating_title_verbexpression.txt","r","utf-8") as f:
        for line in f:
            if len(nltk.tokenize.word_tokenize(line.strip())) >= sent_len_threshold and count < limit:
                corpus.append(line.strip())
                y.append(0)
                count += 1
    print "read corpus end.", len(corpus)
    y_set = [0, 1]

    feature_creator = FeatureFunctionCreator(y_set)


    pos_tags = []
    for c in corpus:
        tokens = nltk.word_tokenize(c)
        pos_tag = nltk.pos_tag(tokens)
        pos_tags.append(pos_tag)
        feature_creator.handle_one_corpu(pos_tag)


    X_train, X_test, y_train, y_test = train_test_split(pos_tags, y, test_size=0.33, random_state=99)


    mn = VEMarkovNetworks(feature_creator.get_feature_num(), feature_creator, optimize='sgd')

    mn.fit(X_train, y_train)

    y_pred, y_probs = mn.predict(X_test)

    y_test = np.array(y_test)

    print np.sum(y_pred == y_test, dtype=float) / len(y_test)

    print classification_report(y_test, y_pred, target_names=["pos", "neg"], digits=4)

    max_probs = np.argsort(-y_probs[:,1])[:100]

    for i in max_probs:
        print '%.3f' % y_probs[i,1],
        print X_test[i]

    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs[:,1])
    plot_pr(0.5, precision, recall, "pos")




if __name__ == '__main__':
    main()
