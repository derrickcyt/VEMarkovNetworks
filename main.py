# -*- coding: utf-8 -*-
import numpy as np
from feature_extract import *
from scipy import misc, optimize
from mn import VEMarkovNetworks


def is_utf8(text):
    try:
        text.decode("utf-8")
        return True
    except:
        return False


def main():
    from sklearn.model_selection import train_test_split

    # read text
    print "read corpus"
    corpus = []
    y = []
    count = 0
    with open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/mouse_1_rating_verbexpression.txt") as f:
        for line in f:
            corpus.append(line.strip().decode("utf-8"))
            y.append(-1)
            count += 1

    count = 0
    with open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/mouse_5_rating_verbexpression.txt") as f:
        for line in f:
            corpus.append(line.strip().decode("utf-8"))
            y.append(1)
            count += 1
    print "read corpus end.", len(corpus)

    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.4, random_state=0)

    y_set = [-1, 1]
    f = []
    f.extend(unigram_pos_f(y_set))
    f.extend(nn_vb_f(y_set))
    f.extend(nn_jj_f(y_set))
    f.extend(rb_vb_f(y_set))

    mn = VEMarkovNetworks(20, f)
    mn.fit(mn.transfer(X_train), y_train)

    y_pred = mn.predict(mn.transfer(X_test))

    y_test = np.array(y_test)

    print np.sum(y_pred == y_test, dtype=float) / len(y_test)


if __name__ == '__main__':
    main()
