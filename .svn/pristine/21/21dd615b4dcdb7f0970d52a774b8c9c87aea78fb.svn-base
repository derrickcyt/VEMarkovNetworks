# -*- coding: utf-8 -*-
import nltk

pos_list = ["VB", "JJ", "NN", "RB"]
pos_NN = ["NN", "NNS", "NNP", "NNPS", "PRP", "WP"]

negators = []

with open("negators.txt") as f:
    for line in f:
        if line.strip():
            negators.append(line.strip())


# feature functions

def unigram_pos_f(y_set):
    f = lambda y_i, pos: lambda X, y: 1 if y == y_i and pos in X.pos_tag else 0
    return [f(y_i, pos) for y_i in y_set for pos in pos_list]


def nn_vb_f(y_set):
    def has_nn(X):
        return True in [pos in pos_NN for pos in X.pos_tag]

    def has_negator(X):
        return True in [X.pos_tag[i] in pos_NN and X.word_list[i] in negators for i in xrange(len(X.word_list))]

    common = lambda X, y: has_nn(X) and "VB" in X.pos_tag
    f1 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) else 0
    f2 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) and has_negator(X) else 0
    return [f1(y_i) for y_i in y_set] + [f2(y_i) for y_i in y_set]


def nn_jj_f(y_set):
    def has_nn(X):
        return True in [pos in pos_NN for pos in X.pos_tag]

    def has_negator(X):
        return True in [X.pos_tag[i] in pos_NN and X.word_list[i] in negators for i in xrange(len(X.word_list))]

    common = lambda X, y: has_nn(X) and "JJ" in X.pos_tag
    f1 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) else 0
    f2 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) and has_negator(X) else 0

    return [f1(y_i) for y_i in y_set] + [f2(y_i) for y_i in y_set]


def rb_vb_f(y_set):
    def has_negator(X):
        return True in [X.pos_tag[i] == "RB" and X.word_list[i] in negators for i in xrange(len(X.word_list))]

    common = lambda X, y: "RB" in X.pos_tag and "VB" in X.pos_tag
    f1 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) else 0
    f2 = lambda y_i: lambda X, y: 1 if y == y_i and common(X, y) and has_negator(X) else 0

    return [f1(y_i) for y_i in y_set] + [f2(y_i) for y_i in y_set]


class Feature(object):
    def __init__(self, sentence):
        self.word_list = [w.lower() for w in nltk.word_tokenize(sentence)]
        self.pos_tag = [pos for word, pos in nltk.pos_tag(self.word_list)]
        self.negators = self.find_negators()

    def find_negators(self):
        nr = []
        for i in xrange(len(self.word_list)):
            if self.word_list[i] in negators:
                nr.append(i)
        return nr


if __name__ == "__main__":
    text = "And now for something not completely different"
    feature = Feature(text)
    print feature.word_list
    print feature.pos_tag
    y_set = [-1, 1]
    f = []
    f.extend(unigram_pos_f(y_set))
    f.extend(nn_vb_f(y_set))
    f.extend(nn_jj_f(y_set))
    f.extend(rb_vb_f(y_set))
    print [f_i(feature, 1) for f_i in f]
