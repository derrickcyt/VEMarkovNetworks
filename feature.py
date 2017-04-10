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
class FeatureFunctionCreator:
    def __init__(self, y_set):
        self.y_set = y_set
        self.unigram_pos_feature = []
        self.nn_vb_feature = []
        self.nn_vb_neg_feature = []
        self.nn_jj_feature = []
        self.nn_jj_neg_feature = []
        self.rb_vb_feature = []
        self.rb_vb_neg_feature = []


    def handle_one_corpu(self, pos_tag):
        self._generate_unigram_pos(pos_tag)
        self._generate_nn_vb(pos_tag)
        self._generate_nn_vb_neg(pos_tag)
        self._generate_nn_jj(pos_tag)
        self._generate_nn_jj_neg(pos_tag)
        self._generate_rb_vb(pos_tag)
        self._generate_rb_vb_neg(pos_tag)

    def get_feature_num(self):
        num = len(self.unigram_pos_feature)+len(self.nn_vb_feature) \
        + len(self.nn_vb_neg_feature) \
        + len(self.nn_jj_feature) \
        + len(self.nn_jj_neg_feature) \
        + len(self.rb_vb_feature) \
        + len(self.rb_vb_neg_feature)

        return num * 2


    def get_all_feature(self, pos_tag, y):
        feature = []

        # unigram
        for word, pos in self.unigram_pos_feature:
            is_added = False
            for w, p in pos_tag:
                if w == word and p == pos:
                    feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
                    is_added = True
                    break
            if not is_added:
                feature.extend([0] * len(self.y_set))

        # nn_vb
        for nn, vb in self.nn_vb_feature:
            has_nn = False
            has_vb = False
            for w, p in pos_tag:
                if w == nn and p in pos_NN:
                    has_nn = True
                if w == vb and 'VB' in p:
                    has_vb = True

            if has_nn and has_vb:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        for nn, vb in self.nn_vb_neg_feature:
            has_nn = False
            has_vb = False
            for w, p in pos_tag:
                if w == nn and p in pos_NN:
                    has_nn = True
                if w == vb and 'VB' in p:
                    has_vb = True

            if has_nn and has_vb:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        # nn_jj
        for nn, jj in self.nn_jj_feature:
            has_nn = False
            has_jj = False
            for w, p in pos_tag:
                if w == nn and p in pos_NN:
                    has_nn = True
                if w == jj and 'JJ' in p:
                    has_jj = True

            if has_jj and has_nn:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        for nn, jj in self.nn_jj_neg_feature:
            has_nn = False
            has_jj = False
            for w, p in pos_tag:
                if w == nn and p in pos_NN:
                    has_nn = True
                if w == jj and 'JJ' in p:
                    has_jj = True

            if has_jj and has_nn:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        # rb_vb
        for rb, vb in self.rb_vb_feature:
            has_rb = False
            has_vb = False
            for w, p in pos_tag:
                if w == rb and 'RB' in p:
                    has_rb = True
                if w == vb and 'VB' in p:
                    has_vb = True

            if has_rb and has_vb:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        for rb, vb in self.rb_vb_neg_feature:
            has_rb = False
            has_vb = False
            for w, p in pos_tag:
                if w == rb and 'RB' in p:
                    has_rb = True
                if w == vb and 'VB' in p:
                    has_vb = True

            if has_rb and has_vb:
                feature.extend([1 if y == y_i else 0 for y_i in self.y_set])
            else:
                feature.extend([0] * len(self.y_set))

        return feature


    def _generate_unigram_pos(self, pos_tag):
        for word, pos in pos_tag:
            if (word, pos) not in self.unigram_pos_feature:
                self.unigram_pos_feature.append((word, pos))


    def _generate_nn_vb(self, pos_tag):
        nns = []
        vbs = []
        for word, pos in pos_tag:
            if pos in pos_NN:
                nns.append(word)
            if 'VB' in pos:
                vbs.append(word)

        for nn in nns:
            for vb in vbs:
                if (nn, vb) not in self.nn_vb_feature:
                    self.nn_vb_feature.append((nn, vb))


    def _generate_nn_vb_neg(self, pos_tag):
        nns = []
        vbs = []
        for word, pos in pos_tag:
            if pos in pos_NN and word in negators:
                nns.append(word)
            if 'VB' in pos:
                vbs.append(word)

        for nn in nns:
            for vb in vbs:
                if (nn, vb) not in self.nn_vb_neg_feature:
                    self.nn_vb_neg_feature.append((nn, vb))


    def _generate_nn_jj(self, pos_tag):
        nns = []
        jjs = []
        for word, pos in pos_tag:
            if pos in pos_NN:
                nns.append(word)
            if 'JJ' in pos:
                jjs.append(word)

        for nn in nns:
            for jj in jjs:
                if (nn, jj) not in self.nn_jj_feature:
                    self.nn_jj_feature.append((nn, jj))


    def _generate_nn_jj_neg(self, pos_tag):
        nns = []
        jjs = []
        for word, pos in pos_tag:
            if pos in pos_NN and word in negators:
                nns.append(word)
            if 'JJ' in pos:
                jjs.append(word)

        for nn in nns:
            for jj in jjs:
                if (nn, jj) not in self.nn_jj_neg_feature:
                    self.nn_jj_neg_feature.append((nn, jj))


    def _generate_rb_vb(self, pos_tag):
        rbs = []
        vbs = []
        for word, pos in pos_tag:
            if 'RB' in pos:
                rbs.append(word)
            if 'VB' in pos:
                vbs.append(word)

        for rb in rbs:
            for vb in vbs:
                if (rb, vb) not in self.rb_vb_feature:
                    self.rb_vb_feature.append((rb, vb))


    def _generate_rb_vb_neg(self, pos_tag):
        rbs = []
        vbs = []
        for word, pos in pos_tag:
            if 'RB' in pos and word in negators:
                rbs.append(word)
            if 'VB' in pos:
                vbs.append(word)

        for rb in rbs:
            for vb in vbs:
                if (rb, vb) not in self.rb_vb_neg_feature:
                    self.rb_vb_neg_feature.append((rb, vb))


if __name__ == '__main__':
    import codecs

    print "read corpus"
    corpus = []
    y = []
    count = 0
    limit = 2000
    sent_len_threshold = 0

    with codecs.open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/keyboard_1_rating_verbexpression.txt",
            "r", "utf-8") as f:
        for line in f:
            if len(nltk.tokenize.word_tokenize(line.strip())) >= sent_len_threshold and count < limit:
                corpus.append(line.strip())
                y.append(1)
                count += 1

    count = 0
    with codecs.open(
            "/Users/Derrick/PycharmProjects/opinion_exp/VEMarkovNetworks/data/keyboard_5_rating_verbexpression.txt",
            "r", "utf-8") as f:
        for line in f:
            if len(nltk.tokenize.word_tokenize(line.strip())) >= sent_len_threshold and count < limit:
                corpus.append(line.strip())
                y.append(0)
                count += 1
    print "read corpus end.", len(corpus)

    creator = FeatureFunctionCreator([1, 0])

    for c in corpus:
        tokens = nltk.word_tokenize(c)
        pos_tag = nltk.pos_tag(tokens)
        creator.handle_one_corpu(pos_tag)

    for c, y_i in zip(corpus, y):
        tokens = nltk.word_tokenize(c)
        pos_tag = nltk.pos_tag(tokens)
        print len(creator.get_all_feature(pos_tag, y_i))
        print creator.get_feature_num()
        break
