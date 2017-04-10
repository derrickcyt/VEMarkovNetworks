# -*- coding: utf-8 -*-
import numpy as np
from feature_extract import *
from scipy import misc, optimize
import random


class VEMarkovNetworks(object):
    def __init__(self, feature_num, feature_creator, sigma=10, optimize='bfgs'):
        self.w = np.zeros(feature_num, dtype=float)
        self.feature_creator = feature_creator
        self.y_set = []
        self.v = sigma ** 2
        self.v2 = self.v * 2
        self.optimize = optimize

    def regulariser(self, w):
        return np.sum(w ** 2) / self.v2

    def regulariser_deriv(self, w):
        return np.sum(w) / self.v

    def fit(self, X_pos_tags, y):
        """
        train model
        :param X: feature through feature_list
        :param y: label
        :return:
        """
        print "train model.", "feature num:", len(self.w)
        self.y_set = list(set(y))

        if self.optimize == 'bgfs':
            l = lambda w: self.neg_likelihood_derivative(X_pos_tags, y, w)
            val = optimize.fmin_l_bfgs_b(l, self.w)
            self.w, _, _ = val
        elif self.optimize == 'sgd':
            self.sgd(X_pos_tags, y)

        print self.w

    def predict(self, X_pos_tags):
        """
        predict labels based on X
        :param X: feature through feature_list
        :return:
        """
        ret = []
        probs = []
        for x_pos_tags in X_pos_tags:
            f_xm_y = [np.array(self.feature_creator.get_all_feature(x_pos_tags, y_i), dtype=float) for y_i in
                      self.y_set]

            p_y_base_xm = np.exp([np.dot(self.w, f_xm_y[i]) for i in xrange(len(self.y_set))])

            z = np.sum(p_y_base_xm)

            p_y_base_xm /= z

            ret.append(self.y_set[np.argmax(p_y_base_xm)])

            probs.append(p_y_base_xm)

        return np.array(ret), np.array(probs)

    def neg_likelihood_derivative(self, X_pos_tags, y, w):
        """
        function return objective function and derivative function for bfgs optimization
        :param X: features
        :param y: labels
        :param w: model parameters
        :return: objective values and derivative values
        """
        likelihood = 0
        derivative = np.zeros(len(w))
        for x_pos_tags, y_ in zip(X_pos_tags, y):
            f_xm_y = [np.array(self.feature_creator.get_all_feature(x_pos_tags, y_i), dtype=float) for y_i in
                      self.y_set]

            p_y_base_xm = np.exp([np.dot(w, f_xm_y[i]) for i in xrange(len(self.y_set))])

            z = np.sum(p_y_base_xm)

            p_y_base_xm /= z

            likelihood += np.dot(w, f_xm_y[self.y_set.index(y_)]) - np.log(z)

            derivative += f_xm_y[self.y_set.index(y_)] - (np.mat(f_xm_y).T * np.mat(p_y_base_xm).T).A1

        print likelihood
        return -likelihood + self.regulariser(w), -derivative + self.regulariser_deriv(w)

    def sgd(self, X_pos_tags, y, num_iter=150):
        mini_batch = 200
        alpha = 0.01

        features = []

        for x_pos_tags in X_pos_tags:
            features.append([np.array(self.feature_creator.get_all_feature(x_pos_tags, y_i), dtype=float) for y_i in
                        self.y_set])

        features = np.array(features, dtype=float)

        print "features have been generated"

        for j in xrange(num_iter):
            data_index = range(len(X_pos_tags))
            for i in xrange(len(X_pos_tags)):
                rand_index = int(random.uniform(0, len(data_index)))

                # x_pos_tags = X_pos_tags[rand_index]
                y_ = y[rand_index]
                f_xm_y = features[rand_index]

                p_y_base_xm = np.exp([np.dot(self.w, f_xm_y[i]) for i in xrange(len(self.y_set))])

                z = np.sum(p_y_base_xm)

                p_y_base_xm /= z

                derivative = f_xm_y[self.y_set.index(y_)] - (np.mat(f_xm_y).T * np.mat(p_y_base_xm).T).A1

                self.w += alpha * derivative

                del data_index[rand_index]

            print "iter", j,
            print "loss", self.loss(features, y)

    def loss(self, features, y):

        likelihood = 0
        for feature, y_ in zip(features, y):
            f_xm_y = feature

            p_y_base_xm = np.exp([np.dot(self.w, f_xm_y[i]) for i in xrange(len(self.y_set))])

            z = np.sum(p_y_base_xm)

            p_y_base_xm /= z

            likelihood += np.dot(self.w, f_xm_y[self.y_set.index(y_)]) - np.log(z)

        return -likelihood + self.regulariser(self.w)
