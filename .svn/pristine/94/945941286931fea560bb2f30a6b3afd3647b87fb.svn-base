# -*- coding: utf-8 -*-
import numpy as np
from feature_extract import *
from scipy import misc, optimize


class VEMarkovNetworks(object):
    def __init__(self, feature_num, feature_list, sigma=10):
        self.w = np.zeros(feature_num, dtype=float)
        self.f_list = feature_list
        self.y_set = []
        self.v = sigma ** 2
        self.v2 = self.v * 2

    def transfer(self, corpus):
        """
        transfer text to Feature
        :param corpus: text set
        :return: Feature set
        """
        print "transfer corpus."
        return [Feature(c) for c in corpus]

    def regulariser(self, w):
        return np.sum(w ** 2) / self.v2

    def regulariser_deriv(self, w):
        return np.sum(w) / self.v

    def fit(self, X_features, y):
        """
        train model
        :param X: feature through feature_list
        :param y: label
        :return:
        """
        print "train model."
        self.y_set = list(set(y))
        l = lambda w: self.neg_likelihood_derivative(X_features, y, w)
        val = optimize.fmin_l_bfgs_b(l, self.w)
        self.w, _, _ = val
        print self.w

    def predict(self, X_features):
        """
        predict labels based on X
        :param X: feature through feature_list
        :return:
        """
        ret = []
        for x_features in X_features:
            f_xm_y = [np.array([f_i(x_features, y_i) for f_i in self.f_list], dtype=float) for y_i in self.y_set]

            p_y_base_xm = np.exp([np.dot(self.w, f_xm_y[i]) for i in xrange(len(self.y_set))])

            z = np.sum(p_y_base_xm)

            p_y_base_xm /= z

            ret.append(self.y_set[np.argmax(p_y_base_xm)])

        return np.array(ret)

    def neg_likelihood_derivative(self, X_features, y, w):
        """
        function return objective function and derivative function for bfgs optimization
        :param X: features
        :param y: labels
        :param w: model parameters
        :return: objective values and derivative values
        """
        likelihood = 0
        derivative = np.zeros(len(w))
        for x_features, y_ in zip(X_features, y):
            f_xm_y = [np.array([f_i(x_features, y_i) for f_i in self.f_list], dtype=float) for y_i in self.y_set]

            p_y_base_xm = np.exp([np.dot(w, f_xm_y[i]) for i in xrange(len(self.y_set))])

            z = np.sum(p_y_base_xm)

            p_y_base_xm /= z

            likelihood += np.dot(w, f_xm_y[self.y_set.index(y_)]) - np.log(z)

            derivative += f_xm_y[self.y_set.index(y_)] - (np.mat(f_xm_y).T * np.mat(p_y_base_xm).T).A1

        print likelihood
        return -likelihood + self.regulariser(w), -derivative + self.regulariser_deriv(w)
