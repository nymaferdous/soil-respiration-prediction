# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp


class WeightedGaussianDiscriminantAnalysis(object):
    '''
    Weighted Gaussian Discriminant Analysis

    A classifier with linear decision boundary , generated by fitting class
    conditional densities with Gaussian. And using using Bayes rule
    to obtain posterior distribution.
    '''

    def __init__(self, stop_learning=1e-3, bias_term=True):
        self.stop_learning = stop_learning
        self.bias_term = bias_term
        self.delta_param_norm = 0
        self.delta_log_like = 0
        self.means = None

    def init_params(self, m, k):
        '''
        Initialises parameters

        Parameters:
        -----------

        m: int
            Dimensionality of data

        k: int
            Number of classes

        '''
        self.k = k
        self.cov = np.eye(m)
        self.means = np.random.random([m, k])
        self.log_priors = -1 * np.log(np.ones(k) * k)

    def _bias_term_pre_processing_X(self, X, bias_term):
        '''
        Preprocesses X and adjusts for bias term

        Returns:
        --------
          X: numpy array of size 'n x (m-1)'
             Design matrix without column of bias_term, which is expected to be
             last column
        '''
        if bias_term is None:
            bias_term = self.bias_term
        if bias_term is True:
            return X[:, :-1]
        return X

    def fit(self, Y, X, weights=None, bias_term=None):
        '''
        Finds parameters of weighted gaussian discriminant analysis that maximise
        maximum likelihood.

        Parameters:
        -----------

        X: numpy array of size 'n x m'
            Expalanatory variables

        Y: numpy array of size 'n x 1'
            Dependent variables that need to be approximated

        weights: numpy array of size 'n x 1'
            Weighting for each observation

        bias_term: bool
            If True, matrix of explanatory variables already contains bias term,
            which should be discarded in estimation (expected that bias term is in last
            column of X matrix)

        '''

        # preprocess X if it contains bias term
        X = self._bias_term_pre_processing_X(X, bias_term)

        n, m = np.shape(X)
        k = self.k
        weights_total = np.sum(weights)

        if weights is None:
            weights = np.ones(n)

        # Interestingly loop was faster than using outer product
        Y_w = (Y.T * weights).T

        # recovery in case of decrease in log-likelihood (NUMERICAL UNDERFLOW ISSUE IN DEEP
        # HIERARCHICAL MIXTURE OF EXPERTS)
        mean_recovery = self.means
        cov_recovery = self.cov
        prior_recovery = self.log_priors
        log_like_before = self.log_likelihood(X, Y_w, weights, weighted_Y=True, bias_term=False)

        # calculate log priors
        weighted_norm = np.sum(Y_w, axis=0)
        self.log_priors = np.log(weighted_norm) - np.log(weights_total)

        # calculate weighted means of Gaussians for each class
        weighted_sum = np.dot(X.T * weights, Y)
        self.means = weighted_sum / weighted_norm

        # calculate pooled covarince matrix
        self.cov = np.zeros([m, m])
        cov = np.zeros([m, m])
        M = np.zeros([m, n])
        for i in range(k):
            np.outer(self.means[:, i], np.ones(n), out=M)
            X_cent = (X - M.T)
            np.dot(X_cent.T * Y_w[:, i], X_cent, out=cov)
            self.cov += cov
        self.cov /= weights_total

        # check that log-likelihood did not dropped (UNDERFLOW IN DEEP HMEs)
        # or incresed by very little (for preventing overfitting and long iteration
        # cycle
        log_like_after = self.log_likelihood(X, Y_w, weights, bias_term=False, weighted_Y=True)
        delta_log_like = (log_like_after - log_like_before) / n
        if delta_log_like < self.stop_learning:
            self.means = mean_recovery
            self.cov = cov_recovery
            self.log_priors = prior_recovery
            delta_log_like = 0

            # saves changes in likelihood and parameters in instance variables
        delta = self.means - mean_recovery
        self.delta_param_norm = np.sum(np.dot(delta.T, delta))
        self.delta_log_like = delta_log_like

    def predict_probs(self, X, bias_term=None):
        '''
        Calculates posterior probability of x belonging to any particular class

        Parameters:
        -----------

        X: numpy array of size 'unknown x m'
            Expalanatory variables

        bias_term: bool
            If True , explanatory variables matrix contains bias_term (bias term should be
            in last column of design matrix)

        Returns:
        --------

        prior_prob: numpy array of size 'unknown x k'
            Posterior probability that class belongs to particular probability

        '''
        prior_prob = np.exp(self.predict_log_probs(X, bias_term))
        return prior_prob

    def predict_log_probs(self, X, bias_term=None):
        '''
        Calculates log of probabilities

        Parameters:
        -----------

        X: numpy array of size 'unknown x m'
            Expalanatory variables

        bias_term: bool
            If True , explanatory variables matrix contains bias_term (bias term should be
            in last column of design matrix)

        Returns:
        --------

        prior_prob: numpy array of size 'unknown x k'
            Posterior probability that class belongs to particular probability

        '''
        X = self._bias_term_pre_processing_X(X, bias_term)
        n, m = np.shape(X)
        log_posterior = np.zeros([n, self.k])
        for i in range(self.k):
            log_posterior[:, i] = mvn.logpdf(X, self.means[:, i], cov=self.cov)
            log_posterior[:, i] += self.log_priors[i]
        normaliser = logsumexp(log_posterior, axis=1)
        posterior_log_prob = (log_posterior.T - normaliser).T
        return posterior_log_prob

    def log_likelihood(self, X, Y, weights=None, bias_term=None, weighted_Y=False):
        '''
        Calculates log likelihood for weighted gaussian discriminant analysis

        Parameters:
        -----------

        X: numpy array of size 'n x m'
             Explanatory variables

        Y: numpy array of size 'n x 1'
             Target variable can take only values 0 or 1

        weights: numpy array of size 'n x 1'
             Weights for observations

        k: int
             Number of classes

        bias_term: bool
             If True excludes bias term (which is expected to be in last column of X)

        weighted_Y:
             If True Y is already weighted (optimisation so that not recalculate Y*w)

        Returns:
        --------

        log_like: float
             Log likelihood
        '''
        X = self._bias_term_pre_processing_X(X, bias_term)
        n, m = np.shape(X)

        # default weights
        if weights is None:
            weights = np.ones(n)

        # log-likelihood
        log_posterior = np.zeros([n, self.k])
        for i in range(self.k):
            log_posterior[:, i] = mvn.logpdf(X, self.means[:, i], cov=self.cov)
            log_posterior[:, i] += self.log_priors[i]
        if weighted_Y is False:
            Y = (Y.T * weights).T
        log_like = np.sum(Y * log_posterior)
        return log_like

    def posterior_log_probs(self, X, Y, bias_term=None):
        '''
        Probability of observing Y given X and parameters
        '''
        X = self._bias_term_pre_processing_X(X, bias_term)
        log_P = np.sum(Y * self.predict_log_probs(X, bias_term=False), axis=1)
        return log_P
