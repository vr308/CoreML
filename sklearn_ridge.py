#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:27:43 2018

@author: vr308
"""

"""
Various bayesian regression
"""

# Authors: V. Michel, F. Pedregosa, A. Gramfort
# License: BSD 3 clause

from math import log
import numpy as np
from scipy import linalg
from sklearn.utils import check_X_y

###############################################################################
# BayesianRidge regression

class BayesianRidge():
    """Bayesian ridge regression
    Fit a Bayesian ridge model and optimize the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).
    Read more in the :ref:`User Guide <bayesian_regression>`.
    Parameters
    ----------
    n_iter : int, optional
        Maximum number of iterations.  Default is 300.
    tol : float, optional
        Stop the algorithm if w has converged. Default is 1.e-3.
    alpha_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter. Default is 1.e-6
    alpha_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.
        Default is 1.e-6.
    lambda_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter. Default is 1.e-6.
    lambda_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.
        Default is 1.e-6
    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        Default is True.
    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    verbose : boolean, optional, default False
        Verbose mode when fitting the model.
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of distribution)
    alpha_ : float
       estimated precision of the noise.
    lambda_ : float
       estimated precision of the weights.
    sigma_ : array, shape = (n_features, n_features)
        estimated variance-covariance matrix of the weights
    scores_ : float
        if computed, value of the objective function (to be maximized)
    """

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, compute_score=True,
                 fit_intercept=False, normalize=False, copy_X=True,
                 verbose=True):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model
        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values. Will be cast to X's dtype if necessary
        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        #X, y, X_offset_, y_offset_, X_scale_ = self._preprocess_data(
        #    X, y, self.fit_intercept, self.normalize, self.copy_X)
        #self.X_offset_ = X_offset_
        #self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        alpha_ = 1. / np.var(y)
        lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.n_iter):

            # Compute mu and sigma
            # sigma_ = lambda_ / alpha_ * np.eye(n_features) + np.dot(X.T, X)
            # coef_ = sigma_^-1 * XT * y
            if n_samples > n_features:
                coef_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
                coef_ = np.dot(coef_, XT_y)
                if self.compute_score:
                    logdet_sigma_ = - np.sum(
                        np.log(lambda_ + alpha_ * eigen_vals_))
            else:
                coef_ = np.dot(X.T, np.dot(
                    U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T))
                coef_ = np.dot(coef_, y)
                if self.compute_score:
                    logdet_sigma_ = lambda_ * np.ones(n_features)
                    logdet_sigma_[:n_samples] += alpha_ * eigen_vals_
                    logdet_sigma_ = - np.sum(np.log(logdet_sigma_))

            # Preserve the alpha and lambda values that were used to
            # calculate the final coefficients
            self.alpha_ = alpha_
            self.lambda_ = lambda_

            # Update alpha and lambda
            rmse_ = np.sum(np.asarray(y - np.dot(X, coef_)) ** 2)
            gamma_ = (np.sum((alpha_ * eigen_vals_) /
                      (lambda_ + alpha_ * eigen_vals_)))
            lambda_ = ((gamma_ + 2 * lambda_1) /
                       (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
                      (rmse_ + 2 * alpha_2))


            # Compute the objective function
            if self.compute_score:
                s = lambda_1 * log(lambda_) - lambda_2 * lambda_
                s += alpha_1 * log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (n_features * log(lambda_) +
                            n_samples * log(alpha_) -
                            alpha_ * rmse_ -
                            (lambda_ * np.sum(coef_ ** 2)) -
                            logdet_sigma_ -
                            n_samples * log(2 * np.pi))
                self.scores_.append(s)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.coef_ = coef_
        sigma_ = np.dot(Vh.T,
                        Vh / (eigen_vals_ + self.lambda_ / self.alpha_)[:, np.newaxis])
        self.sigma_ = (1. / self.alpha_) * sigma_

        #self._set_intercept(X_offset_, y_offset_, X_scale_)
        return self

    def predict(self, X, return_std=False):
        """Predict using the linear model.
        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.
        Returns
        -------
        y_mean : array, shape = (n_samples,)
            Mean of predictive distribution of query points.
        y_std : array, shape = (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X = (X - self.X_offset_) / self.X_scale_
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
            return y_mean, y_std




br = BayesianRidge(n_iter=1)
br.fit(design_matrix, y_noise)