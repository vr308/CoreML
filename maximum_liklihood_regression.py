#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:03:56 2017

@author: vr308
"""

import numpy as np
import numdifftools as ndt
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.optimize as so
import statsmodels.api as smt
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.base.model import GenericLikelihoodModel

# Prediction Interval under maximum likelihood is same as in Least Squares.

def log_likelihood(params, x, y_noise):
    
    slope = params[0]
    intercept = params[1]
    noise_var = params[2]
    fitted = x.dot(slope) + intercept
    l = np.sum(st.norm.logpdf(y_noise, loc=fitted, scale= noise_var))
    return l

def _ll_ols(y, X, beta, sigma):
    mu = X.dot(beta)
    return st.norm(mu,sigma).logpdf(y).sum()

def log_likelihood_matrix(params, X, y_noise):
    
    noise_var = params[-1]
    fitted = X.dot(params[:-1])
    l = -np.sum(st.norm.logpdf(y_noise, loc=fitted, scale= noise_var))
    return l

class MyOLS(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyOLS, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
	#sigma = params[-1]
	#beta = params[:-1]
	ll = log_likelihood(params, self.exog, self.endog)
    #ll = log_likelihood_matix(params, self.exog, self.endog)
	return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
	# we have one additional parameter and we need to add it for summary
	self.exog_names.append('sigma')
	if start_params == None:
	    # Reasonable starting values
	    #start_params = np.append([0.5,-0.05,0.5], 1)
            start_params = np.append([1,1], 1)
	return super(MyOLS, self).fit(start_params=start_params,
				     maxiter=maxiter, maxfun=maxfun,
				     **kwds)

    #Hfun = ndt.Hessian(log_likelihood_matrix(initial, X, y_noise), full_output=True)
    #hessian_ndt, info = Hfun(results.x)
  
    
if __name__ == "__main__":
    
    x = np.linspace(1,20,100)
    y_true = 0.5*np.sin(x) + 0.5*x + -0.02*(x-5)**2
    y_noise = y_true + 0.2*np.std(y_true)*np.random.normal(0,1,100)
    X = smt.add_constant(x)
    
    # Fit a line, one regressor - maximum likelihood method
    
    initial = [1,1,1]
    results = so.minimize(log_likelihood, initial, args=(x,y_noise))
    print results.x
    
    # Fit a curve, multiple regressors
    initial = [0.5,-0.05,0.5,1,1]
    X = np.vstack([x**3, x**2, x, np.ones(len(x))]).T
    results = so.minimize(log_likelihood_matrix, initial, args=(X, y_noise))
    results.x
    
    Xframe = pd.DataFrame({'x3':x**3, 'x2': x**2, 'x1': x, 'const': np.ones(len(x))})
    
    ll_manual = MyOLS(y_noise, Xframe[['x1', 'const']]).fit()
    ll_manual.summary()
    
    plt.figure()
    plt.plot(x, y_true)
    plt.scatter(x, y_noise, s=4)
    plt.plot(x, fitted)