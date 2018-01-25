#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:03:56 2017

@author: vr308
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.optimize as so
import statsmodels.api as smt
import pandas as pd
import numdifftools as ndt

# Prediction Interval under maximum likelihood is same as in Least Squares.

def neg_log_like(params):
    
    sigma = params[-1]
    fitted = X.dot(params[:-1])
    l = -np.sum(st.norm.logpdf(y_noise, loc=fitted, scale=sigma))
    return l
  
def calcMLE_coefs(X,y_noise):
        
    X_plus = np.linalg.inv(np.dot(X.T,X)).dot(X.T)
    return np.dot(X_plus,y_noise)
        
    
if __name__ == "__main__":
    
    x = np.linspace(1,20,100)
    y_true = 0.5*np.sin(x) + 0.5*x + -0.02*(x-5)**2
    y_noise = y_true + 0.2*np.std(y_true)*np.random.normal(0,1,100)    
    
    # Fit a curve, multiple regressors
    initial = [0.5,-0.05,0.5,1,1]
    X = np.vstack([np.sin(x), x**2, x, np.ones(len(x))]).T
    
    mle_coefs = calcMLE_coefs(X, y_noise)
    
    # Least squares coefs to verify (they should match)
    
    lsq_coefs = np.linalg.lstsq(X,y_noise)[0]
        
    results = so.minimize(neg_log_like, initial, method = 'Nelder-Mead', options={'disp': True})
    results.x
    
    Hfun = ndt.Hessian(neg_log_like, full_output=True)
    hessian_ndt, info = Hfun(results['x'])
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    results = pd.DataFrame({'parameters':results['x'],'std err':se})
    results.index=['coef1','coef2','coef3','coef4','Sigma']   
    results.head()
    
    fitted = np.dot(X,mle_coefs)

    # Plotting the data

    plt.figure()
    plt.plot(x, y_true)
    plt.scatter(x, y_noise, s=4)
    plt.plot(x, fitted)
    
    # just check if the standard errors from 
    # MLE estimator of the noise variance sigma^2
    
    mle_noise_var = np.sum(np.square(fitted-y_noise))/len(x)
    
    # OLS estimator of the noise variance
    
    ols_noise_var = np.sum(np.square(fitted-y_noise))/(len(x) - 4)
    
    #4 is the number of regressors