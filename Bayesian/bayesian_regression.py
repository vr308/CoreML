#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:53:41 2017

@author: vr308
"""

# Fit a non-linear curve to data - Bayesian linear method
# Assuming we know the alphas (hyperparameters) and variance of targets sigma^2
# Also assuming conjugate prior with Gaussian Likelihood

# Highlight the differences between Bayes Linear Regression, Bayesian Ridge Rgression and ARD Regression. 

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.optimize as so
import statsmodels.api as smt
import pandas as pd
from scipy.stats import gaussian_kde

def distSquared(X,Y):
    nx = np.size(X, 0)
    ny = np.size(Y, 0)
    D2 =  (np.multiply(X, X).sum(1)  * np.ones((1, ny)) ) + ( np.ones((nx, 1)) * np.multiply(Y, Y).sum(1).T  ) - 2*X*Y.T
    return D2


def determine_gaussian_basis():
    
    

def predictive_distribution():
    
    return

def posterior_update():
    
    return 


def 

if __name__ == "__main__":

    N = 100
    u = np.sort(np.random.uniform(-10,10,N))
    x = np.matrix(u).T
    y_true = np.matrix(5*np.sin(u) + 0.5*u + -0.2*np.square((u-5))).T
    
    noiseToSignal = 0.2
    noise = np.std(y_true, ddof=1) * noiseToSignal
    y_noise = y_true + noise*np.random.randn(N,1)

    plt.figure()
    plt.plot(x,y_true, 'b')
    plt.plot(x,y_noise, 'ko',markersize=2)
    
    #Design Matrix
    
    bw = 0.4 # basis hyperparameter
    BASIS = np.matrix(np.exp(-distSquared(x,x)/(bw**2)))
    plt.plot(x,BASIS)
    
    # Define the parameters of the prior for weights    
    # Select a precision factor for the weights
    
    alpha = 0.5
    
    mu = 0
    A = alpha*np.identity(100)
    
  
   
# Bayesian Regression using pymc

from pymc3 import  *

import numpy as np
import matplotlib.pyplot as plt

size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    
    # Define priors
    #sigma = HalfCauchy('sigma', beta=10, testval=1.)
    sigma = 0.5
    intercept = Normal('Intercept', 0, sd=20)
    x_coeff = Normal('x', 0, sd=20)
    
    # Define likelihood
    likelihood = Normal('y', mu=intercept + x_coeff * x, 
                        sd=sigma, observed=y)
    
    # Inference!
    trace = sample(progressbar=False) # draw posterior samples using NUTS sampling
    
traceplot(trace)
plt.tight_layout();
        
plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
plots.plot_posterior_predictive_glm(trace, samples=100, 
                                    label='posterior predictive regression lines')
plt.plot(x, true_regression_line, label='true regression line', lw=1., c='y')

plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y');


# Testing with hand-written code

from scipy.stats import multivariate_normal as mv_norm

noise_var = 0.25
beta = 1.0/noise_var
prior_mean = np.asarray([0,0]).reshape(-1,1)
prior_var = np.asarray([400,400])*np.identity(2)

X = np.ones((len(data['x']), 2))
X[:, 1] = data['x']
y = data['y'].reshape(data['y'].shape + (1,))

def get_conjugate_normal_posterior(X, y, prior_mean, prior_var):
    
    # note - prior mean and variance is two-dimensional owing to slope and intercept terms
   pos_var = np.linalg.inv(np.linalg.inv(prior_var) + beta*X.T.dot(X))
   pos_mean = pos_var.dot(np.linalg.inv(prior_var).dot(prior_mean) + beta*X.T.dot(y))
   return pos_mean, pos_var

pos_mean, pos_var = get_conjugate_normal_posterior(X, y, prior_mean, prior_var)

print(pos_mean)
print(pos_var)

