#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:53:41 2017

@author: vr308
"""

# Fit a non-linear curve to data - Bayesian linear method

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
    
    #Define Basis Matrix
    
    bw = 0.4
    BASIS = np.matrix(np.exp(-distSquared(x,x)/(bw**2)))
    plt.plot(x,BASIS)
    
    # Define the parameters of the prior
    
    # Normal Prior 
    
    # Select a precision factor for the weights
    
    mu = 0
    Sigma =
    
   # Sparsity Prior 











   from sklearn.neighbors import NearestNeighbors
   X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
   nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
   distances, indices = nbrs.kneighbors(X)
   
   kde = gaussian_kde(u, bw_method=0.2)
   density = kde.evaluate(u)
   plt.plot(u,density)
   plt.hist(u, normed=True)
    
    