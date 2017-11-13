#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:53:41 2017

@author: vr308
"""

# Fit a non-linear curve to data - Bayesian method

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.optimize as so
import statsmodels.api as smt
import pandas as pd


def predictive_distribution():
    
    return

def posterior_update():
    
    return 

if __name__ == "__main__":

    x = np.linspace(1,20,100)
    y_true = 0.5*np.sin(x) + 0.5*x + -0.02*(x-5)**2
    y_noise = y_true + 0.2*np.std(y_true)*np.random.normal(0,1,100)
    
    plt.figure()
    plt.plot(x,y_true, 'b')
    plt.scatter(x,y_noise, s=4)
    
    
    