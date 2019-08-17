#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:09:58 2019

@author: vidhi

Composite Priors - Scratchpad work

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


def generate_gp_latent(X_star, mean, cov):
    
    return np.random.multivariate_normal(mean(X_star).eval(), cov=cov(X_star, X_star).eval())

def generate_gp_training(X_all, f_all, n_train, noise_var, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.5*st.norm.pdf(X_all, 2, 0.7) + 0.5*st.norm.pdf(X_all, 15.0,1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
     
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    X = X_all[train_index]
    f = f_all[train_index]
    X_star = X_all[test_index]
    f_star = f_all[test_index]
    y = f + np.random.normal(0, scale=noise_sd, size=n_train)
    return X, y, X_star, f_star, f

def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')

# Sum of SE Kernels in 1D
   
# Simulate some data 

mean = pm.gp.mean.Zero()
X_all = np.linspace(0,100,300)[:, None]

sig_sd = 10
ls = 10
noise_sd = 1

snr = sig_sd/noise_sd

cov = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)

f_all = generate_gp_latent(X_all, mean, cov)

uniform = True
n_train = 20
X, y, X_star, f_star, f = generate_gp_training(X_all, f_all, n_train, noise_sd, uniform)


# Plot

plot_noisy_data(X, y, X_star, f_star, '')


sig_sd_priors = []
ls_priors = []
n_sd_priors = []


with pm.Model() as sum_se:
      
        sig_sd = 
        ls = 
        noise_sd = 
    
        # Specify the covariance function.
        cov_func = pm.gp.cov.Constant(sig_sd)**2*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
        # Specify the GP. 
        gp = pm.gp.Marginal(cov_func=cov_func)
            
        # Marginal Likelihood
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
        





# Product of SE kernels in 1D






# SE-ARD Kernels






