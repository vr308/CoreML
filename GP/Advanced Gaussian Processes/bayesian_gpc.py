#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:38:21 2019

@author: vidhi
"""

import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib.pylab as plt
import sys

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


np.random.seed(1)

# number of data points
n = 200

# x locations
x = np.linspace(-5, 5, n)

# true covariance
l = 0.1
n = 1.0
cov_func = n**2 * pm.gp.cov.ExpQuad(1, l)
K = cov_func(x[:,None]).eval()

# zero mean function
mean = np.zeros(n)

# sample from the gp prior
f_true = np.random.multivariate_normal(mean, K + 1e-6 * np.eye(n), 1).flatten()

# link function
def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps

y = pm.Bernoulli.dist(p=invlogit(f_true)).random()

# Plotting

fig = plt.figure(figsize=(12,5)); 
ax = fig.gca()
ax.plot(x, invlogit(f_true), 'dodgerblue', lw=3, label="True rate");
ax.plot(x, y, 'ko', ms=3, label="Observed data");
ax.set_xlabel("X"); 
ax.set_ylabel("y"); 
plt.legend();

# Laplace - ML-II







# HMC

with pm.Model() as model:
    # covariance function
    l = pm.Gamma("l", alpha=2, beta=2)
    # informative, positive normal prior on the period
    n = pm.HalfNormal("n", sd=3)
    cov = n**2 * pm.gp.cov.ExpQuad(1, l)

    gp = pm.gp.Latent(cov_func=cov)

    # make gp prior
    f = gp.prior("f", X=x[:,None])

    # logit link and Bernoulli likelihood
    p = pm.Deterministic("p", pm.math.invlogit(f))
    y_ = pm.Bernoulli("y", p=p, observed=y)

    trace = pm.sample(1000, chains=1)
    

n_pred = 200
X_new = np.linspace(0, 2.0, n_pred)[:,None]

with model:
    f_pred = gp.conditional("f_pred", X_new)

with model:
    pred_samples = pm.sample_ppc(trace, vars=[f_pred], samples=1000)
    
    
# plot the results
fig = plt.figure(figsize=(12,5)); 
ax = fig.gca()

# plot the data (with some jitter) and the true latent function
plt.plot(x, invlogit(trace['f'][::20].T), color='grey', alpha=0.5)
plt.plot(x, y, 'ok', ms=3, alpha=0.5, label="y");
plt.plot(x, invlogit(f_true), "k", lw=2, label="True f");
plt.plot(x, invlogit(trace['f'].T).mean(axis=1), color='b', label='HMC Mean')

# axis labels and title
plt.xlabel("X"); 
plt.ylabel("prob.");
plt.title("Posterior distribution over " + r'$(f(x), \theta)$', fontsize='small'); 
plt.legend();
    
# 
    