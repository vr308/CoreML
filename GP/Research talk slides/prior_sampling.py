#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:44 2020

@author: vidhi
"""


import numpy as np
import math
from matplotlib import pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import matplotlib.cm as cmap
import arviz as arv

# A one dimensional column vector of inputs.
X = np.linspace(0, 10, 100)[:,None]

with pm.Model() as fixed_model:

       sig_sd = 2

       # Specify the covariance functions
       cov_func_se = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=2)
       cov_func_per = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.Periodic(1, period=1, ls=2)
       cov_func_mat = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.Matern32(1, ls=2)

       gp_se = pm.gp.Latent(cov_func=cov_func_se)
       gp_per = pm.gp.Latent(cov_func=cov_func_per)
       gp_mat = pm.gp.Latent(cov_func=cov_func_mat)

       f_se = gp_se.prior("f_se", X=X)
       f_per = gp_per.prior("f_per", X=X)
       f_mat = gp_mat.prior("f_mat", X=X)

       trace_fixed = pm.sample_prior_predictive(15, var_names=['f_se','f_per', 'f_mat'])


with pm.Model() as hierarchical_model:

       # prior on lengthscale
       log_ls = pm.Gamma('log_ls',alpha = 2, beta = 1)
       ls = pm.Deterministic('ls', tt.exp(log_ls))

       # prior on period
       log_p = pm.Gamma('log_p',alpha = 2, beta = 1)
       p = pm.Deterministic('p', tt.exp(log_p))

       #fixed signal variance
       #sig_sd = pm.Gamma('sig_sd', mu=0, sigma=0.1)
       sig_sd = 2

       # Specify the covariance functions
       cov_func_se = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
       cov_func_per = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.Periodic(1, period=p, ls=ls)
       cov_func_mat = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.Matern32(1, ls=ls)

       gp_se = pm.gp.Latent(cov_func=cov_func_se)
       gp_per = pm.gp.Latent(cov_func=cov_func_per)
       gp_mat = pm.gp.Latent(cov_func=cov_func_mat)

       f_se = gp_se.prior("f_se", X=X)
       f_per = gp_per.prior("f_per", X=X)
       f_mat = gp_mat.prior("f_mat", X=X)

       trace_hier = pm.sample_prior_predictive(15, var_names=['f_se','f_per', 'f_mat'])


plt.figure(figsize=(15,10))
plt.subplot(231)
plt.plot(X, trace_fixed['f_se'].T, color='grey')
plt.title('SE (ls=2)' )
plt.subplot(232)
plt.plot(X, trace_fixed['f_per'].T,color='grey')
plt.title('Periodic (ls=2, p=1)')
plt.subplot(233)
plt.plot(X, trace_fixed['f_mat'].T,color='grey')
plt.title('Matern32 (ls=2)')
plt.subplot(234)
plt.plot(X, trace_hier['f_se'].T)
plt.title(r'$\mathcal{ls} \sim Gamma(2,1)$')
plt.subplot(235)
plt.plot(X, trace_hier['f_per'].T)
plt.title(r'$\mathcal{ls} \sim Gamma(2,1)$' + '$\mathcal{p} \sim Gamma(2,1)$')
plt.subplot(236)
plt.plot(X, trace_hier['f_mat'].T)
plt.title(r'$\mathcal{ls} \sim Gamma(2,1)$')
plt.suptitle('Sampling from the prior')
