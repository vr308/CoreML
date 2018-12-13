#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:40:22 2018

@author: vr308
"""
import sys
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt

# set the seed
np.random.seed(1)

# Setting up a GP model 

n = 100 # The number of data points
X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

# Define the true covariance function and its parameters
l_true = 1.0
n_true = 2.0
noise_var = 2.0
cov_func = n_true**2 * pm.gp.cov.ExpQuad(1, l_true)
cov_func_noise = n_true**2 * pm.gp.cov.ExpQuad(1, l_true) + pm.gp.cov.WhiteNoise(sigma=0.1)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
f_true = np.random.multivariate_normal(mean_func(X).eval(),
                                       cov_func(X).eval() + 1e-8*np.eye(n), 1).flatten()
y = f_true + noise_var*np.random.randn(n)

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True f");
ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data");
ax.set_xlabel("X"); ax.set_ylabel("The true f(x)"); plt.legend();

# Type II ML / Plug-in approach (Flat prior on hyp and find MAP -> same as type 2 ML)

with pm.Model() as marginal_gp_model:

   # prior on lengthscale 
   l = pm.Uniform('l', 0, 10)
   
   #prior on signal variance
   log_n = pm.Uniform('log_n', lower=-10, upper=5)
   n = pm.Deterministic('n', tt.exp(log_n))
   
   #prior on noise variance
   log_sig = pm.Uniform('log_sig', lower=-10, upper=5)
   sigma = pm.Deterministic('sigma', tt.exp(log_sig))
   
   # covariance functions for the function f and the noise
   f_cov = n*pm.gp.cov.ExpQuad(1, l)
   
   gp = pm.gp.Marginal(cov_func=f_cov)
   y_obs = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)
   #y_obs = pm.gp('y_obs', cov_func=f_cov, sigma=sigma, observed={'X':X, 'Y':y})
   mp = pm.find_MAP()
         
with marginal_gp_model:
      trace = pm.sample(1000)
      
# HMC Sampling approach 



# VI approach 
      
      
      