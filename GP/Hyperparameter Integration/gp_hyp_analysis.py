# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
from theano.tensor.nlinalg import matrix_inverse, det
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from matplotlib.colors import LogNorm

# DATA SET-UP

# Generating data

n = 30 # The number of data points
X = np.sort(np.linspace(0, 10, n))[:, None] # The inputs to the GP, they must be arranged as a column vector
X_star = np.linspace(0,10,1000)[:,None]

# Covariance kernel parameters 

lengthscale = 1.0
noise_var = 0.5
sig_var = 2.0

cov = sig_var*pm.gp.cov.ExpQuad(1, lengthscale)

K = cov(X)
K_s = cov(X, X_star)
K_ss = cov(X_star, X_star)
K_noise = K + noise_var*tt.eye(n)
K_inv = matrix_inverse(K_noise)

# Add very slight perturbation to the covariance matrix diagonal to improve numerical stability
K_stable = K + 1e-12 * tt.eye(n)

# A mean function that is zero everywhere
mean = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
f = np.random.multivariate_normal(mean(X).eval(), cov=K.eval()).flatten()
y = f + np.random.normal(0, scale=np.sqrt(noise_var), size=n)

# Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); 
ax = fig.gca()
ax.plot(X, f, "dodgerblue", lw=3, label="True f");
ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data");
ax.set_xlabel("X"); ax.set_ylabel("The true f(x)"); 
plt.legend();

# Analytically compute posterior mean and posterior covariance
# Algorithm 2.1 in Rasmussen and Williams

L = np.linalg.cholesky(K_noise.eval())
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
v = np.linalg.solve(L, K_s.eval())
post_mean = np.dot(K_s.T.eval(), alpha)

post_cov = K_ss.eval() - K_s.T.dot(K_inv).dot(K_s)
post_std = np.sqrt(np.diag(post_cov.eval()))
#ost_cov = K_ss.eval() - np.dot(v.T, v)

samples = np.random.multivariate_normal(post_mean, post_cov.eval(), 10)

plt.figure()
plt.plot(X_star, samples.T, color='orange', alpha=0.5)
plt.plot(X, f, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
plt.plot(X, y, 'rx', ms=3, alpha=0.8, color='r');
plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
plt.fill_between(np.ravel(X_star), post_mean - 1.96*post_std, 
                 post_mean + 1.96*post_std, alpha=0.2, color='r',
                 label='95% CR')
plt.legend(fontsize='x-small')
plt.title('GPR')

# Vanilla analytical GP in pymc3

with pm.Model() as latent_gp_model:
    # Specify the covariance function.
    cov_func = pm.gp.cov.ExpQuad(1, ls=1)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Latent(cov_func=cov_func)

    # Place a GP prior over the function f.
    f = gp.prior("f", X=X)
    
    f_star = gp.conditional("f_star", X_star)
    
    plt.figure()
    plt.plot(X, f, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X_star, f_star, color='r', lw=2, label='Posterior mean')
   
    
    

# Type II ML estimation 



# HMC Sampling -> lengthscale




# HMC Sampling -> noise var





# HMC Sampling -> signal var




