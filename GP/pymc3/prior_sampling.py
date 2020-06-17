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
X = np.linspace(-1, 1, 200)[:,None]

class SpectralMixtureKernel(pm.gp.cov.Covariance):
    
    r"""
    The Spectral Mixture kernel
    .. math::
     k(x, x') = \sum_{q=1}^{Q} w_{q} \prod_{p=1}^{P}\exp\{-2\pi^{2}\tau_{p}^{2}v_{q}^{(p)}\}\cos\{2\pi\tau_{p}\mu_{q}^{p}\}
    """
    
    def __init__(self, input_dim, n_components, m_weights, m_variances, m_means):
        
        super().__init__(input_dim, active_dims=None)
        
        self.n_components = n_components
        self.m_weights = m_weights
        self.m_variances = m_variances
        self.m_means = m_means
        
        #assert(n_components == len(m_weights) == len(m_variances) == len(m_means))
        
    def square_dist(self, X, Xs):
        X = tt.mul(X, 1.0)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, 1.0)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)
        
    
    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        # make sure tau_sq is broadcastable to k component
        tau_sq = self.square_dist(X, Xs)[..., None]
        diff = tt.sqrt(tau_sq)
        exp_term = tt.exp(-2 * (np.pi ** 2) * tau_sq * self.m_variances)
        cos_term = tt.cos(2 * np.pi * diff * self.m_means)
        return tt.sum(self.m_weights * exp_term * cos_term, axis=-1)
    

with pm.Model() as sm_model:
    
    m_weights = np.array([0.5,0.5])
    m_means = np.array([4, 5])
    m_variances = np.array([4, 3])
    
    #log_var = pm.Gamma('log_var', alpha=2, beta=1) 
    
    var = pm.Lognormal('var', 0, 2)
    mu = pm.Lognormal('mu',0,2) 

    #mu = pm.Gamma('mu', alpha=2, beta=1) 
    
    cov_sm_fixed = SpectralMixtureKernel(1, 2, m_weights, m_variances, m_means)
    cov_sm_dist = SpectralMixtureKernel(1, 2, m_weights, var, mu)
    
    gp_sm_fixed = pm.gp.Latent(cov_func=cov_sm_fixed)
    gp_sm_dist = pm.gp.Latent(cov_func=cov_sm_dist)

    f_sm_fixed = gp_sm_fixed.prior('f_sm_fixed', X=X)
    f_sm_dist = gp_sm_dist.prior('f_sm_dist', X=X)
    
    trace_sm_fixed = pm.sample_prior_predictive(10, var_names=['f_sm_fixed'])
    trace_sm_dist = pm.sample_prior_predictive(10, var_names=['f_sm_dist', 'mu', 'var'])

plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(X, trace_sm_fixed['f_sm_fixed'][0:5].T, color='red', alpha=0.4)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Draws from ' + r'$p(f|\theta)$' + '\n' +  'Fixed ' r'$\theta$', fontsize='small')
plt.subplot(122)
plt.plot(X, trace_sm_dist['f_sm_dist'][0:5].T, color='blue', alpha=0.4)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Draws from ' + r'$p(f|\theta)$' + '\n' +  r'$ \theta \sim p(\theta)$', fontsize='small')

with pm.Model() as fixed_model:

       sig_sd = 1

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
       #log_p = pm.Gamma('log_p',alpha = 2, beta = 1)
       #p = pm.Deterministic('p', tt.exp(log_p))
       p = 3

       #fixed signal variance
       #sig_sd = pm.Gamma('sig_sd', mu=0, sigma=0.1)
       sig_sd = 1

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

       trace_hier = pm.sample_prior_predictive(10, var_names=['f_se','f_per', 'f_mat'])


plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(X, trace_fixed['f_se'].T)
plt.title('Squared Exponential Kernel' )
plt.xlabel('x')
plt.ylabel('f(x)')
plt.subplot(132)
plt.plot(X, trace_fixed['f_per'].T)
plt.title('Periodic Kernel')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.subplot(133)
plt.plot(X, trace_fixed['f_mat'].T)
plt.title('Matern32 kernel')
plt.xlabel('x')
plt.ylabel('f(x)')
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


# Individual plots with labels and heading

plt.figure(figsize=(4,4))
plt.plot(X, trace_fixed['f_se'].T)
plt.title('Sample draws from a GP prior' )
plt.xlabel('x')
plt.ylabel('f(x)')


# mini plots for table
X = np.linspace(-10, 10, 100)[:,None]
eta = 1
tau = 1

diff_x = 0 - X

list_K_se = []

ls = d.random(size=10)
for i in ls:
    #cov_se = eta**2 * pm.gp.cov.ExpQuad(1, ls=i)
    #cov_per = pm.gp.cov.Periodic(1, period=3, ls=i)
    cov_mat = tau * pm.gp.cov.Matern32(1, ls=i)
    list_K_se.append(cov_mat(X))


K_se = cov_se(X).eval()
K_per = cov_per(X).eval()
K_mat = cov_mat(X).eval()

plt.figure(figsize=(4,4))
for i in list_K_se:
    K = i.eval()
    plt.plot(diff_x, K[:,50], c='#1f77b4')
plt.xticks([])
plt.xlabel(r"$x - x'$", fontsize='large')


plt.figure(figsize=(4,4))
plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K_mat.shape[0]), cov=K_mat).random(size=10).T);
plt.ylabel(r"$f(x)$", fontsize='large');
plt.xlabel(r"$x$", fontsize='large');

plt.figure(figsize=(4,4))
plt.plot(X, trace_hier['f_mat'].T);
plt.ylabel(r"$f(x)$", fontsize='large');
plt.xlabel(r"$x$", fontsize='large');
