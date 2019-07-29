#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:50:29 2019

@author: vidhi
"""

import pandas as pd
import pymc3 as pm
from theano import tensor as tt
import scipy as sp
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

def plot_weights(trace):

      fig, ax = plt.subplots(figsize=(8, 6))
      plot_w = np.arange(K) + 1
      ax.bar(plot_w - 0.5, trace['w'].mean(axis=0), width=1., lw=0);
      ax.set_xlim(0.5, K);
      ax.set_xlabel('Component');
      ax.set_ylabel('Posterior expected mixture weight');
      
      
def plot_posterior_densities(y, x_plot, trace, post_pdf_contribs, true_density):
      
      fig, ax = plt.subplots(figsize=(8, 6))
     
      post_pdfs = (trace['w'][:, np.newaxis, :][0:1000] * post_pdf_contribs).sum(axis=-1)
      post_pdf_low_95, post_pdf_high_95 = np.percentile(post_pdfs, [2.5, 97.5], axis=0)
      post_pdf_low_99, post_pdf_high_99 = np.percentile(post_pdfs, [0.5, 99.5], axis=0)
      
      
      #ax.hist(y, bins=100, normed=True,
      #        color='b', lw=0, alpha=0.2);
      ax.fill(x_plot[:, 0], true_density, fc='black', alpha=0.2)
      ax.plot(x_plot[:, 0], true_density, c='black', alpha=0.2, label='Ground truth')
      ax.plot(x_plot, post_pdfs[0],
              c='r', label='Posterior sample densities');
      ax.plot(x_plot, post_pdfs[::10].T, c='r', alpha=0.4);
      ax.plot(x_plot, post_pdfs.mean(axis=0),
              c='k', label='Posterior expected density');
      ax.fill_between(x_plot[:,0], post_pdf_low_99, post_pdf_high_99,
                      color='yellow');
      ax.fill_between(x_plot[:,0], post_pdf_low_95, post_pdf_high_95,
                      color='green');
      ax.set_xlabel('y');
      ax.set_yticklabels([]);
      ax.plot(y, np.full(y.shape[0], -0.01), '+k')
      ax.set_ylabel('Posterior over densities');
      ax.legend(loc=2);


def plot_uncertainty_intervals(x_plot, trace, post_pdf_contribs, true_density):
      
      fig, ax = plt.subplots(figsize=(8, 6))
      post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
      post_pdf_low_95, post_pdf_high_95 = np.percentile(post_pdfs, [2.5, 97.5], axis=0)
      post_pdf_low_99, post_pdf_high_99 = np.percentile(post_pdfs, [0.5, 99.5], axis=0)
      
      ax.fill(x_plot[:, 0], true_density, fc='black', alpha=0.2,
        label='Ground truth')
      ax.fill_between(x_plot[:,0], post_pdf_low_99, post_pdf_high_99,
                      color='yellow', label='95% Credible Interval');
      ax.fill_between(x_plot[:,0], post_pdf_low_95, post_pdf_high_95,
                      color='green', label='99% Credible Interval');
      ax.set_xlabel('y');
      ax.set_yticklabels([]);
      ax.set_ylabel('Posterior over densities');
      ax.legend(loc=2);

      
if__name__== '__main__':
      
# Known Gaussian mixture 

K = 20

w = np.array([0.5,0.3,0.2])
mu = np.array([-7,0,2])
sd = np.array([3, 2, 1])
tau = 1/sd**2 #[0.0123, 0.25, 1]

y_gmix =  pm.NormalMixture.dist(w=w, mu=mu, sd=sd)

gmix_obs_s05 = []
gmix_obs_s1 = []
gmix_obs_s2 = []

for i in np.arange(1000):
      gmix_obs_s05.append(y_gmix.random() + np.random.normal(scale=0.5))
      gmix_obs_s1.append(y_gmix.random() + np.random.normal(scale=1))
      gmix_obs_s2.append(y_gmix.random() + np.random.normal(scale=2))

      
y_05 = np.array(gmix_obs_s05)
y_m_05 = np.array(random.sample(gmix_obs_s05, 300))
y_s_05 = np.array(random.sample(gmix_obs_s05, 100))

y_1 = np.array(gmix_obs_s1)
y_m_1 = np.array(random.sample(gmix_obs_s1, 300))
y_s_1 = np.array(random.sample(gmix_obs_s1, 100))

y_2 = np.array(gmix_obs_s2)
y_m_2 = np.array(random.sample(gmix_obs_s2, 300))
y_s_2 = np.array(random.sample(gmix_obs_s2, 100))
  
plt.hist(gmix_obs, density=True, bins=100)

varnames = ['alpha', 'mu', 'tau']

with pm.Model() as g_model:
      
      # Prior over the concentration parameter
      alpha = pm.Gamma('alpha', 2, 2)
      
      # Stick breaking construction for weights 
      beta = pm.Beta('beta', 1, alpha, shape=K)
      w = pm.Deterministic('w', stick_breaking(beta))
      
      # Prior over component parameters mu and tau
      
      log_tau = pm.Normal('log_tau', 0, 5, shape=K)
      tau = pm.Deterministic('tau', np.exp(log_tau))
      mu = pm.Normal('mu', 0, 8, shape=K)
      #mu_raw = pm.Normal('mu_raw', 0, 1, shape=K)
      #mu = pm.Deterministic('mu', 0 + mu_raw*1/np.sqrt(tau))
      
      #obs 
      obs = pm.NormalMixture('obs', w, mu, tau=tau,
                           observed=y_1)
      
with g_model:
      
      trace_nuts_g_1 = pm.sample(tune=500,draws=1000, chains=2, target_accept=0.95)
      
# Results 

x_plot = np.linspace(-18, 10, 1000)[:, np.newaxis]
true_density=np.exp(y_gmix.logp(x_plot)).eval()
      
# KDE

def get_kernel_density_fit(y):
      
      return KernelDensity(kernel='gaussian').fit(y[:,np.newaxis])
      
kde_05 = get_kernel_density_fit(y_05)
kde_m_05 = get_kernel_density_fit(y_m_05)
kde_s_05 = get_kernel_density_fit(y_s_05)

kde_1 = get_kernel_density_fit(y_1)
kde_m_1 = get_kernel_density_fit(y_m_1)
kde_s_1 = get_kernel_density_fit(y_s_1)

kde_2 = get_kernel_density_fit(y_2)
kde_m_2 = get_kernel_density_fit(y_m_2)
kde_s_2 = get_kernel_density_fit(y_s_2)


plt.figure(figsize=(10,10))

plt.subplot(331)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_05.score_samples(x_plot)))
plt.plot(y_05, np.full(y_05.shape[0], -0.01), '+k')

plt.subplot(334)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_m_05.score_samples(x_plot)))
plt.plot(y_m_05, np.full(y_m_05.shape[0], -0.01), '+k')

plt.subplot(337)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_s_05.score_samples(x_plot)))
plt.plot(y_s_05, np.full(y_s_05.shape[0], -0.01), '+k')


plt.subplot(332)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_1.score_samples(x_plot)))
plt.plot(y_1, np.full(y_1.shape[0], -0.01), '+k')

plt.subplot(335)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_m_1.score_samples(x_plot)))
plt.plot(y_m_1, np.full(y_m_1.shape[0], -0.01), '+k')

plt.subplot(338)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_s_1.score_samples(x_plot)))
plt.plot(y_s_1, np.full(y_s_1.shape[0], -0.01), '+k')

plt.subplot(333)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_2.score_samples(x_plot)))
plt.plot(y_2, np.full(y_2.shape[0], -0.01), '+k')

plt.subplot(336)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_m_2.score_samples(x_plot)))
plt.plot(y_m_2, np.full(y_m_2.shape[0], -0.01), '+k')

plt.subplot(339)
plt.plot(x_plot, true_density, color='g')
plt.plot(x_plot, np.exp(kde_s_2.score_samples(x_plot)))
plt.plot(y_s_2, np.full(y_s_2.shape[0], -0.01), '+k')

plt.suptitle('Non-parametric KDE')


# DPM

trace = trace_nuts_g_05
         
post_pdf_contribs_nuts = sp.stats.norm.pdf(x_plot,
                                            trace['mu'][:, np.newaxis, :][0:1000],
                                            1. / np.sqrt(trace['tau'])[:, np.newaxis, :][0:1000])
      
plot_posterior_densities(y, x_plot, trace, post_pdf_contribs_nuts, true_density)
plot_uncertainty_intervals(x_plot, trace, post_pdf_contribs_nuts, true_density)


# Known Poisson mixture 
      
w = [0.7,0.3]

pois1 = pm.Poisson.dist(10)
pois2 = pm.Poisson.dist(2)

y_pmix = pm.Mixture.dist(w=w, comp_dists= [pois1, pois2])

pmix_obs = []

for i in np.arange(100):
      pmix_obs.append(y_pmix.random())
      
plt.hist(pmix_obs, density=True, bins=100)
plt.plot()

with pm.Model() as p_model:
    alpha = pm.Gamma('alpha',2,2)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    mu = pm.Normal('mu', 0., 25., shape=K)
    obs = pm.Mixture('obs', w, pm.Poisson.dist(mu), observed=pmix_obs)      
      
with p_model:
      
      trace_nuts = pm.sample(tune=1000,draws=1000, chains=2)

# Known Cauchy Mixture 

y_cmix = pm.Mixture.dist()

with pm.Model() as c_model:
    alpha = pm.Gamma('alpha',1,1)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    loc = pm.Normal('mu', 0., 300., shape=K)
    scale = pm.InverseGamma()
    obs = pm.Mixture('obs', w, pm.Cauchy.dist(loc, scale), observed=cmix_obs)      
      

# Known Cauchy Mixture 

     