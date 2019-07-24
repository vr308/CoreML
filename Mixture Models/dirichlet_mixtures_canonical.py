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
      
      
def plot_posterior_densities(trace, post_pdf_contribs, xlabel):
      
      fig, ax = plt.subplots(figsize=(8, 6))
      x_plot = np.linspace(-3, 3, 200)
      n_bins = 20
     
      post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
      post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)
      
      
      ax.hist(old_faithful_df.std_waiting.values, bins=n_bins, normed=True,
              color='b', lw=0, alpha=0.5);
      ax.fill_between(x_plot, post_pdf_low, post_pdf_high,
                      color='gray', alpha=0.45);
      ax.plot(x_plot, post_pdfs[0],
              c='gray', label='Posterior sample densities');
      ax.plot(x_plot, post_pdfs[::2].T, c='gray');
      ax.plot(x_plot, post_pdfs.mean(axis=0),
              c='k', label='Posterior expected density');
      ax.set_xlabel(xlabel);
      ax.set_yticklabels([]);
      ax.set_ylabel('Density');
      ax.legend(loc=2);

def plot_uncertainty_intervals():
      
      return;
      
if__name__== '__main__':
      
# Known Gaussian mixture 

K = 10

w = np.array([0.5,0.3,0.2])
mu = np.array([-5,0,2])
sd = np.array([3, 0.2, 1])

y_gmix =  pm.NormalMixture.dist(w=w, mu=mu, sd=sd)

gmix_obs = []

for i in np.arange(10000):
      gmix_obs.append(y_gmix.random())
      
plt.hist(gmix_obs, density=True, bins=100)

varnames = ['alpha', 'mu', 'sd']

with pm.Model() as g_model:
      
      # Prior over the concentration parameter
      alpha = pm.Gamma('alpha', 2, 2)
      
      # Stick breaking construction for weights 
      beta = pm.Beta('beta', 1, alpha, shape=K)
      w = pm.Deterministic('w', stick_breaking(beta))
      
      # Prior over component parameters mu and sd
      sd = pm.InverseGamma('sd', 1., 1., shape=K)
      mu = pm.Normal('mu', 0, 10, shape=K)
      
      #obs 
      obs = pm.NormalMixture('obs', w, mu, sd=sd,
                           observed=np.array(gmix_obs))
      
with g_model:
      
      trace_nuts = pm.sample(tune=1000,draws=1000, chains=4)
      
with g_model:
      
      fr = pm.FullRankADVI()
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      fr.fit(n=100000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)
    

with model:
      
      nf_planar = pm.NFVI('planar*10',  jitter=0.1)
      nf_planar.fit(25000)
      trace_nf = nf_planar.approx.sample(5000)
      
      
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

y_cmix = 

with pm.Model() as c_model:
    alpha = pm.Gamma('alpha',1,1)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    loc = pm.Normal('mu', 0., 300., shape=K)
    scale = pm.InverseGamma()
    obs = pm.Mixture('obs', w, pm.Cauchy.dist(loc, scale), observed=cmix_obs)      
      

# Known Cauchy Mixture 

     