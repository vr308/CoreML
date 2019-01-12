#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Estimating a pdf using a mixture dirichlet

Created on Tue Oct 23 17:15:13 2018

@author:  vr308
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


#Gaussian mixture 

old_faithful_df = pd.read_csv(pm.get_data('old_faithful.csv'))
old_faithful_df['std_waiting'] = (old_faithful_df.waiting - old_faithful_df.waiting.mean()) / old_faithful_df.waiting.std()

K = 30
N = old_faithful_df.shape[0]

with pm.Model() as model:
    alpha = pm.Gamma('alpha',1,1)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    
    tau = pm.Gamma('tau', 1., 1., shape=K)
    #lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=tau,
                           observed=old_faithful_df.std_waiting.values)
with model:
    trace = pm.sample(500)
    
fig, ax = plt.subplots(figsize=(8, 6))
x_plot = np.linspace(-3, 3, 200)
n_bins = 20

post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x_plot),
                                      trace['mu'][:, np.newaxis, :],
                                      1. / np.sqrt(trace['tau'])[:, np.newaxis, :])
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
ax.set_xlabel('Standardized waiting time between eruptions');
ax.set_yticklabels([]);
ax.set_ylabel('Density');
ax.legend(loc=2);

#Plotting the components

fig, ax = plt.subplots(figsize=(8, 6))

n_bins = 20
ax.hist(old_faithful_df.std_waiting.values, bins=n_bins, normed=True,
        color='blue', lw=0, alpha=0.5);

ax.plot(x_plot, post_pdfs.mean(axis=0),
        c='k', label='Posterior expected density');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0)[:, 0],
        '--', c='k', label='Posterior expected mixture\ncomponents\n(weighted)');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0),
        '--', c='k');

ax.set_xlabel('Standardized waiting time between eruptions');

ax.set_yticklabels([]);
ax.set_ylabel('Density');

ax.legend(loc=2);


#Poisson mixture 

sunspot_df = pd.read_csv(pm.get_data('sunspot.csv'), sep=';', names=['time', 'sunspot.year'], usecols=[0, 1])

K = 50
N = sunspot_df.shape[0]

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

with pm.Model() as model:
    alpha = pm.Gamma('alpha',1,1)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    mu = pm.Uniform('mu', 0., 300., shape=K)
    obs = pm.Mixture('obs', w, pm.Poisson.dist(mu), observed=sunspot_df['sunspot.year'])
    
with model:
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step)
    
x_plot = np.arange(250)
post_pmf_contribs = sp.stats.poisson.pmf(np.atleast_3d(x_plot),
                                         trace['mu'][:, np.newaxis, :])
post_pmfs = (trace['w'][:, np.newaxis, :] * post_pmf_contribs).sum(axis=-1)
post_pmf_low, post_pmf_high = np.percentile(post_pmfs, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(sunspot_df['sunspot.year'].values, bins=40, normed=True, lw=0, alpha=0.75);
ax.fill_between(x_plot, post_pmf_low, post_pmf_high,
                 color='gray', alpha=0.45)
ax.plot(x_plot, post_pmfs[0],
        c='gray', label='Posterior sample densities');
ax.plot(x_plot, post_pmfs[::200].T, c='gray');
ax.plot(x_plot, post_pmfs.mean(axis=0),
        c='k', label='Posterior expected density');
ax.set_xlabel('Yearly sunspot count');
ax.set_yticklabels([]);
ax.legend(loc=1);