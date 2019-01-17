#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vidhi

This script uses VI to infer the parameters of a univariate Gaussian
"""
import os

os.chdir('/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/')

import uninorm
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

def plot_posterior(posterior, axes, xlim=None, ylim=None, color='red'):
    if xlim is None or ylim is None:
        lower = posterior.mean() - 3 * posterior.var()
        upper = posterior.mean() + 3 * posterior.var()
        if xlim is None:
            xlim = [lower[0], upper[0]]
        if ylim is None:
            ylim = [lower[1], upper[1]]

    locs = np.linspace(xlim[0], xlim[1])
    scales = np.linspace(ylim[0], ylim[1])
    pp = posterior.pdf(locs, scales)
    axes.contour(locs, scales, pp, colors=color)
    axes.set_xlabel('$\mu$')
    axes.set_ylabel('$\lambda$')
    

def plot_infer_results(pposterior, qposterior_infer_results, maxit=None, xlim=None, ylim=None):
    opt_index = qposterior_infer_results.opt_index()
    ncols = 3
    if maxit is None:
        nrows = int(np.ceil(float(opt_index) / ncols))
        maxit = nrows * ncols + 1
    else:
        nrows = int(np.ceil(float(maxit + 1) / ncols))
         
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows*5))
    axes = axes.flatten()
    for i in range(0, maxit + 1):
        plot_posterior(pposterior, axes[i], xlim, ylim, color='red')
        qposterior = uninorm.QPosterior(qposterior_infer_results[i])
        plot_posterior(qposterior, axes[i], xlim, ylim, color='green')
        axes[i].set_title('Iteration {:d}{:s}'.format(i, '*' if i == opt_index else ''))   
        

def plot_costs(results):
    fig, axes = plt.subplots(figsize=(15, 5))
    line, = axes.plot(range(1, len(results)), results.c[1:])
    line.set_linewidth(2)
    line.set_marker('s')
    line.set_markerfacecolor('red')
    line.set_color('blue')
    axes.set_xlabel('Iteration')
    axes.set_ylabel(r'$-J(Q(\mu,\lambda))$')
    axes.set_xticks(range(1, len(results)))
    axes.grid()
    
# Setting 1

np.random.seed(0)
# 10 points sampled from N(u=0.0, l=1.0)
x = uninorm.simulate(u=0.0, l=1.0, size=10)
# Prior: E[u]=0.0 (weak), E[l]=1.0 (weak)
pprior_params = uninorm.Params(u=0.0, k=1.0, a=0.001, b=0.001)
pposterior_params = uninorm.infer_pposterior(x, pprior_params)
pposterior = uninorm.PPosterior(pposterior_params)
init = uninorm.Params(u=0.0, k=1.0, a=2.0, b=1.0)
qposterior_infer_results = uninorm.infer_qposterior(x, pprior_params, init, maxit=10)
plot_infer_results(pposterior, qposterior_infer_results, maxit=5, xlim=[-0.5, 1.8], ylim=[0.01, 2.5])
plot_costs(qposterior_infer_results)
print(qposterior_infer_results)


# Setting 2    

np.random.seed(0)
x = uninorm.simulate(u=0.0, l=1.0, size=100)
pprior_params = uninorm.Params(u=0.0, k=1.0, a=0.001, b=0.001)
pposterior_params = uninorm.infer_pposterior(x, pprior_params)
pposterior = uninorm.PPosterior(pposterior_params)
init = uninorm.Params(u=0.0, k=1.0, a=2.0, b=1.0)
qposterior_infer_results = uninorm.infer_qposterior(x, pprior_params, init, maxit=10)
plot_infer_results(pposterior, qposterior_infer_results, maxit=5, xlim=[-0.4, 0.4], ylim=[0.2, 1.5])
plot_costs(qposterior_infer_results)
print(qposterior_infer_results)



with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    sd = pm.Gamma('sd', alpha=0.001, beta=0.001)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=x)
    approx_ADVI = pm.fit()
    approx_fullrankADVI = pm.fit(method='fullrank_advi')


pm.traceplot(approx_ADVI.sample(500))
sns.kdeplot(approx_ADVI.sample(500)['mu'], approx_ADVI.sample(500)['sd'])