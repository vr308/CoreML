#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:52:19 2017

@author: vr308
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymc3 as pm
np.random.seed(123)

# Analytical posterior - conjugate Gaussian prior case

#data = np.random.randn(20)
data = np.random.normal(4,2,100)

def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

plt.figure
x = np.linspace(2.5, 5, 500)
posterior_analytical = calc_posterior_analytical(data, x, 4.0, 2)
plt.plot(x, posterior_analytical)
plt.xlabel('mu')
plt.ylabel('belief') 
plt.title('Analytical posterior')

# MCMC part

def sampler(data, samples=4, mu_init=0.5, proposal_width=0.5, mu_prior_mu =0, mu_prior_sd=1.0):
    
    mu_current = mu_init
    posterior = [0.1]
    for i in range(samples):
        
        # suggest a new position
        mu_proposal = norm(mu_current, proposal_width).rvs()
    
        likelihood_current = norm(mu_current, 1).pdf(data).prod()
        likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()
    
        # Compute prior probability of current and proposed mu        
        prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
        prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
    
        # Nominator of Bayes formula
        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal
    
        p_accept = p_proposal / p_current
    
        accept = np.random.rand() < p_accept
    
        if accept:
            # Update position
            mu_current = mu_proposal
        
        posterior.append(mu_current)
    
    return posterior


pos = sampler(data, 1000, 0, 1)
plt.hist(pos, bins=100, normed=True)

# With Pymc3

with pm.Model():
    mu = pm.Normal('mu', 0, 1)
    sigma = 1.
    returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
    
    step = pm.Metropolis()
    trace = pm.sample(15000, step)
    
sns.distplot(trace[2000:]['mu'], label='PyMC3 sampler');
sns.distplot(posterior[500:], label='Hand-written sampler');
plt.legend();