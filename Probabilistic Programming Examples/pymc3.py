#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:34:42 2019

@author: vidhi
"""

import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt

# Example 1 : Prints the log-likelihood of the model 

data = np.random.randn(100)

with pm.Model() as model:
       
       mu = pm.Normal('mu', mu=0, sd=1)
       obs = pm.Normal('obs', mu=mu, sd=1, observed=data)
       print(model.logp({'mu':0}))
       

# Example 2: Do inference, where we are trying to predict the mean of the normal distribution
       
data = np.random.normal(1,1,100)

with pm.Model() as model:
      
      mu = pm.Normal('mu', mu=0, sd=1) # prior distribution
      obs = pm.Normal('obs', mu=mu, sd=1, observed=data)
      trace = pm.sample(1000, tune=500 ) #discard_tuned_samples=False
      
d=pm.traceplot(trace, combined=True, priors=[pm.Normal.dist(0,1)])
pm.plot_posterior(trace)

