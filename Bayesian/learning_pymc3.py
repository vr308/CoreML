#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:34:42 2019

@author: vidhi
"""

import pymc3 as pm
import numpy as np

x_beta_binomial = np.array([1,1,1,0,0,0,0,0,0,0])

with pm.Model() as beta_binomial_model:
      
      p_beta_binomial = pm.Uniform('p', 0, 1)
      x_obs = pm.Bernoulli('y', p_beta_binomial, observed=x_beta_binomial)
      trace = pm.sample(200)
     
      
 pm.plot_posterior()      