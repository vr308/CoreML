#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:25:32 2019

@author: vidhi

Laplace approximation for arbitrary distributions in 1d and 2d

"""
import pymc3 as pm
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp


density_1 = lambda x_ : 0.4*st.norm.pdf(x_, 2, 1) + 0.6*st.norm.pdf(x_, -1, 1)

# Laplace in pymc3 for arbitary distribution

with pm.Model() as model:
      
    g = pm.Gamma('g', 2, 0.5)
    map_estimate = pm.find_MAP() 
    hessian = pm.find_hessian(map_estimate)
      

x_ = np.linspace(0,20, 1000)
plt.plot(x_, st.gamma.pdf(x_, 2, scale=1/0.5))
plt.plot(x_, st.norm.pdf(x_, map_estimate['g'], 2))
      
# Laplace in pymc3 for a problem with intractable posterior
    
    
    
# Laplace in edward for a problem with intractable posterior