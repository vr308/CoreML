#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:01:09 2019

@author: vidhi

Script to test the difference between HMC and NUTS on simple examples

"""

import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import scipy as sp
from matplotlib import cm
import seaborn as sns
import pymc3 as pm
plt.style.use("ggplot")

# Bivariate normals

mean1 = [0,2]
cov1 = [[1, 0.8], [0.8,1]]

mean2 = [3,4]
cov2 = [[1, -0.4], [-0.4,1]]

X1, X2 = np.meshgrid(x1_, x2_)

points = np.vstack([X1.ravel(), X2.ravel()]).T

mixture = lambda points : 0.3*st.multivariate_normal.pdf(points, mean1, cov1) + 0.7*st.multivariate_normal.pdf(points, mean2, cov2)

plt.contourf(X1, X2, mixture(points).reshape(1000,1000))


w = [0.3, 0.7]
mu = [mean1, mean2]
cov = [cov1, cov2]

mix_idx = np.random.choice([0,1], size=1000, replace=True, p=w)
x = [st.multivariate_normal.rvs(mean=mu[i], cov=cov[i]) for i in mix_idx]

with pm.Model() as bi_sampler:
      
      #comp1 = pm.MvNormal('comp1', mu=np.array(mean1), cov=np.array(cov1), shape=2)
      #comp2 = pm.MvNormal('comp2', mu=np.array(mean2), cov=np.array(cov2), shape=2)
      x = pm.Mixture('x', w=w, comp_dists=[comp1.distribution,comp2.distribution])
      
with bi_sampler:
      
      trace_nuts = pm.sample()
      samples = pm.sample_posterior_predictive(trace_nuts)
      
      step_hmc = pm.HamiltonianMC(path_length=10, adapt_step_size=False, step_scale=0.5)
      trace_hmc = pm.sample(step = step_hmc)

