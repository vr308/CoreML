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
import theano.tensor as tt
import pymc3 as pm
from sampled import sampled
import seaborn as sns
plt.style.use("ggplot")

#TODO: Investigate path length issue
#TODO: How to extract HMC trajectories 
#TODO: Animate?

# Banana shaped distribution

def pot1(z):
    z = z.T
    a = 1.
    b = 100.
    return (a-z[0])**2 + b*(z[1] - z[0]**2)**2

def cust_logp(z):
    return -pot1(z)

with pm.Model() as pot1m:
    pm.DensityDist('pot1', logp=cust_logp, shape=(2,))
    trace1 = pm.sample(1000, step=pm.NUTS())
    trace2 = pm.sample(1000, step=pm.Metropolis())

plt.plot(trace1['pot1'][:,0], trace1['pot1'][:,1], 'bo', markersize=1)

# Bivariate normals

mean1 = np.array([0,2])
cov1 = np.array([[1, 0.8], [0.8,1]])

mean2 = np.array([3,4])
cov2 = np.array([[1, -0.4], [-0.4,1]])

x1_ = np.linspace(-3, 6, 1000)
x2_ = np.linspace(-5, 10, 1000)
X1, X2 = np.meshgrid(x1_, x2_)

points = np.vstack([X1.ravel(), X2.ravel()]).T

mixture = lambda points : 0.3*st.multivariate_normal.pdf(points, mean1, cov1) + 0.7*st.multivariate_normal.pdf(points, mean2, cov2)

plt.contourf(X1, X2, mixture(points).reshape(1000,1000))

w = (0.3, 0.7)
mu = [mean1, mean2]
cov = [cov1, cov2]

mix_idx = np.random.choice([0,1], size=1000, replace=True, p=w)
x = [st.multivariate_normal.rvs(mean=mu[i], cov=cov[i]) for i in mix_idx]

def normal_pdf(x, mean, cov):
    
      x = x.T
      return 1/np.sqrt(2*np.pi*np.linalg.det(cov))* \
  np.exp((x - mean.T).T.dot(np.linalg.inv(cov).dot(x-mean)))

def bivariate_mixture_pdf(weights=w):
      def logp(x):
            return np.log(w[0]*)
      return logp

@sampled
def bi_mixture(weights=w, **observed):
      pm.DensityDist('bi_mixture', logp=bivariate_mixture_pdf(w), shape=2, testval=[0,1])
      
with bi_mixture(weights=w):
      metropolis_sample = pm.sample(draws=500, step=pm.Metropolis())
      
with bi_sampler:
      
      trace_nuts = pm.sample()
      samples = pm.sample_posterior_predictive(trace_nuts)
      
      step_hmc = pm.HamiltonianMC(path_length=10, adapt_step_size=False, step_scale=0.5)
      trace_hmc = pm.sample(step = step_hmc)
      
   
# Donut distribution example 
      
x_ = np.linspace(-1.3,1.3,1000)
X1, X2 = np.meshgrid(x_, x_)

points = np.vstack([X1.ravel(), X2.ravel()]).T

def tt_donut_pdf(scale):
    def logp(x):
         return -tt.square((1 - x.norm(2)) / scale)
    return logp

def logp(x):
         return -((1 - np.linalg.norm(x, axis=1)) / 0.5)**2

density = lambda x : logp(x)

# The log pdf gives the shape of the distribution

plt.figure(figsize=(6,6))
plt.contourf(X1, X2, density(points).reshape(1000,1000), levels=50)


@sampled 
def donut(scale=0.05, **observed):
    """Gets samples from the donut pdf, and allows adjusting the scale of the donut at sample time."""
    pm.DensityDist('donut', logp=tt_donut_pdf(scale), shape=2, testval=[0, 1])
    
# Why does the HMC not accept a fixed path length

with donut(scale=0.05):
    metropolis_sample = pm.sample(draws=500, step=pm.Metropolis())
    hmc_sample = pm.sample(draws=500, step=pm.HamiltonianMC(path_length=2, adapt_step_size=False))
    nuts_sample = pm.sample(draws=500)
    
y_met = metropolis_sample.get_values('donut')
y_hmc = hmc_sample.get_values('donut')
y_nuts = nuts_sample.get_values('donut')

sns.jointplot(y_met[:,0], y_met[:,1])
sns.jointplot(y_hmc[:,0], y_hmc[:,1])
sns.jointplot(y_nuts[:,0], y_nuts[:,1])