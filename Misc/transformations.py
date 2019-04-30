#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:48:13 2019

@author: vidhi

Learning how probability transforms work

"""

import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import pymc3 as pm
import seaborn as sns

x = np.linspace(0, 10, 1000)
x_rv = st.norm.rvs(4,1, 1000)
y_rv = np.log(x_rv)
z_rv = np.exp(x_rv) # Z has a log-normal distribution 

# pdf of x
plt.figure()
plt.hist(x_rv, 100, normed=True)
plt.plot(x, st.norm.pdf(x, 4, 1))

# pdf of log(x)
plt.figure()
plt.hist(y_rv, 100, normed=True)
plt.plot(x, st.norm.pdf(np.exp(x), 4, 1)*np.exp(x))

# pdf of e^x

x = np.linspace(0,1500, 10000)
plt.figure()
plt.hist(z_rv, 100, normed=True)
plt.plot(x, st.norm.pdf(np.log(x),4, 1)/x)
#plt.plot(x, st.lognorm.pdf(x, loc=4, s=1, scale=np.exp(4)))

# MFVB and FR on a correlated Gaussian

mu = pm.floatX([0., 0.])
cov = pm.floatX([[1, 0.9], [0.9, 1.]])

x = st.multivariate_normal.rvs(mu, cov, 1000)

with pm.Model() as model:
    
    pm.MvNormal('x', mu=mu, cov=cov, shape=2)
    trace_hmc = pm.sample(1000)
    
    approx_ADVI = pm.fit(method='advi')
    approx_fullrankADVI = pm.fit(method='fullrank_advi')
    
    trace_mf = approx_ADVI.sample(2000)
    trace_fr = approx_fullrankADVI.sample(2000)
    

sns.kdeplot(x[:,0], x[:,1],shade=True, shade_lowest=False)
plt.axvline(x=0)
plt.axhline(y=0)
sns.kdeplot(trace_mf['x'][:,0], trace_mf['x'][:,1], shade=True, shade_lowest=False)
sns.kdeplot(trace_fr['x'][:,0], trace_fr['x'][:,1], shade=True, shade_lowest=False, alpha=0.5)
sns.kdeplot(trace_hmc['x'][:,0], trace_hmc['x'][:,1], shade=True, shade_lowest=False, alpha=0.5)

# Transformed Gaussian

import theano.tensor as tt

def pot1(z):
    z = z.T
    return .5*((z.norm(2, axis=0)-2.)/.4)**2 - tt.log(tt.exp(-.5*((z[0]-2.)/.6)**2) +
                                                      tt.exp(-.5*((z[0]+2.)/.6)**2))
    
def cust_logp(z):
    #return bound(-pot1(z), z>-5, z<5)
    return -pot1(z)

with pm.Model() as pot1m:
    pm.DensityDist('pot1', logp=cust_logp, shape=(2,))

with pot1m:
        
    trace_hmc = pm.sample(1000)
    
    approx_ADVI = pm.fit(method='advi')
    approx_fullrankADVI = pm.fit(method='fullrank_advi')
    
    trace_mf = approx_ADVI.sample(2000)
    trace_fr = approx_fullrankADVI.sample(2000)
    
sns.kdeplot(trace_mf['pot1'][:,0], trace_mf['pot1'][:,1], shade=True, shade_lowest=False)
sns.kdeplot(trace_fr['x'][:,0], trace_fr['x'][:,1], shade=True, shade_lowest=False, alpha=0.5)
sns.kdeplot(trace_hmc['x'][:,0], trace_hmc['x'][:,1], shade=True, shade_lowest=False, alpha=0.5)

