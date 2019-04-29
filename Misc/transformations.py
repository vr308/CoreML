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

# VI to fit a bivariate distribution of Gaussian / Non-Gaussian

mean = [0,0]
cov = np.array([[1, 0.8],[0.8,1]])

x = st.multivariate_normal.rvs(mean, cov, 1000)
sns.kdeplot(x, shade=True)

with pm.Model() as model:
  
    #mu = pm.MvNormal('mu', mu=[0,0], cov=cov, shape=2)
    
    obs = pm.MvNormal('obs', mu=mu, cov=cov, shape=2, dtype=theano.config.floatX)
    
    approx_ADVI = pm.fit(method='advi')
    approx_fullrankADVI = pm.fit(method='fullrank_advi')


samples_mf = st.multivariate_normal.rvs(mean=approx_ADVI.mean.eval(), cov=approx_ADVI.cov.eval(),size=1000)
samples_fr = st.multivariate_normal.rvs(mean=approx_fullrankADVI.mean.eval(), cov=approx_fullrankADVI.cov.eval(),size=1000)


pm.traceplot(trace_mf)

sns.kdeplot(x[:,0], x[:,1],shade=True)
sns.kdeplot(samples_mf[:,0], samples_mf[:,1], shade=True, shade_lowest=False)
sns.kdeplot(samples_fr[:,0], samples_fr[:,1], shade=True, shade_lowest=False)


plt.figure()
sns.kdeplot(trace_fr['mu'][:,0], trace_fr['mu'][:,1], shade=True, shade_lowest=False)


mu = pm.floatX([0., 0.])
cov = pm.floatX([[1, .5], [.5, 1.]])
with pm.Model() as model:
    pm.MvNormal('x', mu=mu, cov=cov, shape=2)
    trace_hmc = pm.sample(1000)
    
    approx_ADVI = pm.fit(method='advi')
    approx_fullrankADVI = pm.fit(method='fullrank_advi')
    
    trace_mf = approx_ADVI.sample(2000)
    trace_fr = approx_fullrankADVI.sample(2000)
    

sns.kdeplot(x[:,0], x[:,1],shade=True, shade_lowest=False)
sns.scatterplot(trace_mf['x'][:,0], trace_mf['x'][:,1], kwargs={'s':0.5})
sns.scatterplot(trace_fr['x'][:,0], trace_fr['x'][:,1], size=0.5)
sns.kdeplot(trace_hmc['x'][:,0], trace_hmc['x'][:,1], shade=True, shade_lowest=False, alpha=0.5)


