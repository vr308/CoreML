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

x = st.lognorm.rvs(loc=0, scale=1.0, s=1, size=1000)
y = st.lognorm.rvs(loc=1.0, scale=2.0, s=1, size=1000)
sns.kdeplot(data= x, data2=y, shade=True, shade_lowest=False)


with pm.Model() as model:
  
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=np.random.randn(100))
    approx_ADVI = pm.fit()
    approx_fullrankADVI = pm.fit(method='fullrank_advi')



