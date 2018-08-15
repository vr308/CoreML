# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as st 
import sklearn as skt
from scipy.stats import beta
from scipy.stats import dirichlet
import matplotlib.pylab as plt
from statsmodels.nonparametric.kde import KDEUnivariate
from mpl_toolkits.mplot3d import Axes3D

c1 = st.cauchy.rvs(loc=-1,scale=2, size=1000)
c2 = st.cauchy.rvs(loc=4,scale=1.2, size=1000)

x_grid = np.linspace(-4,10, 10000)
pdf1= st.cauchy.pdf(x_grid, -1, 2)
pdf2= st.cauchy.pdf(x_grid, 4, 1.2)

plt.figure()
plt.plot(x_grid, 0.5*pdf1,linestyle='--', color='b', label='components')
plt.plot(x_grid, 0.5*pdf2, linestyle='--', color='b')
plt.plot(x_grid, 0.5*pdf1 + 0.5*pdf2, color='y', label='True dist')
plt.title('Mixture of 1d Cauchy variables')
plt.legend()

# Mixture of 2 Cauchy dist. 
c = np.ravel(np.vstack((c1,c2)))
plt.hist(c, bins=100, density=True, alpha=0.4)

kde = KDEUnivariate(c)
kde.fit(bw=1)

plt.plot(x_grid, kde.evaluate(x_grid))

plt.title('Mixture of Cauchy\'s and KDE estimate')

# Beta distribution 

samples_beta = np.random.beta(7,10, size=10000)

x = np.linspace(0,1,1000)

plt.figure()
plt.hist(samples_beta, bins=100)

alpha=[0.1,0.5,0.7]
beta_=alpha

ab = [(0.5,0.5), (5,1), (1,2), (1,5), (2,4),(1,1)]

means=[]
for i in ab:
    means.append(i[0]/(i[0] + i[1]))
    
for i in ab:
        plt.plot(x,beta.pdf(x, i[0], i[1]), label='(' + str(i[0]) + ',' +  str(i[1]) + ')')
        plt.vlines(x=(i[0]/(i[0] + i[1])), ymin=0, ymax=3,label='Mean')
plt.legend()

#Bayesian updates with beta distribution and coin toss

prior_beta = beta.pdf(x,2,2)

N = 13
m = 9

posterior_beta = beta.pdf(x, 11,6)

plt.figure()
plt.plot(x, prior_beta)
plt.plot(x, posterior_beta)

# Dirichlet
sd = np.random.dirichlet(alpha=(5,5,5), size=10000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sd[:,0],sd[:,1],sd[:,2], c='k', depthshade=False)

# Dirichlet process

import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

N = 20
K = 30

alpha = 2.
P0 = sp.stats.norm

# Sampling using the stick breaking process

beta = sp.stats.beta.rvs(1, 2, size=(N, K))
w = np.empty_like(beta)

w[:,0] = beta[:,0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
omega = P0.rvs(size=(N, K))

plt.hist(omega[2],weights=w[2], bins=100, normed=False)


x_plot = np.linspace(-3, 3, 200)
sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot)).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_plot, sample_cdfs[0], c='gray', alpha=0.75,
        label='DP sample CDFs');
ax.plot(x_plot, sample_cdfs[1:].T, c='gray', alpha=0.75);
ax.plot(x_plot, P0.cdf(x_plot), c='k', label='Base CDF');

ax.set_title(r'$\alpha = {}$'.format(alpha));
ax.legend(loc=2);


# Estimating a pdf using a mixture dirichlet



