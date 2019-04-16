#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:51:46 2019

@author: vidhi

Building confidence intervals for gaussian mixtures 
 
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

# Generating some random variates

m1 = st.norm.rvs(-2, 1, 500)
m2 = st.norm.rvs(0.5,1.4,500)
m3 = st.norm.rvs(4,0.9,500)

data = np.concatenate((m1,m2,m3)).reshape(-1,1)

# Creating a mixture

mu = [-2, 0.5, 4]
sd = [1, 1.4, 0.9]

x_ = np.linspace(-6,12,1000)

d1 = st.norm.pdf(x_, mu[0], sd[0])
d2 = st.norm.pdf(x_, mu[1], sd[1])
d3 = st.norm.pdf(x_, mu[2], sd[2])

w = [0.4, 0.2, 0.4]

# Sampling from the mixture

mix_idx = np.random.choice([0,1,2], size=10000, replace=True, p=w)
x = np.array([st.norm.rvs(loc=mu[i], scale=sd[i]) for i in mix_idx])

mixture = w[0]*d1 + w[1]*d2 + w[2]*d3

mix_f = lambda x : w[0]*st.norm.pdf(x, mu[0], sd[0]) + w[1]*st.norm.pdf(x, mu[1], sd[1]) + w[2]*st.norm.pdf(x, mu[2], sd[2])

plt.figure()
plt.plot(x_, 0.4*d1, 'b-', markersize=1)
plt.plot(x_, 0.2*d2, 'g-', markersize=1)
plt.plot(x_, 0.4*d3, 'r-', markersize=1)
plt.plot(x_, mixture, '--')
plt.hist(x, bins=200, normed=True)

p1, p2 = st.scoreatpercentile(x, per=[2.5,97.5])
plt.vlines(st.scoreatpercentile(x, per=[2.5,97.5]), ymin=0, ymax=max(mixture))

# 95 % of pmass is within this range

sp.integrate.quad(mix_f, p1, p2)