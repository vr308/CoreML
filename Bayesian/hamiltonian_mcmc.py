#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vidhi
"""

import numpy as np
import scipy.stats as st
from scipy.spatial import distance
import matplotlib.pylab as plt
import sklearn.metrics as metrics
import pyhmc as hmc
import seaborn as sns

# Generating some random variates

m1 = st.norm.rvs(-2, 1, 50)
m2 = st.norm.rvs(0.5, 1.4, 50)
m3 = st.norm.rvs(4, 0.9, 50)

data = np.concatenate((m1, m2, m3)).reshape(-1, 1)

# Creating a mixture

x_ = np.linspace(-6, 12, 1000)

d1 = st.norm.pdf(x_, -2, 1)
d2 = st.norm.pdf(x_, 0.5, 1.4)
d3 = st.norm.pdf(x_, 4, 0.7)

mixture = 0.4*d1 + 0.2*d2 + 0.4*d3

plt.figure()
plt.plot(x_, 0.4*d1, 'b-', markersize=1)
plt.plot(x_, 0.2*d2, 'g-', markersize=1)
plt.plot(x_, 0.4*d3, 'r-', markersize=1)
plt.plot(x_, mixture, '--')
plt.plot(m1, [0]*len(m1), 'kx', markersize=1)
plt.plot(m2, [0]*len(m2), 'kx', markersize=1)
plt.plot(m3, [0]*len(m3), 'kx', markersize=1) 

# Concentration of measure

sample_1d = np.random.normal(0,1,1000)
sample_2d = np.random.multivariate_normal([0, 0], np.eye(2), 1000)
sample_5d = np.random.multivariate_normal([0]*5, np.eye(5), 1000)
sample_10d = np.random.multivariate_normal([0]*10, np.eye(10), 1000)

radius_2d = np.mean(sample_2d, axis=1)
radius_5d = np.mean(sample_5d, axis=1)
radius_10d = np.mean(sample_10d, axis=1)

plt.figure()
plt.hist(sample_1d, bins=100, alpha=0.4)
plt.hist(radius_2d, bins=100, alpha=0.4)
plt.hist(radius_5d, bins=100, alpha=0.4)
plt.hist(radius_10d, bins=100, alpha=0.4)

for N in (2, 4, 8, 16, 32):
    x = st.multivariate_normal(cov=np.eye(N)).rvs(size=10000)
    _ = sns.distplot(np.linalg.norm(x, axis=1))



