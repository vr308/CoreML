#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vidhi

# Sample posterior densities from estimated mixtures - in the sklearn framework 
# Construct a density from a histogram and sample from it
# Fit DPGMM and BGM to the cms data.

"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import pandas as pd

N=100000

# Chris data 

data = pd.read_csv('1dDensity.csv', sep=',', names=['x','density'])

log_data = np.log(data)
probabilities = log_data['density']/np.sum(log_data['density'])
cprob = np.cumsum(probabilities)
x = log_data['x']
log_data['prob'] = probabilities
log_data['cdf'] = cprob

#q = lambda x: stats.cauchy.pdf(x,-1, 10)
#M = 0.05

plt.figure()
plt.plot(log_data['x'], 200*probabilities, '-')
#plt.plot(log_data['x'], M*q(log_data['x']))

#x_samples = np.random.choice(x,N)

u = np.random.uniform(np.min(cprob), 1, (N, ))

#Find the location 

samples = []
for i in u:
    loc = np.where(log_data['cdf'] <= i)[0][-1]
    samples.append(log_data['x'][loc])

#samples = pd.Series([(x_samples[i]) for i in range(N) if u[i] < probabilities[np.where(x == x_samples[i])[0][0]] / (M * q(x_samples[i]))])

plt.figure()
#plt.hist(samples, bins=100, density=True)
plt.plot(x, probabilities*2000, 'bo', markersize=2)
plt.plot(x, probabilities*2000, 'r-', markersize=0.2)
plt.vlines(x, ymin=0, ymax=probabilities*2000, alpha=0.2)
#plt.bar(x, probabilities*2000, align='edge', alpha=0.2)

#plt.title('Sampling from a discrete distribution - Rejection sampling')
#
#X = np.asarray(samples).reshape(-1,1)

#GMM

estimator_gmm = mixture.GaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)


#BGMM

estimator_bgmm = BayesianGaussianMixture(n_components=5, 
                                    covariance_type='full', 
                                    weight_concentration_prior_type='dirichlet_distribution',
                                    weight_concentration_prior=None,
                                    verbose=1).fit(X)

# Infinite Bayesian mixture 

estimator_dpgmm = BayesianGaussianMixture(n_components=4, 
                                    covariance_type='full', 
                                    weight_concentration_prior_type='dirichlet_process',
                                    weight_concentration_prior=10,
                                    verbose=1).fit(X)

x_ = np.linspace(-1,5,1000)
plot_results(X, x_, estimator_gmm, 'GMM, Inference: EM')
plot_results(X, x_, estimator_bgmm, 'Bayesian GMM')
plot_results(X,x_, estimator_dpgmm, 'Dirichlet process mixture model')
