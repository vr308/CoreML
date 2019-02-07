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
import scipy 

N=100000

# Chris data 

data = pd.read_csv('Data/1dDensity.csv', sep=',', names=['x','density'])

log_data = np.log(data)
probabilities = log_data['density']/np.sum(log_data['density'])
cprob = np.cumsum(probabilities)
x = log_data['x']
log_data['prob'] = probabilities
log_data['cdf'] = cprob

#q = lambda x: stats.cauchy.pdf(x,-1, 10)
#M = 0.05

plt.figure()
plt.plot(log_data['x'], probabilities, '-')
plt.xlabel('log_x')
plt.ylabel('log_density')
plt.title('Raw data')
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
plt.hist(samples, bins=1000, density=True)
plt.plot(x, probabilities*2000, 'bo', markersize=2)
plt.plot(x, probabilities*2000, 'r-', markersize=0.2)
#plt.vlines(x, ymin=0, ymax=probabilities*2000, alpha=0.2)
#plt.bar(x, probabilities*2000, align='edge', alpha=0.2)

# Given 2 points of a discrete distribution, form a continuous pdf connecting the two points

point_A = (log_data['x'][19], log_data['density'][19])
point_B = (log_data['x'][30], log_data['density'][30])

x = [point_A[0], point_B[0]]
h = [point_A[1], point_B[1]]

def slope_(point_A, point_B):
      
      x1 = point_A[0]
      h1 = point_A[1]
      x2 = point_B[0]
      h2 = point_B[1]
      slope = (h2 - h1)/(x2 - x1)
      return slope

slope = slope_(point_A, point_B) 

normalizer = 0.5*(point_A[1] + point_B[1])*(point_B[0] - point_A[0])

func = lambda x : x*slope + point_A[1] - slope*point_A[0]
pdf = lambda x : (x*slope + point_A[1] - slope*point_A[0])/normalizer

# Check if the pdf is valid 
scipy.integrate.quad(pdf, x[0], x[1])

cdf = lambda x : ((h[0]*x[1] - h[1]*x[0])/(normalizer*(x[1] - x[0])))*((h[1] - h[0])/2)*(x**2 - x[0]*x[0])
cdf = lambda x : (slope/normalizer)*(x**2/2 - x*x[0] + h[0]*(x - x[0]) + x[0]**2/2)
cdf = lambda x : 1/normalizer * (h[0]*(x - x[0]) + slope*(x**2/2 - x*x[0] + x[0]**2/2))


x_range = np.arange(point_A[0],point_B[0],0.01)
y_range = func(x_range)
y_pdf = pdf(x_range)
y_cdf = cdf(x_range)

plt.figure()
plt.plot(point_A[0], point_A[1], 'bo')
plt.plot(point_B[0], point_B[1], 'bo')
plt.plot(x, h, 'r-')
plt.plot(x_range, y_range)
plt.plot(x_range, y_pdf)
plt.plot(x_range, y_cdf)


      















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
