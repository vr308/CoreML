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

home_path = '~/Desktop/Workspace/CoreML/Mixture Models/'
uni_path = '/home/vidhi/Desktop//Workspace/CoreML/Mixture Models/'
      
path = uni_path

data = pd.read_csv('Data/1dDensity.csv', sep=',', names=['x','density'])
data['prob'] = data['density']/ np.sum(data['density'])
data['cdf'] = np.cumsum(data['prob'])

log_data = np.log(data)
log_data['prob'] = log_data['density']/np.sum(log_data['density'])
log_data['cdf'] = np.cumsum(log_data['prob'])


plt.figure()
plt.plot(data['x'], data['prob'], '-')
plt.plot(data['x'], data['density'], '-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log_x')
plt.ylabel('log_density')
plt.title('Raw data')

plt.figure()
plt.plot(np.log(data['x']), np.log(data['density']))
plt.xlabel('log_x')
plt.ylabel('log_density')
plt.yscale('log')

u = np.random.uniform(np.min(log_data['cdf']), 1, (N, ))

#Find the location 
   
samples = []
for i in u:
    loc = np.where(log_data['cdf'] <= i)[0][-1]
    lower_limit = log_data['x'][loc]
    upper_limit = log_data['x'][loc+1]
    samples.append(np.random.uniform(lower_limit, upper_limit))


plt.figure()
plt.hist(samples, bins=500, density=True, log=True, alpha=0.8)
plt.title('Samples generated using inverse CDF', fontsize='small')
plt.xscale('log')

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

cdf = lambda x : 1/normalizer * (h[0]*(x - x[0]) + slope*(x**2/2 - x*x[0] + x[0]**2/2))
cdf_inv = lambda x : np.sqrt((x*normalizer + h[0]**2/(2*slope))/(slope/2)) - h[0]/slope + x[0]

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
