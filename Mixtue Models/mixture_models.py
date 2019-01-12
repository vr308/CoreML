#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:46:37 2018

@author:  vr308
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

# Generating some random variates

m1 = st.norm.rvs(-2, 1, 50)
m2 = st.norm.rvs(0.5,1.4,50)
m3 = st.norm.rvs(4,0.9,50)

data = np.concatenate((m1,m2,m3)).reshape(-1,1)

# Creating a mixture

x_ = np.linspace(-6,12,1000)

d1 = st.norm.pdf(x_,-2,1)
d2 = st.norm.pdf(x_,0.5,1.4)
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

def plot_results(data, x_, estimator, model_name):
    
    weights = estimator.weights_
    means = estimator.means_
    covariances = estimator.covariances_
    components = []
    for counter, value in enumerate(zip(means, covariances)):
            components.append(weights[counter]*st.norm.pdf(x_, value[0], np.sqrt(value[1])).reshape(1000,))     
    estimated_density = np.sum(components, axis=0)
    plt.figure()
    #plt.plot(x_,true_density, 'k', alpha=0.2, label='True Density')
    plt.plot(x_, estimated_density, alpha=0.7, label='Estimated Density')
    #plt.plot(data, [0]*len(data), 'kx', markersize=2, label='Observations')
    plt.title(model_name, fontsize='small')
    plt.hist(data, bins=400, density=True)
    plt.legend(fontsize='small')
    
# Gaussian non-Bayesian mixture
estimator_gmm = GaussianMixture(n_components=5, 
                                covariance_type='full',
                                  verbose=1, n_init=5).fit(data)
                                
# Finite Bayesian mixture

estimator_bgmm = BayesianGaussianMixture(n_components=5, 
                                    covariance_type='full', 
                                    weight_concentration_prior_type='dirichlet_distribution',
                                    weight_concentration_prior=None,
                                    verbose=1).fit(data)

# Infinite Bayesian mixture 

estimator_dpgmm = BayesianGaussianMixture(n_components=4, 
                                    covariance_type='full', 
                                    weight_concentration_prior_type='dirichlet_process',
                                    weight_concentration_prior=10,
                                    verbose=1).fit(data)

plot_results(data, x_, estimator_gmm, 'GMM, Inference: EM')
plot_results(data, x_, estimator_bgmm, 'Bayesian GMM')
plot_results(data, x_, estimator_dpgmm, 'Dirichlet Process prior GMM')



