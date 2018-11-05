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

def plot_results(data, x_,true_density, estimator, model):
    
    weights = estimator.weights_
    means = estimator.means_
    covariances = estimator.covariances_
    components = []
    for counter, value in enumerate(zip(means, covariances)):
            components.append(weights[counter]*st.norm.pdf(x_, value[0], np.sqrt(value[1])).reshape(1000,))     
    estimated_density = np.sum(components, axis=0)
    plt.figure()
    plt.plot(x_,true_density, 'k', alpha=0.2, label='True Density')
    plt.plot(x_, estimated_density, alpha=0.7, label='Estimated Density')
    plt.plot(data, [0]*len(data), 'kx', markersize=2, label='Observations')
    plt.title(model, fontsize='small')
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

plot_results(data, x_, mixture, estimator_gmm, 'GMM, Inference: EM')
plot_results(data, x_, mixture, estimator_bgmm, 'Bayesian GMM')
plot_results(data, x_, mixture, estimator_dpgmm, 'Dirichlet Process prior GMM')


# Sample posterior densities from estimated mixtures - in the sklearn framework 
# Construct a density from a histogram and sample from it
# Fit DPGMM and BGM to the cms data.

# 2d example from sklearn website


import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()