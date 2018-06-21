#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:02:03 2018

@author: vr308
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcessRegressor
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as Ck, WhiteKernel
import GPy as gp
from sklearn import mixture
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == "__main__":
    
    gauss = np.random.normal(0,2, 200)
    ray = np.random.rayleigh(2,100)
    alld = np.sort(np.concatenate([gauss, ray])) 
    
    X = alld.reshape(300,1)
    k = gp.kern.RBF(input_dim=1,lengthscale=1)

    mu = np.zeros((10))
    C = 3*k.K(X[0:10],X[0:10])

    # Generate 20 samples from the multivariate Gaussian with mean mu and covariance C

    Z = np.random.multivariate_normal(mu,C,10)
    
    plt.figure(figsize=(5,5))    
    for i in range(10):
        plt.plot(X[0:10],Z[i,:])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    plt.figure(figsize=(5,5))    
    for i in range(10):
        plt.hist(Z[i,:], bins=50)
    plt.xlabel('x')
    plt.ylabel('Hist of f(x)')
    
    plt.figure()
    plt.hist(X, bins=50)
    
    plt.figure()
    plt.plot(X[0:10], Z[0,:])
    plt.vlines(X[0:10],ymin=-6, ymax=4, lw=1)
    
    
    
    
    # Regressing on points drawn from a pdf
    
    
    kde = KDEUnivariate(X)
    kde.fit(bw=0.2)
    x_grid = np.linspace(-6,6,1000)
    pdf = kde.evaluate(x_grid)
    
    plt.plot(x_grid, pdf)
    plt.plot(x_grid[::100], pdf[::100], 'bo')
    plt.hist(X, bins=50, normed=True)
    
    X = np.atleast_2d(x_grid[::100]).T
    y = np.atleast_2d(pdf[::100]).T
    
    kernel = Ck(1.0, (1e-10, 1e3)) * RBF(2, length_scale_bounds=(2, 5))
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y = True)

    # Fit to data 
    gpr.fit(X, y)       
    
    plt.figure()
    plt.plot(x_grid[::100], pdf[::100], 'bo')
    samples=gpr.sample_y(x_grid.reshape(1000,1), 10)
    for i in np.arange(10):
        print i
        plt.plot(np.atleast_2d(x_grid).T, samples[:,:,i])

  
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)
    
    import itertools

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
        
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')


    plt.plot(X, dpgmm.predict(X))
    
    
    