#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:07 2020

@author: vidhi
"""

import numpy as np
import pymc3 as pm
import scipy.stats as st
import matplotlib.pylab as plt
import theano.tensor as tt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel

def generate_gp_latent(X_all, mean, cov, size):

    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval(), size=size)

def generate_gp_training(X_all, f_all, n_train, noise_sd, uniform):

    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         print('in here')
         pdf = 0.5*st.norm.pdf(X_all, 2, 1) + 0.5*st.norm.pdf(X_all, 8, 1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())

    train_index = []
    for i in X:
        train_index.append(X_all.ravel().tolist().index(i))

    X_train = np.empty(shape=(100,n_train))
    y_train = np.empty(shape=(100,n_train))
    for j in np.arange(len(f_all)):
        X = X_all[train_index]
        f = f_all[j][train_index]
        y = f + np.random.normal(0, scale=noise_sd, size=n_train)
        X_train[j] = X.ravel()
        y_train[j] = y.ravel()
    return X_train, y_train

# Simulate datasets

if __name__ == "__main__":

    n_star = 500

    xmin = 0
    xmax = 10

    X_all = np.linspace(xmin, xmax,n_star)[:,None]

    # A mean function that is zero everywhere

    mean = pm.gp.mean.Zero()

    # Kernel Hyperparameters

    sig_sd_true = 5.0
    lengthscale_true = 2.0

    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)

    # This will change the shape of the function

    f_all = generate_gp_latent(X_all, mean, cov, size=100)

    # Data attributes

    noise_sd_true = np.sqrt(1)

    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)

    uniform = False

    # Generating datasets

    X_10, y_10 = generate_gp_training(X_all, f_all, 10, noise_sd_true, uniform)
    X_12, y_12 = generate_gp_training(X_all, f_all, 12, noise_sd_true, uniform)
    X_15, y_15 = generate_gp_training(X_all, f_all, 15, noise_sd_true, uniform)
    X_20, y_20 = generate_gp_training(X_all, f_all, 20, noise_sd_true, uniform)
    X_25, y_25 = generate_gp_training(X_all, f_all, 25, noise_sd_true, uniform)
    X_40, y_40 = generate_gp_training(X_all, f_all, 40, noise_sd_true, uniform)
    X_60, y_60 = generate_gp_training(X_all, f_all, 60, noise_sd_true, uniform)
    X_80, y_80 = generate_gp_training(X_all, f_all, 80, noise_sd_true, uniform)
    X_100, y_100 = generate_gp_training(X_all, f_all, 100, noise_sd_true, uniform)
    X_120, y_120 = generate_gp_training(X_all, f_all, 120, noise_sd_true, uniform)
    X_150, y_150 = generate_gp_training(X_all, f_all, 150, noise_sd_true, uniform)

    X = [X_10, X_12, X_15, X_20, X_25, X_40, X_60, X_80, X_100, X_120, X_150]
    y = [y_10, y_12, y_15, y_20, y_25, y_40, y_60, y_80, y_100, y_120, X_150]

    seq = [10,12,15,20,25, 40,60, 80, 100, 120,150]

    nd = len(seq)

    # Sanity check data
    plt.figure()
    plt.plot(X_all, f_all[10])
    plt.plot(X_150[10], y_150[10], 'bo')

    # 100 datasets of each of the 12 sizes

    s = np.empty(shape=(nd,100))
    ls = np.empty(shape=(nd,100))

    # sklearn - learning - unconstrained optim and fixed noise

    for j in np.arange(nd):

        print('Analysing data-sets of size ' + str(np.shape(X[j])[1]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=1, noise_level_bounds="fixed")

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0).fit(X[j][i][:,None], y[j][i])

            ls[j][i] = gp.kernel_.k1.k2.length_scale
            s[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)

     # sklearn - learning - unconstrained optim and noise not fixed

    ls_noise =  np.empty(shape=(nd,100))
    s_noise =  np.empty(shape=(nd,100))
    noise =  np.empty(shape=(nd,100))

    for j in np.arange(nd):

        print('Analysing data-sets of size ' + str(np.shape(X[j])[1]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=1)

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0).fit(X[j][i][:,None], y[j][i])

            ls_noise[j][i] = gp.kernel_.k1.k2.length_scale
            s_noise[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)
            noise[j][i] = np.sqrt(gp.kernel_.k2.noise_level)


    # Plotting
    ls_mean_nf = np.mean(ls, axis=1)
    ls_mean_n = np.mean(ls_noise, axis=1)
    plt.plot(seq, np.log(ls_mean_nf), 'bo-', label='Noise level fixed')
    plt.plot(seq, np.log(ls_mean_n), 'go-', label='Noise level estimated')
    plt.axhline(y=np.log(lengthscale_true), color='r', label='True lengthscale')
    plt.title("Learning the lenghtscale under ML-II" + '\n' + 'Avg. across 100 datasets per training set size', fontsize='small')
    plt.legend(fontsize='small')
    plt.xlabel('Training set size', fontsize='small')
    plt.ylabel('Avg. estimated log-lengthscale', fontsize='small')

    noise_means = np.mean(noise, axis=1)
    noise_std = np.std(noise, axis=1)
    plt.figure()
    plt.plot(seq, noise_means, 'o-', label='Noise level')
    #plt.errorbar(seq, noise_means, yerr=noise_std)
    plt.title("Learning the noise level under ML-II" + '\n' + 'Avg. across 100 datasets per training set size', fontsize='small')
    plt.legend(fontsize='small')
    plt.axhline(y=noise_sd_true, color='r', label='True lengthscale')
    plt.xlabel('Training set size', fontsize='small')
    plt.ylabel('Avg. estimated noise std', fontsize='small')

