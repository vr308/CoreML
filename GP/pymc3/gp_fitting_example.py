#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:53:11 2020

@author: vidhi
"""

# Full Bayesian inference for sparse and dense datasets

import numpy as np
import pymc3 as pm
import scipy.stats as st
import pandas as pd
import matplotlib.pylab as plt
from sampled import sampled
import theano.tensor as tt
from theano.tensor.nlinalg import matrix_inverse
import csv
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
import posterior_analysis as pa

def generate_gp_latent(X_all, mean, cov):

    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval())

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

    #size = len(f_all)

    #X_train = np.empty(shape=(size,n_train))
    #y_train = np.empty(shape=(size,n_train))
    X = X_all[train_index]
    f = f_all[train_index]
    y = f + np.random.normal(0, scale=noise_sd, size=n_train)
    return X, y

def generate_gp_test(X_all, f_all, X):

    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    print(len(test_index))
    return X_all[test_index], f_all[test_index]

def get_kernel_matrix_blocks(X, X_star, n_train, point):

          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K = cov(X)
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          K_noise = K + np.square(point['noise_sd'])*tt.eye(n_train)
          K_inv = matrix_inverse(K_noise)
          return K, K_s, K_ss, K_noise, K_inv


def analytical_gp_predict_noise(y, K, K_s, K_ss, K_noise, K_inv, noise_var):

          L = np.linalg.cholesky(K_noise.eval())
          alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
          v = np.linalg.solve(L, K_s.eval())
          post_mean = np.dot(K_s.eval().T, alpha)
          post_cov = K_ss.eval() - v.T.dot(v)
          #post_std = np.sqrt(np.diag(post_cov))
          post_std_y = np.sqrt(np.diag(post_cov) + noise_var)
          return post_mean,  post_std_y


def compute_log_marginal_likelihood(K_noise, y):

      return np.log(st.multivariate_normal.pdf(y, cov=K_noise.eval()))


def write_posterior_predictive_samples(trace, thin_factor, X, y, X_star, path, method):

      means_file = path + 'means_' + method + '_' + str(len(X)) + '.csv'
      std_file = path + 'std_' + method + '_' + str(len(X)) + '.csv'
      trace_file = path + 'trace_' + method + '_' + str(len(X)) + '.csv'

      varnames = ['sig_sd', 'ls', 'noise_sd']
      means_writer = csv.writer(open(means_file, 'w'))
      std_writer = csv.writer(open(std_file, 'w'))
      trace_writer = csv.writer(open(trace_file, 'w'))

      means_writer.writerow(X_star.flatten())
      std_writer.writerow(X_star.flatten())
      trace_writer.writerow(varnames + ['lml'])

      for i in np.arange(len(trace))[::thin_factor]:

            print('Predicting ' + str(i))
            K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), trace[i])
            #post_mean, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
            post_mean, post_std = analytical_gp_predict_noise(y, K, K_s, K_ss, K_noise, K_inv, np.square(trace[i]['noise_sd']))
            marginal_likelihood = compute_log_marginal_likelihood(K_noise, y)
            #mu, var = pm.gp.Marginal.predict(Xnew=X_star, point=trace[i], pred_noise=False, diag=True)
            #std = np.sqrt(var)
            list_point = [trace[i]['sig_sd'], trace[i]['ls'], trace[i]['noise_sd'], marginal_likelihood]

            print('Writing out ' + str(i) + ' predictions')
            means_writer.writerow(np.round(post_mean, 3))
            std_writer.writerow(np.round(post_std, 3))
            trace_writer.writerow(np.round(list_point, 3))



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

    f_all = generate_gp_latent(X_all, mean, cov)

    # Data attributes

    noise_sd_true = np.sqrt(5)

    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)

    uniform = True

    # Generating datasets

    X_20, y_20 = generate_gp_training(X_all, f_all, 20, noise_sd_true, uniform)
    X_star_20, f_star_20 = generate_gp_test(X_all, f_all, X_20)

    X_10, y_10 = generate_gp_training(X_all, f_all, 10, noise_sd_true, uniform)
    X_star_10, f_star_10 = generate_gp_test(X_all, f_all, X_10)

    #Plot

    plt.plot(X_all, f_all)
    plt.plot(X_10, y_10,'ko')

    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10).fit(X_10, y_10)

    ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
    post_mean, post_cov = gpr.predict(X_star_10, return_cov = True) # sklearn always predicts with noise
    post_std = np.sqrt(np.diag(post_cov))

    @sampled
    def generative_model(X, y):

       # prior on lengthscale
       log_ls = pm.Normal('log_ls',mu = 0, sd = 2)
       ls = pm.Deterministic('ls', tt.exp(log_ls))

        #prior on noise variance
       log_n = pm.Normal('log_n', mu = 0 , sd = 2)
       noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))

       #prior on signal variance
       log_s = pm.Normal('log_s', mu=0, sd = 2)
       sig_sd = pm.Deterministic('sig_sd', tt.exp(log_s))
       #sig_sd = pm.InverseGamma('sig_sd', 4, 4)

       # Specify the covariance function.
       cov_func = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)

       gp = pm.gp.Marginal(cov_func=cov_func)

       #Prior
       trace_prior = pm.sample(draws=1000)

       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)

#-----------------------------Full Bayesian HMC---------------------------------------

    # Generating traces with NUTS

    with generative_model(X=X_20, y=y_20): trace_hmc_20 = pm.sample(draws=500, tune=500)
    with generative_model(X=X_10, y=y_10): trace_hmc_10 = pm.sample(draws=500, tune=500)

    path = '/home/vidhi/Desktop/Workspace/CoreML/GP/pymc3/'
    write_posterior_predictive_samples(trace_hmc_20, 5, X_20, y_20, X_star_20, path, 'hmc')
    write_posterior_predictive_samples(trace_hmc_10, 5, X_10, y_10, X_star_10, path, 'hmc')

    means_hmc_20 = pd.read_csv(path + 'means_hmc_20.csv')
    std_hmc_20 = pd.read_csv(path + 'std_hmc_20.csv')
    lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(means_hmc_20, std_hmc_20)

    pp_mean_hmc_20 = np.mean(means_hmc_20)

    means_hmc_10 = pd.read_csv(path + 'means_hmc_10.csv')
    std_hmc_10 = pd.read_csv(path + 'std_hmc_10.csv')
    lower_hmc_10, upper_hmc_10 = pa.get_posterior_predictive_uncertainty_intervals(means_hmc_10, std_hmc_10)

    pp_mean_hmc_10 = np.mean(means_hmc_10)
    # Plot overlapping distributions with HMC

    X = X_10
    y = y_10
    X_star = X_star_10
    f_star = f_star_10

    means_hmc = means_hmc_10
    std_hmc = std_hmc_10

    lower_hmc  = lower_hmc_10
    upper_hmc  = upper_hmc_10

    pp_mean_hmc = pp_mean_hmc_10

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    for i in range(0,100):
          plt.fill_between(np.ravel(X_star), means_hmc.ix[i] - 2*std_hmc.ix[i],  means_hmc.ix[i] + 2*std_hmc.ix[i],
                           alpha=0.3, color='grey')
    plt.plot(X_star, means_hmc.T, alpha=0.7)
    plt.plot(X_star, f_star, "black", lw=2.0, linestyle='dashed', label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3)
    plt.title('Posterior means and 95% CI per ' + r'$\theta_{i}$')
    #plt.ylim(-9,9)

    plt.subplot(122)
    plt.title('With estimated 95% CI ' + 'for mixture dist.')
    plt.fill_between(np.ravel(X_star), lower_hmc, upper_hmc, alpha=0.3, color='grey', label='95% CI')
    plt.plot(X_star, f_star, "black", lw=2.0, linestyle='dashed', label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3)
    plt.plot(X_star, pp_mean_hmc, color='blue', alpha=0.8)

    # Plot
