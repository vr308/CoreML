#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:07:13 2019

@author: vidhi
"""

import pymc3 as pm
import theano.tensor as tt
from sampled import sampled
import pandas as pd
import numpy as np
from matplotlib import cm
import time as time
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import seaborn as sns
from theano.tensor.nlinalg import matrix_inverse
import csv
import scipy.stats as st
import warnings
import posterior_analysis as pa
import synthetic_data_generation as sdg
warnings.filterwarnings("ignore")

# Plot with function draws from a single lengthscale 

lengthscale = 2.0
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(0, 10, 200)[:,None]
K = cov(X).eval()

plt.figure(figsize=(4,6))
plt.subplot(211)
plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=5).T, color='g');
plt.title("Fixed lengthscale", fontsize='x-small');
plt.ylabel("y");
plt.xlabel("X");

plt.subplot(212)
for i in np.arange(5):
      lengthscale = np.random.gamma(1,1)
      cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)
      K = cov(X).eval()
      plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T);
plt.title("Prior over lengthscales", fontsize='x-small');
plt.ylabel("y");
plt.xlabel("X");


# Plot with lml surface and two modes with predictions 

plt.figure(figsize=(14,4))

rng = np.random.RandomState(0)
X = rng.uniform(0, 5, 20)[:, np.newaxis]
y = 0.5 * np.sin(10 * X[:, 0]) + rng.normal(0, 0.2, X.shape[0])

# First run
plt.subplot(133)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp1 = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp1.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'r', lw=1, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'k', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)
#plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gp.kernel_,
#             gp.log_marginal_likelihood(gp.kernel_.theta)))
lml = np.round(gp1.log_marginal_likelihood(gp1.kernel_.theta), 3)
plt.title('LML:' + str(lml), fontsize='x-small')
plt.tight_layout()

# Second run
plt.subplot(132)

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp2 = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp2.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'r', lw=1, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'k', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)
#plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gp.kernel_,
#             gp.log_marginal_likelihood(gp.kernel_.theta)))
lml = np.round(gp2.log_marginal_likelihood(gp2.kernel_.theta), 3)
plt.title('LML:' + str(lml), fontsize='x-small')
plt.tight_layout()

#Third run
plt.subplot(131)
kernel = 1.0 * RBF(length_scale=300, length_scale_bounds=(1e-5, 1e3)) \
    + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-10, 1e+1))
gp3 = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp3.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'r', lw=1, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'k', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)
#plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gp.kernel_,
#             gp.log_marginal_likelihood(gp.kernel_.theta)))
lml = np.round(gp3.log_marginal_likelihood(gp3.kernel_.theta), 3)
plt.title('LML:' + str(lml), fontsize='x-small')
plt.tight_layout()
print(gp3.kernel_)

# Plot LML landscape
plt.subplot(131)
theta0 = np.logspace(-2, 4, 70)
theta1 = np.logspace(-3, 0, 70)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
LML = np.array(LML).T

vmin, vmax = (-LML).min(), (-LML).max()
vmax = 50
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 80), decimals=2)
plt.contourf(Theta0, Theta1, -LML,
            levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.coolwarm, alpha=1, extend='both')
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.scatter(gp1.kernel_.k1.k2.length_scale, gp1.kernel_.k2.noise_level, marker='x', color='r')
plt.scatter(gp2.kernel_.k1.k2.length_scale, gp2.kernel_.k2.noise_level, marker='x', color='r')
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("Negative Log-marginal-likelihood", fontsize='x-small')
plt.tight_layout()


# Compression of lengthscale posterior in fixed domain data

def load_datasets(path, n_train, n_star):
      
       n_test = n_star - n_train
       X = np.asarray(pd.read_csv(path + 'X_' + str(n_train) + '.csv', header=None)).reshape(n_train,1)
       y = np.asarray(pd.read_csv(path + 'y_' + str(n_train) + '.csv', header=None)).reshape(n_train,)
       X_star = np.asarray(pd.read_csv(path + 'X_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,1)
       f_star = np.asarray(pd.read_csv(path + 'f_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,)
       return X, y, X_star, f_star
 
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
       
data_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/Unif/snr_2/'

n_star = 500

X_10, y_10, X_star_10, f_star_10 = load_datasets(data_path, 10, n_star)
X_20, y_20, X_star_20, f_star_20 = load_datasets(data_path, 20, n_star)
X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40, n_star)
X_80, y_80, X_star_80, f_star_80 = load_datasets(data_path, 80, n_star)
X_100, y_100, X_star_100, f_star_100 = load_datasets(data_path, 100, n_star)
X_120, y_120, X_star_120, f_star_120 = load_datasets(data_path, 120, n_star)
X_200, y_200, X_star_200, f_star_200 = load_datasets(data_path, 200, n_star)

#-----------------------------Full Bayesian HMC--------------------------------------- 
            
# Generating traces with NUTS
      
with generative_model(X=X_10, y=y_10): trace_hmc_10 = pm.sample(draws=1000, tune=1000)
with generative_model(X=X_20, y=y_20): trace_hmc_20 = pm.sample(draws=1000, tune=1000)
with generative_model(X=X_40, y=y_40): trace_hmc_40 = pm.sample(draws=1000)
with generative_model(X=X_80, y=y_80): trace_hmc_80 = pm.sample(draws=1000)
with generative_model(X=X_100, y=y_100): trace_hmc_100 = pm.sample(draws=1000)
with generative_model(X=X_120, y=y_120): trace_hmc_120 = pm.sample(draws=1000)
with generative_model(X=X_200, y=y_200): trace_hmc_200 = pm.sample(draws=1000)


plt.figure()
plt.hist(np.log(trace_hmc_10['ls']), bins=100, normed=True, label='10', alpha=0.8)
plt.hist(np.log(trace_hmc_20['ls']), bins=100, normed=True, label='20', alpha=0.8)
plt.hist(np.log(trace_hmc_40['ls']), bins=100, normed=True, label='40', alpha=0.8)
plt.hist(np.log(trace_hmc_80['ls']), bins=100, normed=True, label='80', alpha=0.6)
plt.hist(np.log(trace_hmc_100['ls']), bins=100, normed=True, label='100', alpha=0.6)
plt.hist(np.log(trace_hmc_120['ls']), bins=100, normed=True, label='120', alpha=0.6)
#plt.hist(np.log(trace_hmc_200['ls']), bins=100, normed=True, label='200', alpha=0.6)
plt.title('Log Lengthscale Posterior', fontsize='x-small')
plt.axvline(np.log(20), color='r', label='True')
plt.legend()




# 1d full example - 3 rows of plots 
#Row 1 -  ML, HMC, FR predictions 
#Row 2 - Marginal posteriors with ml2 lines
#Row 3 bivariate plots

X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40, n_star)


X = X_80
y = y_80
X_star = X_star_80
f_star = f_star_80

# Generate some data from a SE-Kernel 

def plot_gp(X_star, f_star, X, y, post_mean, lower, upper, post_samples, title, clr):
    
    if ~post_samples.empty:
          plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
    plt.plot(X_star, f_star, "k", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3, alpha=0.5)
    plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), lower, 
                     upper, alpha=0.3, color=clr,
                     label='95% CI')
    plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')


# ML-II


    kernel = Ck(50, (1e-10, 1e3)) * RBF(1, length_scale_bounds=(1e-5, 1e3)) + WhiteKernel(1.0, noise_level_bounds=(1e-10,1000))
          
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
  
    # Fit to data 
    gpr.fit(X, y)        
    ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
    post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
    post_std = np.sqrt(np.diag(post_cov))
    post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
    post_samples = np.random.multivariate_normal(post_mean, post_cov , 10)
    rmse_ = pa.rmse(post_mean, f_star)
    lpd_ = -pa.log_predictive_density(f_star, post_mean, post_std)
    title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + '-LPD: ' + str(lpd_)     
    #ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
    #'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
    
    lower = post_mean - 1.96*post_std
    upper = post_mean + 1.96*post_std
    
    


# HMC

results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/1d/Unif/snr_2/'
write_posterior_predictive_samples(trace_hmc_80, 20, X, y, X_star, results_path, 'hmc_2') 

post_means_hmc, post_stds_hmc = load_post_samples(results_path + 'means_hmc_80.csv', results_path + 'std_hmc_80.csv')
post_means_hmc2, post_stds_hmc2 = load_post_samples(results_path + 'means_hmc_2_80.csv', results_path + 'std_hmc_2_80.csv')

lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(post_means_hmc, post_stds_hmc)
lower_hmc2, upper_hmc2 = pa.get_posterior_predictive_uncertainty_intervals(post_means_hmc2, post_stds_hmc2)

pp_mean_hmc = np.mean(post_means_hmc)

rmse_hmc = pa.rmse(pp_mean_hmc, f_star)

lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(f_star, post_means_hmc, post_stds_hmc)

title_hmc = 'RMSE: ' + str(rmse_hmc) + '\n' + '-LPD: ' + str(-lpd_hmc)
       

# FR-VI

with generative_model(X=X_80, y=y_80): 
      
      fr = pm.FullRankADVI()
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=20000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)


write_posterior_predictive_samples(trace_fr, 80, X, y, X_star, results_path, 'fr') 
sample_means_fr = pd.read_csv(results_path + 'means_fr_80.csv')
sample_stds_fr = pd.read_csv(results_path + 'std_fr_80.csv')
      
mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
rmse_fr = pa.rmse(mu_fr, f_star)
lppd_fr, lpd_fr = pa.log_predictive_mixture_density(f_star, sample_means_fr, sample_stds_fr)
                 

plt.figure()
plt.plot(X_star, post_means_hmc.T, color='b', alpha=0.2, label='HMC means')
plt.plot(X_star, sample_means_fr.T, color='coral', alpha=0.4, label='VI means')

 
 # Convergence 
   
ad.convergence_report(tracker_fr, fr.hist, varnames, 'Full-Rank Convergence')

plt.figure(figsize=(12,10))
plt.subplot(231)
#plot_gp(X_star, f_star, X, y, post_mean, lower, upper, pd.Series([]), '', 'r')
plt.subplot(232)
plot_gp(X_star, f_star, X, y, pp_mean_hmc, lower_hmc2, upper_hmc2, post_means_hmc, '', 'b')
plt.subplot(233)
plot_gp(X_star, f_star, X, y, mu_fr, lower_fr, upper_fr, sample_means_fr, title, 'g')

# Traceplots


# 2d full example - sample as above




# FR-VI compared to Taylor VI

 