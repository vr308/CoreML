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

def get_kernel_matrix_blocks(X, X_star, n_train, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K = cov(X)
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          K_noise = K + np.square(point['noise_sd'])*tt.eye(n_train)
          K_inv = matrix_inverse(K_noise)
          return K, K_s, K_ss, K_noise, K_inv

def analytical_gp_predict_latent(y, K, K_s, K_ss, K_noise, K_inv):
    
          L = np.linalg.cholesky(K_noise.eval())
          alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
          v = np.linalg.solve(L, K_s.eval())
          post_mean = np.dot(K_s.eval().T, alpha)
          post_cov = K_ss.eval() - v.T.dot(v)
          post_std = np.sqrt(np.diag(post_cov))
          #post_std_y = np.sqrt(np.diag(post_cov))
          return post_mean,  post_std

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

def load_post_samples(means_file, std_file):
      
      return pd.read_csv(means_file, sep=',', header=0), pd.read_csv(std_file, sep=',', header=0)

def write_posterior_predictive_samples(trace, thin_factor, X, y, X_star, path, method):
      
      means_file = path + 'means_' + method + '_' + str(len(X)) + '.csv'
      std_file = path + 'std_' + method + '_' + str(len(X)) + '.csv'
      trace_file = path + 'trace_' + method + '_' + str(len(X)) + '.csv'
    
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
            

# Plot with function draws from a single lengthscale 

lengthscale = 1.0
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(0, 10, 200)[:,None]
K = cov(X).eval()

plt.figure(figsize=(4,4))
plt.subplot(211)
plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=5).T, color='g');
plt.title("Fixed lengthscale", fontsize='x-small');
plt.ylabel("y", fontsize='x-small');
plt.xlabel("X", fontsize='x-small');
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')


plt.subplot(212)
for i in np.arange(5):
      lengthscale = np.random.gamma(2,2)
      cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)
      K = cov(X).eval()
      plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T);
plt.title("Prior over lengthscales", fontsize='x-small');
plt.ylabel("y", fontsize='x-small');
plt.xlabel("X", fontsize='x-small');
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.tight_layout()

# Plot with lml surface and two modes with predictions 


rng = np.random.RandomState(0)
X = rng.uniform(0, 10, 20)[:, np.newaxis]
f = np.exp(np.cos((0.4 - X[:,0])))
y = f + rng.normal(0, 0.5, X.shape[0])

X_star = np.linspace(0, 10, 100)
f_star = np.exp(np.cos((0.4 - X_star)))

plt.plot(X_star, f_star, 'k', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)


plt.figure(figsize=(8,2))

# First run
plt.subplot(133)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e4)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp1 = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)

y_mean, y_cov = gp1.predict(X_star[:, np.newaxis], return_cov=True)
plt.plot(X_star, y_mean, 'r', lw=1, zorder=9)
plt.fill_between(X_star, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2, color='k')
plt.plot(X_star, f_star, 'k', lw=2, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)
#plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gp.kernel_,
#             gp.log_marginal_likelihood(gp.kernel_.theta)))
lml = np.round(gp1.log_marginal_likelihood(gp1.kernel_.theta), 3)
plt.title('LML:' + str(lml), fontsize='x-small')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.tight_layout()

# Second run
plt.subplot(132)

kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e+1))
gp2 = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp2.predict(X_star[:, np.newaxis], return_cov=True)
plt.plot(X_star, y_mean, 'r', lw=1, zorder=9)
plt.fill_between(X_star, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2, color='k')
plt.plot(X_star, f_star, 'k', lw=2, zorder=9)
plt.scatter(X[:, 0], y, c='k', s=10)
#plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#          % (kernel, gp.kernel_,
#             gp.log_marginal_likelihood(gp.kernel_.theta)))
lml = np.round(gp2.log_marginal_likelihood(gp2.kernel_.theta), 3)
plt.title('LML:' + str(lml), fontsize='x-small')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')

plt.tight_layout()


# Plot LML landscape
plt.subplot(131)
theta0 = np.logspace(-2, 4, 70)
theta1 = np.logspace(-5, 1, 70)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[gp1.log_marginal_likelihood(np.log([gp1.kernel_.k1.k1.constant_value, Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
LML = np.array(LML).T

vmin, vmax = (-LML).min(), (-LML).max()
vmax = 50
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=3)
plt.contourf(Theta0, Theta1, -LML,
            levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.coolwarm, alpha=1, extend='both')
plt.xscale("log")
plt.yscale("log")
plt.scatter(gp1.kernel_.k1.k2.length_scale, gp1.kernel_.k2.noise_level, marker='x', color='r')
plt.scatter(gp2.kernel_.k1.k2.length_scale, gp2.kernel_.k2.noise_level, marker='x', color='r')
plt.xlabel("Length-scale", fontsize='x-small')
plt.ylabel("Noise-level", fontsize='x-small')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.title("Negative Log-marginal-likelihood", fontsize='x-small')

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
       
data_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/Unif/snr_6/'

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


plt.figure(figsize=(4,4))
plt.hist(np.log(trace_hmc_10['ls']), bins=100, normed=True, label='10', alpha=0.8)
plt.hist(np.log(trace_hmc_20['ls']), bins=100, normed=True, label='20', alpha=0.8)
plt.hist(np.log(trace_hmc_40['ls']), bins=100, normed=True, label='40', alpha=0.8)
plt.hist(np.log(trace_hmc_80['ls']), bins=100, normed=True, label='80', alpha=0.6)
plt.hist(np.log(trace_hmc_100['ls']), bins=100, normed=True, label='100', alpha=0.6)
plt.hist(np.log(trace_hmc_120['ls']), bins=100, normed=True, label='120', alpha=0.6)
#plt.hist(np.log(trace_hmc_200['ls']), bins=100, normed=True, label='200', alpha=0.6)
plt.title('Marginal Posterior', fontsize='x-small')
plt.ylabel('Normalized Counts', fontsize='x-small')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.axvline(np.log(18), color='r', label='True')
plt.xlabel('Log(length-scale)', fontsize='x-small')
plt.legend(fontsize='x-small')

# 1d full example - 3 rows of plots 
#Row 1 -  ML, HMC, FR predictions 
#Row 2 - Marginal posteriors with ml2 lines
#Row 3 bivariate plots

varnames = ['sig_sd', 'ls', 'noise_sd']
varnames_log = ['log_s', 'log_ls', 'log_n']

data_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/NUnif/snr_10/'


X_10, y_10, X_star_10, f_star_10 = load_datasets(data_path, 10, n_star)
X_20, y_20, X_star_20, f_star_20 = load_datasets(data_path, 20, n_star)
X_60, y_60, X_star_60, f_star_60 = load_datasets(data_path, 60, n_star)
X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40, n_star)
X_80, y_80, X_star_80, f_star_80 = load_datasets(data_path, 80, n_star)

X = X_20
y = y_20
X_star = X_star_20
f_star = f_star_20

# Generate some data from a SE-Kernel 

def plot_gp(X_star, f_star, X, y, post_mean, lower, upper, post_samples, title, clr):
    
    if post_samples.empty == False:
          plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
    plt.plot(X_star, f_star, "k", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3, alpha=0.5)
    plt.plot(X_star, post_mean, color=clr, lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), lower, 
                     upper, alpha=0.3, color=clr,
                     label='95% CI')
    #plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')


# ML-II

    kernel = Ck(50, (1e-10, 1e3)) * RBF(1, length_scale_bounds=(1e-5, 1e3)) + WhiteKernel(1.0, noise_level_bounds=(1e-10,1000))
          
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
  
    # Fit to data 
    gpr.fit(X, y)        
    ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
    post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
    post_std = np.sqrt(np.diag(post_cov))
    post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
    post_samples = np.random.multivariate_normal(post_mean, post_cov - np.eye(len(X_star))*ml_deltas[2], 30)
    post_samples=pd.Series()
    rmse_ = pa.rmse(post_mean, f_star)
    lpd_ = -pa.log_predictive_density(f_star, post_mean, post_std)
    title = 'ML-II' + '\n' + 'RMSE: ' + str(np.round(rmse_,3)) + '\n' + '-LPD: ' + str(lpd_)     
    ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
    'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
    
    lower = post_mean - 1.9*post_std
    upper = post_mean + 1.9*post_std
    
    plot_gp(X_star, f_star, X, y, post_mean, lower, upper, pd.Series(), title, 'r')


# HMC
    with pm.Model() as syn_model:
          
       # prior on lengthscale 
       #log_ls = pm.Normal('log_ls',mu = 0, sd = 1.5)
       #ls = pm.Deterministic('ls', tt.exp(log_ls))
       ls = pm.Gamma('ls', 2, 0.1)
       
        #prior on noise variance
       log_n = pm.Normal('log_n', mu = 0 , sd = 3)
       noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))
         
       #prior on signal variance
       log_s = pm.Normal('log_s', mu=0, sd = 3)
       sig_sd = pm.Deterministic('sig_sd', tt.exp(log_s))
       #sig_sd = pm.InverseGamma('sig_sd', 4, 4)
       
       # Specify the covariance function.
       cov_func = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
       
       #Prior
       trace_prior = pm.sample(draws=1000)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)
       
       trace_hmc_20 = pm.sample(draws=1000)
       
       #####

 results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/1d/NUnif/snr_10/'

 pa.plot_prior_posterior_plots(trace_prior, trace_hmc_20, varnames, ml_deltas_dict, '')

 write_posterior_predictive_samples(trace_hmc_20, 10, X, y, X_star, results_path, 'hmc') 
 
 post_means_hmc, post_stds_hmc = load_post_samples(results_path + 'means_hmc_20.csv', results_path + 'std_hmc_20.csv')

 lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(post_means_hmc, post_stds_hmc)

 pp_mean_hmc = np.mean(post_means_hmc)

 rmse_hmc = pa.rmse(pp_mean_hmc, f_star)

 lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(f_star, post_means_hmc, post_stds_hmc)

 title_hmc = 'HMC' + '\n' + 'RMSE: ' + str(np.round(rmse_fr,3)) + '\n' + '-LPD: ' + str(-lpd_fr)
 
# FR-VI

    with syn_model: 
      
      fr = pm.FullRankADVI()
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=20000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)

write_posterior_predictive_samples(trace_fr, 40, X, y, X_star, results_path, 'fr') 
sample_means_fr = pd.read_csv(results_path + 'means_fr_20.csv')
sample_stds_fr = pd.read_csv(results_path + 'std_fr_20.csv')
      
mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
rmse_fr = pa.rmse(mu_fr, f_star)
lppd_fr, lpd_fr = pa.log_predictive_mixture_density(f_star, sample_means_fr, sample_stds_fr)
                 
title_fr = 'Full Rank VI' + '\n' + 'RMSE: ' + str(np.round(rmse_hmc,3)) + '\n' + '-LPD: ' + str(-lpd_hmc)

plt.figure(figsize=(10,8))
plt.subplot(231)
plot_gp(X_star, f_star, X, y, post_mean, lower, upper, pd.Series(), title, 'r')
plt.subplot(232)
plot_gp(X_star, f_star, X, y, pp_mean_hmc, lower_hmc, upper_hmc, pd.Series(), title_hmc, 'b')
plt.subplot(233)
plot_gp(X_star, f_star, X, y, mu_fr, lower_fr, upper_fr, pd.Series(), title_fr, 'g')
plt.subplot(234)
plt.plot(X_star, post_samples.T, color='r', alpha=0.4, label='ML')
plt.plot(X_star, f_star, "k", lw=1.4, label="True f",alpha=0.7);
plt.ylim(-80,+40)
plt.subplot(235)
plt.plot(X_star, post_means_hmc.T, color='b', alpha=0.2, label='HMC means')
plt.plot(X_star, f_star, "k", lw=1.4, label="True f",alpha=0.7);
plt.ylim(-80,+40)
plt.subplot(236)
plt.plot(X_star, sample_means_fr.T, color='g', alpha=0.3, label='VI means')
plt.plot(X_star, f_star, "k", lw=1.4, label="True f",alpha=0.7);
plt.ylim(-80,+40)


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


# FR-VI compared to Taylor VI

import autograd.numpy as np
from autograd import elementwise_grad, grad, jacobian

def kernel(theta, X1, X2):
        
     # se +  sexper + rq + se 
     
     s_1 = theta[0]
     ls_2 = theta[1]
     
     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
     dist = np.abs(np.sum(X1,1).reshape(-1,1) - np.sum(X2,1))
     sk = s_1**2 * np.exp(-0.5 / ls_2**2 * sqdist)
    
     return sk

def gp_mean(theta, X, y, X_star):
      
     n_4 = theta[2]
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk2 = n_4**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk2*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

def gp_cov(theta, X, y, X_star):
      
     n_4 = theta[2]
   
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk2 = n_4**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk2*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     K_ss = kernel(theta, X_star, X_star)
     return K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), K_s)
            
   
def get_vi_analytical(X, y, X_star, dh, d2h, d2g, theta, mu_theta, cov_theta, results_path):
                  
    #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), theta)      
    #pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
    #pred_vi_var =  np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))
    
    pred_g_mean = gp_mean(theta, X, y, X_star)
    pred_g_var = np.diag(gp_cov(theta, X, y, X_star))

    pred_ng_mean = []
    pred_ng_var = []
    
    # To fix this 
    
    #pred_ng_mean = pred_g_mean + 0.5*np.trace(np.matmul(d2h(theta, X, y, X_star), np.array(cov_theta)))
    #pred_ng_var = pred_vi_var + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta))

    for i in np.arange(len(X_star)): # To vectorize this loop
          
          print(i)
          x_star = X_star[i].reshape(1,1)

          pred_ng_mean.append(pred_g_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta))))
          pred_ng_var.append(pred_g_var[i] + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta)))

    np.savetxt(fname=results_path  + 'mu_taylor.csv', X=pred_ng_mean, delimiter=',', header='')   
    np.savetxt(fname=results_path + 'std_taylor.csv', X=np.sqrt(pred_ng_var), delimiter=',', header='')   

    return pred_ng_mean, np.sqrt(pred_ng_var)

# Load some data 
    
data_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/Unif/snr_10/'

X_60, y_60, X_star_60, f_star_60 = load_datasets(data_path, 60, n_star)

X = X_60
y = y_60
X_star = X_star_60
    
with pm.Model() as vi_test_model:
          
       # prior on lengthscale 
       log_ls = pm.Normal('log_ls',mu = 0, sd = 2)
       ls = pm.Deterministic('ls', tt.exp(log_ls))
       #ls = pm.Gamma('ls', 2, 0.1)
       
        #prior on noise variance
       log_n = pm.Normal('log_n', mu = 0 , sd = 3)
       noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))
         
       #prior on signal variance
       log_s = pm.Normal('log_s', mu=0, sd = 3)
       sig_sd = pm.Deterministic('sig_sd', tt.exp(log_s))
       #sig_sd = pm.InverseGamma('sig_sd', 4, 4)
       
       # Specify the covariance function.
       cov_func = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)
       
with vi_test_model:
         
      fr = pm.FullRankADVI()
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=40000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)  

ad.convergence_report(tracker_fr, fr.hist, varnames, 'Full-Rank Convergence')

write_posterior_predictive_samples(trace_fr, 100, X, y, X_star, results_path, 'fr') 
sample_means_fr = pd.read_csv(results_path + 'means_fr_40.csv')
sample_stds_fr = pd.read_csv(results_path + 'std_fr_40.csv')

mu_fr = np.mean(sample_means_fr)
# Get variational posterior map

trace_fr_df = pm.trace_to_dataframe(trace_fr)

mu_theta = pm.summary(trace_fr).ix[varnames]['mean']
theta = pm.summary(trace_fr).ix[varnames]['mean'].values
cov_theta = pa.get_empirical_covariance(trace_fr_df, varnames)


dh = elementwise_grad(gp_mean)
d2h = jacobian(dh)
dg = grad(gp_cov)
d2g = jacobian(dg) 

mu_taylor, std_taylor = get_vi_analytical(X_40, y_40, X_star_40, dh, d2h, d2g, theta, mu_theta, cov_theta, results_path)

plt.figure(figsize=(4,4))
plt.plot(X_star_40, f_star_40, 'k-', label='True')
plt.plot(X, y, 'ko', markersize=2)
plt.fill_between(X_star_40.ravel(), lower_fr, upper_fr, color='g', alpha=0.4)
plt.plot(X_star_40, mu_fr, 'g', label='Sampling PP')
plt.plot(X_star_40, mu_taylor, 'dodgerblue')
 