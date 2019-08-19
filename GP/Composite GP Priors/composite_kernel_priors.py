#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:09:58 2019

@author: vidhi

Composite Priors - Scratchpad work

"""

import sys

sys.path.append('/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP')

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import posterior_analysis as pa
import scipy.stats as st
import csv
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP')


def generate_gp_latent(X_star, mean, cov):
    
    return np.random.multivariate_normal(mean(X_star).eval(), cov=cov(X_star, X_star).eval())

def generate_gp_training(X_all, f_all, n_train, noise_sd, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.5*st.norm.pdf(X_all, 2, 0.7) + 0.5*st.norm.pdf(X_all, 15.0,1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
     
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    X = X_all[train_index]
    f = f_all[train_index]
    X_star = X_all[test_index]
    f_star = f_all[test_index]
    y = f + np.random.normal(0, scale=noise_sd, size=n_train)
    return X, y, X_star, f_star, f

def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')
    
    
def write_posterior_predictive_samples(trace, thin_factor, X, y, X_star, path, method, gp):
      
      means_file = path + 'means_' + method + '.csv'
      std_file = path + 'std_' + method + '.csv'
      #trace_file = path + 'trace_' + method + '_' + str(len(X)) + '.csv'
          
      means_writer = csv.writer(open(means_file, 'w')) 
      std_writer = csv.writer(open(std_file, 'w'))
      #trace_writer = csv.writer(open(trace_file, 'w'))
      
      means_writer.writerow(X_star.flatten())
      std_writer.writerow(X_star.flatten())
      #trace_writer.writerow(varnames + ['lml'])
      
      for i in np.arange(len(trace))[::thin_factor]:
            
            print('Predicting ' + str(i))
            post_mean, post_var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            post_std = np.sqrt(post_var)
            #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), trace[i])
            #post_mean, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
            #marginal_likelihood = compute_log_marginal_likelihood(K_noise, y)
            #mu, var = pm.gp.Marginal.predict(Xnew=X_star, point=trace[i], pred_noise=False, diag=True)
            #std = np.sqrt(var)
            #list_point = [trace[i]['sig_sd'], trace[i]['ls'], trace[i]['noise_sd'], marginal_likelihood]
            
            print('Writing out ' + str(i) + ' predictions')
            means_writer.writerow(np.round(post_mean, 3))
            std_writer.writerow(np.round(post_std, 3))
            #trace_writer.writerow(np.round(list_point, 3))
            

def plot_lml_surface_3way(gpr, sig_sd, lengthscale, noise_sd):
    
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    l_log = np.logspace(-5, 5, 100)
    noise_log  = np.logspace(-6, 6, 100)
    l_log_mesh, noise_log_mesh = np.meshgrid(l_log, noise_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(sig_sd), l_log_mesh[i, j], np.square(noise_log_mesh[i, j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, noise_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, noise_sd, 'rx')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise sd")
    
    plt.subplot(132)
    l_log = np.logspace(-5, 5, 100)
    signal_log  = np.logspace(-5, 5, 100)
    l_log_mesh, signal_log_mesh = np.meshgrid(l_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), l_log_mesh[i, j], np.square(noise_sd)]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Signal sd")
    
    plt.subplot(133)
    noise_log = np.logspace(-5, 4, 100)
    signal_log  = np.logspace(-5, 2, 100)
    noise_log_mesh, signal_log_mesh = np.meshgrid(noise_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), lengthscale, np.square(noise_log_mesh[i,j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(noise_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(noise_sd, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise sd")
    plt.ylabel("Signal sd")    
    plt.suptitle('LML Surface ' + '\n' + str(gpr.kernel_), fontsize='small')
            
if __name__ == "__main__":


      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Composite GP Priors/'
      
      # Sum of SE Kernels in 1D
         
      # Simulate some data 
      
      mean = pm.gp.mean.Zero()
      X_all = np.linspace(0,100,300)[:, None]
      
      sig_sd = [10, 50]
      ls = [10, 100]
      noise_sd = 10
      
      #snr = sig_sd/noise_sd
      
      cov = pm.gp.cov.Constant(sig_sd[0]**2)*pm.gp.cov.ExpQuad(1, ls=ls[0]) + pm.gp.cov.Constant(sig_sd[1]**2)*pm.gp.cov.ExpQuad(1, ls=ls[1])
      
      f_all = generate_gp_latent(X_all, mean, cov)
      
      uniform = True
      n_train = 40
      X, y, X_star, f_star, f = generate_gp_training(X_all, f_all, n_train, noise_sd, uniform)
      
      true_deltas = {'ls1': ls[0], 'ls2': ls[1], 'sig_sd1': sig_sd[0], 'sig_sd2': sig_sd[1], 'noise_sd': noise_sd}
      
      # Plot
      
      plot_noisy_data(X, y, X_star, f_star, '')
      
      # Priors
      
      priors = {'uniform': pm.Uniform.dist(lower=-10, upper=10), 
                'normal': pm.Normal.dist(mu=0, sd=2), 
                'inv-gamma': pm.InverseGamma.dist(alpha=5, beta=2),
                'half-normal': pm.HalfNormal.dist(sd=1), 
                'gamma': pm.Gamma.dist(alpha=5, beta=2)}
      

      with pm.Model() as sum_se:
            
               # prior on lengthscale 
               #log_ls1 = pm.Uniform('log_ls1', lower=-10, upper=10)
               #log_ls2 = pm.Uniform('log_ls2', lower=-10, upper=10)
               
               #log_ls1 = pm.Normal('log_ls1', mu=0, sd=2)
               #log_ls2 = pm.Normal('log_ls2', mu=0, sd=2)

               #ls1 = pm.Deterministic('ls1', tt.exp(log_ls1))
               #ls2 = pm.Deterministic('ls2', tt.exp(log_ls2))
               
               # Lengthscale prior
               
               ls1 = pm.InverseGamma('ls1', alpha=5, beta=2)
               ls2 = pm.InverseGamma('ls2', alpha=5, beta=2)

               #prior on noise variance
               #log_n =  pm.Uniform('log_n', lower=-10, upper=10)

               log_n = pm.Normal('log_n', mu=0, sd=2)
               noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))
               
               #prior on signal variance
               #log_s1 = pm.Uniform('log_s1', lower=-10, upper=10)
               #log_s2 = pm.Uniform('log_s2', lower=-10, upper=10)
               log_s1 = pm.Normal('log_s1', mu=0, sd=2)
               log_s2 = pm.Normal('log_s2', mu=0, sd=2)

               sig_sd1 = pm.Deterministic('sig_sd1', tt.exp(log_s1))
               sig_sd2 = pm.Deterministic('sig_sd2', tt.exp(log_s2))
            
               # Specify the covariance function.
               cov_func = pm.gp.cov.Constant(sig_sd1**2)*pm.gp.cov.ExpQuad(1, ls=ls1) + pm.gp.cov.Constant(sig_sd2**2)*pm.gp.cov.ExpQuad(1, ls=ls2)
               # Specify the GP. 
               gp = pm.gp.Marginal(cov_func=cov_func)
              
               trace_prior = pm.sample()
                  
               # Marginal Likelihood
               y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)
              
               trace_posterior = pm.sample()
            
      
      prior_tag = 'inv-gamma' #/ 'normal' / 'half-norm' / 'gamma'
      
      varnames = ['ls1','sig_sd1', 'ls2', 'sig_sd2', 'noise_sd']
            
      # Save traceplot  
      
      pa.traceplots(trace_posterior, varnames=varnames, deltas=true_deltas, sep_idx=5, combined=True)
      
      # Save prior / posterior plots
      
      plt.figure(figsize=(14,8))
      for i in np.arange(len(varnames)):
            plt.subplot(2,3, i+1)
            plt.hist(trace_prior[varnames[i]], bins=1000, alpha=0.4, normed=True, label='Prior')
            plt.hist(trace_posterior[varnames[i]], bins=1000, alpha=0.7, normed=True, label='Posterior')
            plt.axvline(x=true_deltas[varnames[i]], ymin=0, color='r')
            plt.legend(fontsize='x-small')
      plt.suptitle('Posterior Contraction - Normal(0,2) Priors', fontsize='small')
      
      # Save trace_summary
      
      trace_summary = pm.summary(trace_posterior).ix[varnames]
      trace_summary.to_csv(results_path + 'trace_summary_' + prior_tag + '_10.csv')
      
      # Save autocorrelation
      
      pm.autocorrplot(trace_posterior, varnames)
      
      # Save predictions

      pa.write_posterior_predictive_samples(trace_posterior, 10, X, y, X_star, results_path + 'predictions/', 'hmc_' + prior_tag + '_prior', gp)
      
      sample_means = pd.read_csv(results_path  + 'predictions/' + 'means_hmc_' + prior_tag + '_prior.csv')
      sample_stds = pd.read_csv(results_path  + 'predictions/' + 'std_hmc_' + prior_tag + '_prior.csv')
      
      mean_hmc = pa.get_posterior_predictive_mean(sample_means)
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds)
      
      plt.figure()
      plt.plot(X_star, f_star, "k", lw=2, label="True f")
      plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
      plt.plot(X_star, sample_means.T, color='grey', alpha=0.2)
      plt.plot(X_star, np.mean(sample_means), color='b', label='Integrated Mean')
      plt.fill_between(X_star.flatten(), lower_hmc, upper_hmc, color='b', alpha=0.2)
      plt.title('Inv-Gamma Priors ($K_{se1} + K_{se2}$)', fontsize='x-small')
      
      # Save ppc check













log_raw1 = st.uniform.rvs(-10, 20, 10000)
log_raw2 = st.uniform.rvs(-7, 14, 10000)

log_norm1 = st.norm.rvs(0,2, 10000)
log_norm2 = st.norm.rvs(0,1, 10000)

fig = plt.figure(figsize=(14,8))

plt.subplot(221)
plt.hist(log_raw1, bins=100, normed=True, alpha=0.2, color='b')
plt.hist(log_raw2, bins=100, normed=True, alpha=0.2, color='r')

plt.subplot(222)
plt.hist(np.exp(log_raw1), bins=1000,  normed=True, alpha=0.3, color='b')
plt.hist(np.exp(log_raw2), bins=100,  normed=True, alpha=0.3, color='r')

plt.subplot(223)
plt.hist(log_norm1, bins=100, normed=True, alpha=0.2, color='b')
plt.hist(log_norm2, bins=100, normed=True, alpha=0.2, color='r')

plt.subplot(224)
plt.hist(np.exp(log_norm1), bins=1000, normed=True,  alpha=0.3, color='b')
plt.hist(np.exp(log_norm2), bins=100,  normed=True, alpha=0.3, color='r')

# Product of SE kernels in 1D




# SE-ARD Kernels






